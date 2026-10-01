"""Post-train Laya in MLX (Apple silicon). Same objective as train.py (PyTorch reference):

  soft cross-entropy on the option-marker logits
+ laya's RLCD term (REINFORCE on Gaussian-perturbed logits, strictly proper reward)
+ act-head loss: target mass on the predicted candidate; uniform targets escalate
+ grouped consistency, uncertainty, high-confidence-error and selective-risk losses
  enabled by configs/continue-mlx.args, shared with PyTorch

Token embeddings are frozen (see train.py for why). The existing action head is preserved unless explicitly reset.
"""
import argparse
import json
import math
import os
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import torch
from mlx.utils import tree_flatten

from .mlx_model import export_laya, load_laya
from .model_io import DEFAULT_BASE, load_tokenizer
from .train import collate, token_batches
from .training_recipe import add_recipe_arguments, validate_recipe, grouped_batches, recipe_metadata

QTYPE_SCORE = 1


def aligned_support(p, qtype, mask, support_index, support_flip):
    """Map a view to its group's reference proposition, including negations."""
    nopts = mx.maximum(mask.astype(mx.float32).sum(-1), 2.0)
    levels = mx.arange(p.shape[-1]).astype(mx.float32)
    positive = mx.take_along_axis(p, support_index[:, None], axis=1)[:, 0]
    score_support = (p * levels[None]).sum(-1) / (nopts - 1)
    raw_support = mx.where(qtype == QTYPE_SCORE, score_support, positive)
    return mx.where(support_flip.astype(mx.bool_), 1 - raw_support, raw_support)


def proper_reward(q, target, qtype, mask, w_sph=0.75, w_rps=1.0):
    q = q * mask
    logq = mx.maximum(mx.log(mx.maximum(q, 1e-12)), -9.21)
    r = (target * logq).sum(-1) + w_sph * (target * q).sum(-1) / mx.maximum(mx.sqrt((q * q).sum(-1)), 1e-9)
    k = mx.maximum(mask.sum(-1), 2).astype(mx.float32)
    rps = (((mx.cumsum(q, -1) - mx.cumsum(target, -1)) ** 2) * mask).sum(-1) / (k - 1)
    return r - w_rps * rps * (qtype == QTYPE_SCORE)


def loss_fn(model, ids, att, mpos, mmask, qtype, target, equiv_group, support_index, support_flip,
            sigma, group=4, w_act=0.1, w_consistency=0.0, w_uncertain=0.0,
            w_overconf=0.0, w_selective=0.0, noise=None):
    logits, act = model(ids, att, mpos, mmask, qtype, train=True)
    maskf = mmask.astype(mx.float32)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    ce = -(target * logp).sum(-1).mean()

    k = maskf.sum(-1, keepdims=True)
    eps = (mx.random.normal((group,) + logits.shape) if noise is None else noise) * sigma * maskf
    eps = (eps - eps.sum(-1, keepdims=True) / k) * maskf
    z = mx.stop_gradient(logits)[None] + eps
    q = mx.softmax(mx.where(mmask[None], z, -1e4), axis=-1)
    r = mx.stop_gradient(proper_reward(q, target[None], qtype[None], maskf[None]))
    adv = r - r.mean(0, keepdims=True)
    adv = adv / (mx.sqrt(((adv - adv.mean()) ** 2).mean()) + 1e-6)
    lp = -(((z - logits[None]) ** 2) * maskf[None]).sum(-1) / (2 * sigma ** 2)
    rl = -(adv * lp).mean()

    pred = mx.argmax(logits, -1)
    gold = mx.argmax(target, -1)
    correct = pred == gold
    levels = mx.arange(target.shape[-1]).astype(mx.float32)
    p = mx.softmax(logits, -1)
    close = mx.abs((p * levels).sum(-1) - (target * levels).sum(-1)) <= 0.5
    correct = mx.where(qtype == QTYPE_SCORE, close, correct)
    nopts = mx.maximum(maskf.sum(-1), 2.0)
    determinate = target.max(-1) >= 0.99
    uniform = mx.abs(target.max(-1) - 1 / nopts) < 1e-5
    # Soft teacher distributions still contain correctness information: use the
    # probability mass on the model's predicted option. Only genuinely uniform
    # targets mean that no answer is supported, so they train escalation.
    act_target = mx.stop_gradient(mx.where(
        uniform, 0.0, mx.take_along_axis(target, pred[:, None], axis=1)[:, 0]))
    act_logp = act - mx.logsumexp(act, axis=-1, keepdims=True)
    act_l = -(act_target * act_logp[:, 0] + (1 - act_target) * act_logp[:, 1]).mean()
    # Every view maps to one proposition-support axis. For a reversed ordinal
    # scale, or for a counterfactual state where the proposition is negated,
    # support_flip maps that view back to the group's reference proposition.
    # The target CE still keeps its original, unflipped meaning.
    support = aligned_support(p, qtype, mmask, support_index, support_flip)
    paired = ((equiv_group[:, None] == equiv_group[None, :]) &
              (equiv_group[:, None] >= 0) &
              (mx.arange(ids.shape[0])[:, None] != mx.arange(ids.shape[0])[None, :]))
    pair_weight = paired.astype(mx.float32)
    consistency = (((support[:, None] - support[None, :]) ** 2) * pair_weight).sum() / mx.maximum(pair_weight.sum(), 1)

    maxp = p.max(-1)
    uniform_weight = uniform.astype(mx.float32)
    uncertain = (((maxp - 1 / nopts) ** 2) * uniform_weight).sum() / mx.maximum(uniform_weight.sum(), 1)
    gold_mass = (p * target).sum(-1)
    det_weight = determinate.astype(mx.float32)
    overconf = (((mx.maximum(maxp - 0.8, 0) ** 2) * (1 - gold_mass)) * det_weight).sum() / mx.maximum(det_weight.sum(), 1)
    act_p = mx.softmax(act, -1)[:, 0]
    selective = ((act_p ** 2) * ((1 - mx.stop_gradient(gold_mass)) ** 2) * det_weight).sum() / mx.maximum(det_weight.sum(), 1)
    total = (ce + rl + w_act * act_l + w_consistency * consistency + w_uncertain * uncertain
             + w_overconf * overconf + w_selective * selective)
    accuracy = (correct & determinate).astype(mx.float32).sum() / mx.maximum(determinate.sum(), 1)
    return total, (ce, rl, act_l, accuracy, consistency, uncertain, overconf, selective)


def reset_act_head(model, seed=0):
    rng = np.random.default_rng(seed)
    for lyr in (model.act_head[0], model.act_head[2]):
        fan_out, fan_in = lyr.weight.shape
        a = 0.5 * math.sqrt(6.0 / (fan_in + fan_out))
        lyr.weight = mx.array(rng.uniform(-a, a, size=lyr.weight.shape).astype(np.float32))
        lyr.bias = mx.zeros(lyr.bias.shape)


def schedule(lr, total, warm):
    return optim.join_schedules([optim.linear_schedule(lr * 0.01, lr, warm),
                                 optim.cosine_decay(lr, max(1, total - warm), lr * 0.02)], [warm])


def main():
    ap = argparse.ArgumentParser(fromfile_prefix_chars='@')
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--train", default="work/train.pt")
    ap.add_argument("--out", default="work/continued-mlx")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-items", type=int, default=64)
    ap.add_argument("--lr-enc", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--sigma-start", type=float, default=0.4)
    ap.add_argument("--sigma-end", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--cache-gb", type=float, default=3.0)
    ap.add_argument("--reset-act-head", action="store_true", help="Explicitly reset the action head")
    ap.add_argument("--seed", type=int, default=20260928)
    add_recipe_arguments(ap)
    args = ap.parse_args()
    validate_recipe(args)

    # MLX keeps freed buffers in a cache that is unbounded by default; on a machine that is
    # already swapping, that cache is what pushes the process into paging (measured: 19 GB
    # footprint, 28 -> 8 items/s). Cap it.
    mx.set_cache_limit(int(args.cache_gb * 2 ** 30))
    mx.random.seed(args.seed)
    rng = random.Random(args.seed)
    tok = load_tokenizer(args.base)
    model, cfg, _ = load_laya(args.base)
    if args.reset_act_head:
        reset_act_head(model, args.seed)
    model.encoder.embeddings.tok_embeddings.freeze()
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"mlx device={mx.default_device()} trainable={n_train/1e6:.1f}M", flush=True)

    items = torch.load(args.train, map_location='cpu', weights_only=True)
    if not items:
        raise ValueError('Training data is empty')
    ep_batches = token_batches(items, args.max_tokens, args.max_items, rng)
    equiv_items = torch.load(args.equiv, map_location='cpu', weights_only=True) if args.equiv else []
    eq_batches = grouped_batches(equiv_items, args.max_tokens, args.max_items, rng) if args.equiv else []
    multiplier = args.equiv_every / (args.equiv_every - 1) if equiv_items else 1.0
    total = max(1, int(len(ep_batches) * args.epochs * multiplier))
    warm = max(1, int(total * args.warmup))
    print(f"train items={len(items)} batches/epoch={len(ep_batches)} equiv={len(equiv_items)} "
          f"equiv_batches={len(eq_batches)} steps={total}", flush=True)
    opt = optim.MultiOptimizer(
        [optim.AdamW(schedule(args.lr_enc, total, warm), betas=[0.9, 0.98], eps=1e-6, weight_decay=args.wd),
         optim.AdamW(schedule(args.lr_head, total, warm), betas=[0.9, 0.98], eps=1e-6, weight_decay=args.wd)],
        [lambda path, _: path.startswith("encoder.")])
    def objective(model, ids, att, mpos, mmask, qtype, target, equiv_group, support_index, support_flip, sigma):
        return loss_fn(model, ids, att, mpos, mmask, qtype, target, equiv_group, support_index, support_flip,
                       sigma, w_act=0.2 if equiv_items else 0.1, w_consistency=args.w_consistency,
                       w_uncertain=args.w_uncertain, w_overconf=args.w_overconf, w_selective=args.w_selective)

    vg = nn.value_and_grad(model, objective)
    state = [model.state, opt.state, mx.random.state]

    def _step(ids, att, mpos, mmask, qtype, target, equiv_group, support_index, support_flip, sigma):
        (loss, parts), grads = vg(model, ids, att, mpos, mmask, qtype, target, equiv_group,
                                  support_index, support_flip, sigma)
        grads, gnorm = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss, parts, gnorm

    step_c = mx.compile(_step, inputs=state, outputs=state) if args.compile else _step

    def step(b, sigma):
        return step_c(*b, mx.array(sigma, dtype=mx.float32))

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    t0 = time.time()
    seen = toks = 0
    agg = np.zeros(9)
    batches, bi = ep_batches, 0
    eqi = 0
    for s in range(total):
        use_equiv = bool(eq_batches) and (s + 1) % args.equiv_every == 0
        if use_equiv:
            if eqi >= len(eq_batches):
                eq_batches, eqi = grouped_batches(equiv_items, args.max_tokens, args.max_items, rng), 0
            idx = eq_batches[eqi]
            eqi += 1
            source = equiv_items
        else:
            if bi >= len(batches):
                batches, bi = token_batches(items, args.max_tokens, args.max_items, rng), 0
            idx = batches[bi]
            bi += 1
            source = items
        tb = collate([source[i] for i in idx], tok.pad_token_id, k_fixed=16)
        b = (mx.array(tb["input_ids"].numpy()), mx.array(tb["attention_mask"].numpy()),
             mx.array(tb["marker_pos"].numpy()), mx.array(tb["marker_mask"].numpy()),
             mx.array(tb["qtype"].numpy()), mx.array(tb["target"].numpy()),
             mx.array(tb["equiv_group"].numpy()), mx.array(tb["support_index"].numpy()),
             mx.array(tb["support_flip"].numpy()))
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * s / max(1, total - 1)
        loss, parts, gnorm = step(b, sigma)
        mx.eval(loss, parts, gnorm, state)
        seen += len(idx)
        toks += int(tb["attention_mask"].sum())
        agg += np.array([loss.item()] + [part.item() for part in parts])
        if (s + 1) % 25 == 0 or args.bench:
            el = time.time() - t0
            n = 25 if not args.bench else 1
            rec = {"step": s + 1, "of": total, "epoch": round((s + 1) / len(ep_batches), 3),
                   "items_s": round(seen / el, 1), "tok_s": round(toks / el), "eta_h": round((total - s - 1) * el / (s + 1) / 3600, 2),
                   "sigma": round(sigma, 3), "gnorm": round(gnorm.item(), 3),
                       **{k: round(v / n, 4) for k, v in zip(("loss", "ce", "rl", "act", "acc",
                                                             "consistency", "uncertain", "overconf", "selective"), agg)},
                   "mem_gb": round(mx.get_active_memory() / 2 ** 30, 2), "peak_gb": round(mx.get_peak_memory() / 2 ** 30, 2)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            agg[:] = 0
        if args.bench and s + 1 >= args.bench:
            el = time.time() - t0
            print(f"BENCH items/s={seen/el:.1f} tok/s={toks/el:.0f} epoch_h={len(ep_batches)*el/(s+1)/3600:.2f}")
            return
        if args.save_every and (s + 1) % args.save_every == 0:
            export_laya(model, cfg, args.base, os.path.join(args.out, "rolling"),
                        meta={"step": s + 1, "of": total, "items_seen": seen})
    cfg = dict(cfg)
    cfg["model_name"] = "xDecision"
    cfg["posttrain"] = {"steps": total, "items_seen": seen, "hours": round((time.time() - t0) / 3600, 2),
                        "train_items": len(items), "framework": "mlx", "base_model": "xDecision",
                        "recipe": recipe_metadata(args)}
    export_laya(model, cfg, args.base, os.path.join(args.out, "final"), meta={"done": True, **cfg["posttrain"]})
    print("DONE", json.dumps(cfg["posttrain"]), flush=True)


if __name__ == "__main__":
    main()
