"""Post-train a checkpoint on tokenized decision items on CUDA, ROCm, XPU, MPS, MLX or CPU.

Loss per question (all on the model's own option-marker logits):
  * soft cross-entropy against the gold distribution            (what moves accuracy)
  * the official RLCD term: GRPO-style REINFORCE on Gaussian-perturbed logits with a strictly
    proper reward (log + spherical, + RPS for ordinal score)     (kept from laya's recipe)
  * act head: target mass on the predicted candidate; uniform targets escalate
  * optional same-proposition consistency, uncertainty and selective-risk losses
    shared with MLX; configs/continue.args enables them

Token embeddings are frozen: the 256k-row vocabulary covers 100+ languages and most rows are
never seen in any fine-tuning batch; letting AdamW touch them only degrades languages the
data does not contain, and freezing them removes ~2.4 GB of optimizer state.
"""
import argparse
import json
import math
import os
import random
import time

import torch

from laya.common import proper_reward

from .model_io import DEFAULT_BASE, load_model, load_tokenizer, save_checkpoint
from .devices import (pick_backend, pick_device, pick_precision, autocast_context, synchronize,
                      memory_stats, device_info, mlx_report, cpu_report)
from .training_recipe import add_recipe_arguments, validate_recipe, grouped_batches, recipe_metadata

QTYPE_SCORE = 1


LEN_STEP = 64
K_BUCKETS = (2, 4, 6, 8, 12, 16, 24, 32, 64)


def bucket_len(n):
    return ((n + LEN_STEP - 1) // LEN_STEP) * LEN_STEP


def bucket_k(k):
    return next((b for b in K_BUCKETS if b >= k), k)


def collate(items, pad_id, fixed=True, k_fixed=None):
    """Pad to bucketed shapes. MPS builds a kernel graph per tensor shape and its allocator
    fragments on every new one; a handful of fixed shapes keeps both bounded."""
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    if fixed:
        L, kmax = bucket_len(L), max(bucket_k(kmax), k_fixed or 0)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it["target"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "target": target, "qtype": torch.tensor([it["qtype"] for it in items]),
            "equiv_group": torch.tensor([it.get("equiv_group", -1) for it in items]),
            "support_index": torch.tensor([it.get("support_index", 0) for it in items]),
            "support_flip": torch.tensor([it.get("support_flip", False) for it in items])}


def token_batches(items, max_tokens, max_items, rng, drop_last=False):
    """Batches of one length bucket each, sized to the padded-token budget, shuffled.
    With drop_last every batch of a bucket has the same size, so the set of tensor shapes is
    one per length bucket (what a compiled step needs); dropped items return next epoch."""
    by_len = {}
    for i, it in enumerate(items):
        by_len.setdefault(bucket_len(len(it["ids"])), []).append(i)
    batches = []
    for L, idx in by_len.items():
        rng.shuffle(idx)
        bs = max(1, min(max_items, max_tokens // L))
        batches += [idx[s:s + bs] for s in range(0, len(idx), bs) if not drop_last or s + bs <= len(idx)]
    rng.shuffle(batches)
    return batches


def forward(model, b):
    """DecisionModel.forward, except the act head reads a detached pooled state: its job is to
    predict whether the answer is right, not to reshape the encoder."""
    h = model.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
    h = h + model.type_emb(b["qtype"])[:, None, :]
    pad = ~b["attention_mask"].bool()
    for layer in model.head.layers:
        h = layer(h, src_key_padding_mask=pad)
    idx = b["marker_pos"].clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    logits = model.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
    mm = b["marker_mask"]
    logits = logits.masked_fill(~mm, -1e4)
    p = torch.softmax(logits.detach(), -1)
    k = mm.sum(-1).clamp(min=2).float()
    ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
    top2 = p.topk(2, -1).values
    feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
    act = model.act_head(torch.cat([h[:, 0].detach().float(), feats], -1))
    return logits, act


def reset_act_head(model):
    for m in model.act_head:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
            torch.nn.init.zeros_(m.bias)


def aligned_support(p, qtype, mask, support_index, support_flip):
    nopts = mask.sum(-1).clamp_min(2).float()
    levels = torch.arange(p.size(-1), device=p.device, dtype=p.dtype)
    positive = p.gather(1, support_index[:, None])[:, 0]
    ordinal = (p * levels).sum(-1) / (nopts - 1)
    support = torch.where(qtype == QTYPE_SCORE, ordinal, positive)
    return torch.where(support_flip.bool(), 1 - support, support)


def loss_fn(logits, act_logits, batch, sigma, group=4, w_rl=1.0, w_act=0.1,
            w_consistency=0.0, w_uncertain=0.0, w_overconf=0.0, w_selective=0.0,
            noise=None):
    logits = logits.float()
    mask = batch["marker_mask"]
    target = batch["target"]
    qtype = batch["qtype"]
    masked = logits.masked_fill(~mask, -1e4)
    logp = torch.log_softmax(masked, -1)
    ce = -(target * logp).sum(-1).mean()

    k = mask.sum(-1, keepdim=True).float()
    eps = (torch.randn((group,) + logits.shape, device=logits.device) if noise is None else noise) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std(unbiased=False) + 1e-6)
    lp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    rl = -(adv * lp).mean()

    with torch.no_grad():
        # For ordinal score the answer is the expected level; "right" means within half a level.
        pred = masked.argmax(-1)
        gold = target.argmax(-1)
        correct = pred == gold
        is_score = qtype == QTYPE_SCORE
        if is_score.any():
            levels = torch.arange(target.size(-1), device=target.device, dtype=torch.float32)
            p = torch.softmax(masked, -1)
            exp_pred = (p * levels).sum(-1)
            exp_gold = (target * levels).sum(-1)
            correct = torch.where(is_score, (exp_pred - exp_gold).abs() <= 0.5, correct)
        nopts = mask.sum(-1).clamp_min(2).float()
        determinate = target.max(-1).values >= 0.99
        uniform = (target.max(-1).values - 1 / nopts).abs() < 1e-5
        act_target = torch.where(uniform, 0.0, target.gather(1, pred[:, None])[:, 0])
    act_logp = torch.log_softmax(act_logits.float(), -1)
    act = -(act_target * act_logp[:, 0] + (1-act_target) * act_logp[:, 1]).mean()
    p = torch.softmax(masked, -1)
    support = aligned_support(p, qtype, mask, batch['support_index'], batch['support_flip'])
    gids = batch['equiv_group']
    pairs = ((gids[:, None] == gids[None, :]) & (gids[:, None] >= 0) &
             ~torch.eye(len(gids), dtype=torch.bool, device=gids.device))
    consistency = (((support[:, None]-support[None, :])**2)*pairs).sum()/pairs.sum().clamp_min(1)
    maxp = p.max(-1).values
    uncertain = (((maxp-1/nopts)**2)*uniform).sum()/uniform.sum().clamp_min(1)
    gold_mass = (p*target).sum(-1)
    overconf = ((maxp-0.8).clamp_min(0).square()*(1-gold_mass)*determinate).sum()/determinate.sum().clamp_min(1)
    act_p = torch.softmax(act_logits.float(), -1)[:, 0]
    selective = (act_p.square()*(1-gold_mass.detach()).square()*determinate).sum()/determinate.sum().clamp_min(1)
    total = (ce + w_rl * rl + w_act * act + w_consistency * consistency +
             w_uncertain * uncertain + w_overconf * overconf + w_selective * selective)
    return total, {"ce": ce.item(), "rl": rl.item(), "act": act.item(),
                   "acc": ((correct & determinate).float().sum()/determinate.sum().clamp_min(1)).item(),
                   "consistency": consistency.item(), "uncertain": uncertain.item(),
                   "overconf": overconf.item(), "selective": selective.item()}


def param_groups(model, lr_enc, lr_head, wd):
    groups = {"enc_decay": [], "enc_nodecay": [], "head_decay": [], "head_nodecay": []}
    frozen = 0
    for n, p in model.named_parameters():
        if "tok_embeddings" in n:
            p.requires_grad_(False)
            frozen += p.numel()
            continue
        side = "enc" if n.startswith("encoder.") else "head"
        decay = p.ndim >= 2 and "norm" not in n and "type_emb" not in n
        groups[f"{side}_{'decay' if decay else 'nodecay'}"].append(p)
    out = []
    for key, ps in groups.items():
        if ps:
            out.append({"params": ps, "lr": lr_enc if key.startswith("enc") else lr_head,
                        "weight_decay": wd if key.endswith("_decay") and not key.endswith("nodecay") else 0.0})
    return out, frozen


# Options both trainers accept; the rest are PyTorch-only.
SHARED_OPTIONS = ("base", "train", "out", "epochs", "max_tokens", "max_items", "lr_enc", "lr_head", "wd",
                  "warmup", "sigma_start", "sigma_end", "save_every", "bench", "seed", "equiv",
                  "equiv_every", "w_consistency", "w_uncertain", "w_overconf", "w_selective")


def run_mlx(args):
    """Hand the shared options to mlx_train, which implements the same objective in MLX."""
    notes = []
    if args.accum != 1:
        notes.append(f"--accum {args.accum} ignored: the MLX trainer updates after every micro-batch")
    if args.grad_ckpt:
        notes.append("--grad-ckpt ignored: not implemented by the MLX trainer")
    if args.precision != "auto":
        notes.append("--precision ignored: MLX trains FP32 parameters with BF16 matmuls")
    print(json.dumps({"backend": "mlx", "mlx": mlx_report(), "cpu": cpu_report(), "notes": notes}), flush=True)
    if args.dry_run:
        return
    argv = []
    for name in SHARED_OPTIONS:
        value = getattr(args, name)
        if value is not None:
            argv += ["--" + name.replace("_", "-"), str(value)]
    if args.reset_act_head:
        argv.append("--reset-act-head")
    from . import mlx_train
    mlx_train.main(argv)


def main(argv=None):
    ap = argparse.ArgumentParser(fromfile_prefix_chars='@')
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--reset-act-head", action="store_true", help="Explicitly reinitialize the action head")
    ap.add_argument("--train", default="work/train.pt")
    ap.add_argument("--out", default="work/continued")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=12288, help="padded tokens per micro-batch")
    ap.add_argument("--max-items", type=int, default=64)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr-enc", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--sigma-start", type=float, default=0.4)
    ap.add_argument("--sigma-end", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=1500, help="optimizer steps between rolling saves")
    ap.add_argument("--bench", type=int, default=0, help="run N micro-batches, report throughput, exit")
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "mlx"],
                    help="auto: PyTorch CUDA/ROCm/XPU, else an MLX GPU, else PyTorch MPS/CPU")
    ap.add_argument("--device", default="auto", help="PyTorch device: auto, cuda[:index] (NVIDIA or ROCm), "
                    "xpu[:index], mps or cpu")
    ap.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    ap.add_argument("--dry-run", action="store_true", help="Print selected hardware without loading data/model")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=20260928)
    add_recipe_arguments(ap)
    args = ap.parse_args(argv)
    if pick_backend(args.backend, args.device) == "mlx":
        return run_mlx(args)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = pick_device(args.device)
    precision = pick_precision(device, args.precision)
    hardware = device_info(device, precision)
    print(json.dumps(hardware), flush=True)
    if args.dry_run:
        return
    validate_recipe(args)
    if args.epochs <= 0 or min(args.accum, args.max_items, args.max_tokens) <= 0:
        ap.error("epochs, accum, max-items and max-tokens must be positive")
    if args.bench < 0 or args.save_every < 0:
        ap.error("bench and save-every must be nonnegative")
    if min(args.sigma_start, args.sigma_end) <= 0:
        ap.error("sigma must be positive")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "xpu":
        torch.xpu.set_device(device)
    tok = load_tokenizer(args.base)
    model, cfg = load_model(args.base, device)
    # Every updated checkpoint, including rolling saves, needs fresh calibration.
    cfg = dict(cfg)
    cfg["model_name"] = "xDecision"
    cfg["temperature"] = [1.0, 1.0, 1.0]
    cfg.pop("temperature_by_options", None)
    cfg.pop("calibration", None)
    model.temperature.fill_(1.0)
    if args.reset_act_head:
        reset_act_head(model)
    if args.grad_ckpt:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    groups, frozen = param_groups(model, args.lr_enc, args.lr_head, args.wd)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    print(f"device={device} frozen_params={frozen/1e6:.1f}M "
          f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M", flush=True)

    items = torch.load(args.train, map_location="cpu", weights_only=True)
    if not items:
        raise ValueError("Training data is empty")
    print(f"train items: {len(items)}", flush=True)

    ep_batches = token_batches(items, args.max_tokens, args.max_items, rng)
    equiv_items = torch.load(args.equiv, map_location='cpu', weights_only=True) if args.equiv else []
    eq_batches = grouped_batches(equiv_items, args.max_tokens, args.max_items, rng) if args.equiv else []
    multiplier = args.equiv_every / (args.equiv_every-1) if eq_batches else 1.0
    total_micro = int(len(ep_batches) * args.epochs * multiplier)
    if args.bench:
        total_micro = min(total_micro, args.bench)
    if total_micro < 1:
        raise ValueError("This epoch fraction contains no micro-batches")
    total_steps = math.ceil(total_micro / args.accum)
    warm = max(1, int(total_steps * args.warmup))
    base_lrs = [g["lr"] for g in opt.param_groups]

    def set_lr(step):
        f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))
        f = max(f, 0.02)
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    synchronize(device)
    t0 = time.time()
    micro, step, seen, tokens = 0, 0, 0, 0
    metric_keys = ('ce', 'rl', 'act', 'acc', 'consistency', 'uncertain', 'overconf', 'selective')
    agg = dict.fromkeys((*metric_keys, 'n'), 0.0)
    eqi = 0
    epoch = 0
    batches = ep_batches
    bi = 0
    opt.zero_grad(set_to_none=True)
    while micro < total_micro:
        if eq_batches and (micro + 1) % args.equiv_every == 0:
            if eqi >= len(eq_batches):
                eq_batches, eqi = grouped_batches(equiv_items, args.max_tokens, args.max_items, rng), 0
            idx, source = eq_batches[eqi], equiv_items
            eqi += 1
        else:
            if bi >= len(batches):
                epoch += 1
                batches = token_batches(items, args.max_tokens, args.max_items, rng)
                bi = 0
            idx, source = batches[bi], items
            bi += 1
        b = collate([source[i] for i in idx], tok.pad_token_id)
        b = {k: (v.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else v.to(device))
             for k, v in b.items()}
        progress = micro / max(1, total_micro)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * progress
        with autocast_context(device, precision):
            logits, act_logits = forward(model, b)
        loss, parts = loss_fn(logits, act_logits, b, sigma,
                             w_act=0.2 if eq_batches else 0.1,
                             w_consistency=args.w_consistency, w_uncertain=args.w_uncertain,
                             w_overconf=args.w_overconf, w_selective=args.w_selective)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite training loss; no automatic device/precision fallback")
        window_start = (micro // args.accum) * args.accum
        window_size = min(args.accum, total_micro - window_start)
        scaler.scale(loss / window_size).backward()
        micro += 1
        seen += len(idx)
        tokens += int(b["attention_mask"].sum())
        for k in metric_keys:
            agg[k] += parts[k]
        agg["n"] += 1
        if micro % args.accum == 0 or micro == total_micro:
            set_lr(step)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0,
                                          error_if_nonfinite=not scaler.is_enabled())
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if scaler.get_scale() < scale_before:
                print(json.dumps({"micro": micro, "skipped_nonfinite_gradient": True,
                                  "gradient_scale": scaler.get_scale()}), flush=True)
                continue
            step += 1
            if step % 25 == 0 or args.bench:
                synchronize(device)
                el = time.time() - t0
                rec = {"step": step, "of": total_steps, "epoch": round(micro / len(ep_batches), 3),
                       "items_s": round(seen / el, 2), "tok_s": round(tokens / el),
                       "eta_h": round((total_micro - micro) * (el / micro) / 3600, 2),
                       "lr_enc": opt.param_groups[0]["lr"], "sigma": round(sigma, 3),
                       **hardware, "memory": memory_stats(device),
                       **{k: round(agg[k] / agg["n"], 4) for k in metric_keys}}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                agg = dict.fromkeys((*metric_keys, 'n'), 0.0)
            if args.save_every and step % args.save_every == 0 and not args.bench:
                save_checkpoint(model, cfg, args.base, os.path.join(args.out, "rolling"),
                                meta={"step": step, "of": total_steps})
    log.close()
    if step == 0:
        raise FloatingPointError("No optimizer update succeeded; checkpoint not saved")
    synchronize(device)
    if args.bench:
        el = time.time() - t0
        print(f"BENCH micro={micro} optimizer_steps={step} items/s={seen/el:.2f} tok/s={tokens/el:.0f} "
              f"memory={json.dumps(memory_stats(device))} "
              f"full-epoch-h={len(ep_batches)*el/micro/3600:.2f}", flush=True)
        return
    cfg.setdefault("posttrain", {})
    cfg["posttrain"] = {"steps": step, "micro_batches": micro, "items_seen": seen,
                        "hours": round((time.time() - t0) / 3600, 2), "base_model": "xDecision",
                        "hardware": hardware, "recipe": recipe_metadata(args)}
    save_checkpoint(model, cfg, args.base, os.path.join(args.out, "final"),
                    meta={"step": step, "of": total_steps, "done": True})
    print("DONE", json.dumps(cfg["posttrain"]), flush=True)


if __name__ == "__main__":
    main()
