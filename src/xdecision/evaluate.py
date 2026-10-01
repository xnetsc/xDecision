"""Evaluate a checkpoint directory with laya's own PyTorch forward (the code path consumers load),
and fit calibration temperatures.

  python -m xdecision.evaluate --ckpt DIR [--calibrate] [--suites typed_decisions,emotion,...]

--calibrate fits one temperature per (question type, option-count bucket) on data/built/calib.pt
by held-out NLL and writes them into DIR/rl_agent_config.json before evaluating.
"""
import argparse
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch

from laya.common import QTYPE_NAMES, clamp_temperature, ece_score, temp_bucket

from .model_io import load_config, load_model, load_tokenizer
from .train import collate
from .devices import pick_device

SUITES = ["typed_decisions", "emotion", "sst5", "prompt_injection", "xcopa", "xwinograd", "belebele", "xstory",
          "wd_heldout", "wiki_heldout", "hans_formats", "unseen_workflows", "market_heldout", "uncertainty", "fitness_unseen",
          "unseen_workflows_extra"]


@torch.no_grad()
def logits_for(model, items, pad_id, device, max_tokens=16384):
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    out = [None] * len(items)
    act_out = [None] * len(items)
    i = 0
    while i < len(order):
        L = len(items[order[i]]["ids"])
        bs = max(1, min(128, max_tokens // max(64, ((L + 63) // 64) * 64)))
        idx = order[i:i + bs]
        # the longest item in this slice sets the padded length
        while len(idx) > 1 and len(items[idx[-1]]["ids"]) * len(idx) > max_tokens * 1.5:
            idx = idx[:-1]
        b = collate([items[j] for j in idx], pad_id)
        b = {k: v.to(device) for k, v in b.items()}
        lg, act = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        lg = lg.float().cpu().numpy()
        act = torch.softmax(act.float(), -1)[:, 0].cpu().numpy()
        for r, j in enumerate(idx):
            out[j] = lg[r, : len(items[j]["markers"])]
            act_out[j] = float(act[r])
        i += len(idx)
    return out, act_out


def temps_from_cfg(cfg):
    t = [clamp_temperature(x) for x in cfg.get("temperature", [1.0, 1.0, 1.0])]
    tb = {k: clamp_temperature(v) for k, v in (cfg.get("temperature_by_options") or {}).items()}
    return t, tb


def probs(z, qtype, t, tb):
    T = tb.get(temp_bucket(qtype, len(z)), t[qtype])
    z = np.asarray(z, dtype=np.float64) / T
    p = np.exp(z - z.max())
    return p / p.sum()


def fit_temperature(pairs, lo=0.5, hi=5.0):
    """1-D NLL minimisation over log T (golden section; the NLL is unimodal in log T)."""
    def nll(logT):
        T = math.exp(logT)
        s = 0.0
        for z, tgt in pairs:
            z = np.asarray(z, dtype=np.float64) / T
            lse = z.max() + math.log(np.exp(z - z.max()).sum())
            s -= float(np.dot(tgt, z - lse))
        return s / len(pairs)
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(d)
    return math.exp((a + b) / 2)


def calibrate(model, tok, device, calib_items, min_n=40):
    lg, _ = logits_for(model, calib_items, tok.pad_token_id, device)
    by_bucket, by_type = defaultdict(list), defaultdict(list)
    for z, it in zip(lg, calib_items):
        pair = (z, np.asarray(it["target"], dtype=np.float64))
        by_bucket[temp_bucket(it["qtype"], len(z))].append(pair)
        by_type[it["qtype"]].append(pair)
    t = [round(fit_temperature(by_type[q]), 4) if len(by_type[q]) >= min_n else 1.0 for q in range(3)]
    tb = {k: round(fit_temperature(v), 4) for k, v in sorted(by_bucket.items()) if len(v) >= min_n}
    return t, tb, {k: len(v) for k, v in by_bucket.items()}


def auroc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    ranks = s.argsort().argsort() + 1
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def score_items(items, lg, act, t, tb):
    rows = []
    for it, z, a in zip(items, lg, act):
        q = it["qtype"]
        p = probs(z, q, t, tb)
        tgt = np.asarray(it["target"])
        gold = int(it.get("label", int(tgt.argmax())))
        pred = int(p.argmax())
        row = {"src": it["src"], "lang": it.get("lang", "en"), "t": QTYPE_NAMES[q], "correct": pred == gold,
               "conf": float(p.max()), "soft": float((p * tgt).sum()), "act": a, "pred": pred, "gold": gold,
               "gold_prob": float(p[gold]),
               "brier": float(np.sum((p - np.eye(len(p), dtype=np.float64)[gold]) ** 2)),
               "determinate": bool(it.get("determinate", bool(tgt.max() >= 0.99) or "label" in it))}
        if q == 1:
            lv = np.arange(len(p))
            row["mae"] = abs(float((p * lv).sum()) - gold)
            row["first"] = pred == 0
        if "workflow" in it:
            row["workflow"] = it["workflow"]
        rows.append(row)
    return rows


def summarize(rows):
    def agg(rs):
        det = [r for r in rs if r["determinate"]]
        if not det:
            conf = np.asarray([r["conf"] for r in rs], dtype=np.float64)
            act = np.asarray([r["act"] for r in rs], dtype=np.float64)
            return {"n": len(rs), "mean_conf": round(float(conf.mean()), 4),
                    "p95_conf": round(float(np.percentile(conf, 95)), 4),
                    "high_conf_rate_075": round(float(np.mean(conf >= 0.75)), 4),
                    "mean_act": round(float(act.mean()), 4),
                    "p95_act": round(float(np.percentile(act, 95)), 4)}
        conf = np.asarray([r["conf"] for r in det], dtype=np.float64)
        correct = np.asarray([r["correct"] for r in det], dtype=bool)
        accepted = conf >= 0.8
        order = np.argsort(-conf)
        cumulative_risk = np.cumsum(~correct[order]) / np.arange(1, len(det) + 1)
        d = {"n": len(det), "acc": round(float(np.mean([r["correct"] for r in det])), 4),
             "soft_acc": round(float(np.mean([r["soft"] for r in det])), 4),
             "ece": round(ece_score(np.array([r["conf"] for r in det]), np.array([r["correct"] for r in det], float)), 4),
             "act_auroc": round(auroc([r["act"] for r in det], [r["correct"] for r in det]), 4),
             "nll": round(float(np.mean([-math.log(max(r["gold_prob"], 1e-12)) for r in det])), 4),
             "brier": round(float(np.mean([r["brier"] for r in det])), 4),
             "wrong_high_conf_rate_080": round(float(np.mean((~correct) & accepted)), 4),
             "coverage_at_080": round(float(accepted.mean()), 4),
             "accuracy_at_080": round(float(correct[accepted].mean()), 4) if accepted.any() else None,
             "aurc": round(float(cumulative_risk.mean()), 4)}
        sc = [r for r in det if "mae" in r]
        if sc:
            d["score_mae"] = round(float(np.mean([r["mae"] for r in sc])), 4)
            d["first_level_rate"] = round(float(np.mean([r["first"] for r in sc])), 4)
            d["first_level_gold_rate"] = round(float(np.mean([r["gold"] == 0 for r in sc])), 4)
        return d
    out = {"all": agg(rows)}
    for key in ("t", "lang", "workflow", "src"):
        groups = defaultdict(list)
        for r in rows:
            if key in r:
                groups[r[key]].append(r)
        if len(groups) > 1:
            out["by_" + key] = {k: agg(v) for k, v in sorted(groups.items())}
    und = [r for r in rows if not r["determinate"]]
    if und:
        conf = np.asarray([r["conf"] for r in und], dtype=np.float64)
        act = np.asarray([r["act"] for r in und], dtype=np.float64)
        out["undetermined_mean_conf"] = round(float(conf.mean()), 4)
        out["undetermined_p95_conf"] = round(float(np.percentile(conf, 95)), 4)
        out["undetermined_high_conf_rate_075"] = round(float(np.mean(conf >= 0.75)), 4)
        out["undetermined_mean_act"] = round(float(act.mean()), 4)
        out["undetermined_p95_act"] = round(float(np.percentile(act, 95)), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/built")
    ap.add_argument("--suites", default=",".join(SUITES + ["val"]))
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calib-file", default="calib.pt")
    ap.add_argument("--device", default="auto", help="auto, cuda[:index], mps or cpu; evaluation remains FP32")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-per-suite", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    device = pick_device(args.device)
    print(f"device={device} precision=fp32", flush=True)
    if args.threads:
        torch.set_num_threads(args.threads)
    tok = load_tokenizer(args.ckpt)
    model, cfg = load_model(args.ckpt, device)
    model.eval()
    t0 = time.time()
    shipped = temps_from_cfg(cfg)
    if args.calibrate:
        calib = torch.load(os.path.join(args.data, args.calib_file), weights_only=False)
        t, tb, counts = calibrate(model, tok, device, calib)
        cfg["temperature"] = t
        cfg["temperature_by_options"] = tb
        cfg["calibration"] = {"items": len(calib), "per_bucket": counts, "method": "held-out NLL, T in [0.5, 5]"}
        json.dump(cfg, open(os.path.join(args.ckpt, "rl_agent_config.json"), "w"), indent=2, ensure_ascii=False)
        print(f"calibrated in {time.time()-t0:.0f}s: temperature={t} by_options={tb}", flush=True)
    t, tb = temps_from_cfg(cfg)
    report = {"ckpt": os.path.abspath(args.ckpt), "device": str(device), "precision": "fp32",
              "temperature": t, "temperature_by_options": tb, "suites": {},
              "suites_as_shipped": {}}
    for name in args.suites.split(","):
        path = os.path.join(args.data, f"{name}.pt" if name == "val" else f"eval_{name}.pt")
        if not os.path.exists(path):
            continue
        items = torch.load(path, weights_only=False)
        if args.max_per_suite:
            items = items[: args.max_per_suite]
        lg, act = logits_for(model, items, tok.pad_token_id, device)
        report["suites"][name] = summarize(score_items(items, lg, act, t, tb))
        if args.calibrate:
            report["suites_as_shipped"][name] = summarize(score_items(items, lg, act, *shipped))
        a = report["suites"][name]["all"]
        print(f"[{time.time()-t0:5.0f}s] {name:17s} n={a.get('n')} acc={a.get('acc')} ece={a.get('ece')} "
              f"act_auroc={a.get('act_auroc')}", flush=True)
    out = args.out or os.path.join(args.ckpt, "eval_report.json")
    json.dump(report, open(out, "w"), indent=1, ensure_ascii=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
