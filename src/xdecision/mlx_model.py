"""ModernBERT/mmBERT encoder and complete xDecision heads in native MLX.

Parameter names mirror the PyTorch state dict exactly, so a checkpoint loads and
exports one-to-one (see load_laya / export_laya).

Training defaults to fp32 parameters/residuals and bf16 matmuls/attention.
Inference may use resident fp16 parameters and residuals; output logits and
action probabilities retain fp32 computation.
"""
import json
import os
import shutil

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def _lin(x, m, cdt):
    y = x.astype(cdt) @ m.weight.astype(cdt).T
    if "bias" in m:
        y = y + m.bias.astype(cdt)
    return y


def _ln(x, m, eps, dtype=mx.float32):
    return mx.fast.layer_norm(x.astype(dtype), m.weight, m["bias"] if "bias" in m else None, eps)


class Embeddings(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab, d)
        self.norm = nn.LayerNorm(d, bias=False)


class Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Wqkv = nn.Linear(d, 3 * d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)


class MLP(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.Wi = nn.Linear(d, 2 * f, bias=False)
        self.Wo = nn.Linear(f, d, bias=False)


class EncLayer(nn.Module):
    def __init__(self, d, f, first):
        super().__init__()
        if not first:
            self.attn_norm = nn.LayerNorm(d, bias=False)
        self.attn = Attn(d)
        self.mlp_norm = nn.LayerNorm(d, bias=False)
        self.mlp = MLP(d, f)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, f = cfg["hidden_size"], cfg["intermediate_size"]
        self.embeddings = Embeddings(cfg["vocab_size"], d)
        self.layers = [EncLayer(d, f, i == 0) for i in range(cfg["num_hidden_layers"])]
        self.final_norm = nn.LayerNorm(d, bias=False)


class MHA(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.in_proj_weight = mx.zeros((3 * d, d))
        self.in_proj_bias = mx.zeros((3 * d,))
        self.out_proj = nn.Linear(d, d)


class HeadLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = MHA(d)
        self.linear1 = nn.Linear(d, 4 * d)
        self.linear2 = nn.Linear(4 * d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)


class Head(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.layers = [HeadLayer(d) for _ in range(n)]


class LayaMLX(nn.Module):
    def __init__(self, enc_cfg, head_layers=2, n_act=2, hidden_dtype=mx.float32):
        super().__init__()
        self.hidden_dtype = hidden_dtype
        d = enc_cfg["hidden_size"]
        self.ecfg = enc_cfg
        self.encoder = Encoder(enc_cfg)
        self.head = Head(d, head_layers)
        self.type_emb = nn.Embedding(3, d)
        self.scorer = [nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)]
        self.act_head = [nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act)]
        self.n_heads = enc_cfg["num_attention_heads"]
        self.eps = enc_cfg.get("norm_eps", 1e-5)
        self.window = enc_cfg.get("local_attention", 128) // 2
        self.layer_types = enc_cfg["layer_types"]
        rp = enc_cfg.get("rope_parameters") or {}
        self.theta = {lt: float((rp.get(lt) or {}).get("rope_theta", enc_cfg.get("global_rope_theta", 160000)))
                      for lt in ("full_attention", "sliding_attention")}
        self.head_dropout = 0.1

    # ------------------------------------------------------------------ encoder
    def _attention(self, x, layer, masks, lt, cdt):
        B, L, D = x.shape
        H = self.n_heads
        dh = D // H
        qkv = _lin(x, layer.attn.Wqkv, cdt).reshape(B, L, 3, H, dh)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        q = mx.fast.rope(q.astype(self.hidden_dtype), dh, traditional=False, base=self.theta[lt], scale=1.0, offset=0).astype(cdt)
        k = mx.fast.rope(k.astype(self.hidden_dtype), dh, traditional=False, base=self.theta[lt], scale=1.0, offset=0).astype(cdt)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=dh ** -0.5, mask=masks[lt])
        return _lin(o.transpose(0, 2, 1, 3).reshape(B, L, D), layer.attn.Wo, cdt)

    def encode(self, ids, att, cdt=mx.bfloat16):
        B, L = ids.shape
        x = _ln(self.encoder.embeddings.tok_embeddings(ids), self.encoder.embeddings.norm, self.eps, self.hidden_dtype)
        keep = att.astype(mx.bool_)[:, None, None, :]
        pos = mx.arange(L)
        band = mx.abs(pos[:, None] - pos[None, :]) <= self.window
        neg = mx.array(-float('inf'), dtype=cdt)
        zero = mx.array(0.0, dtype=cdt)
        masks = {"full_attention": mx.where(keep, zero, neg),
                 "sliding_attention": mx.where(keep & (band[None, None] | ~att.astype(mx.bool_)[:, None, :, None]), zero, neg)}
        for i, layer in enumerate(self.encoder.layers):
            lt = self.layer_types[i]
            h = _ln(x, layer.attn_norm, self.eps, self.hidden_dtype) if i > 0 else x
            x = x + self._attention(h, layer, masks, lt, cdt).astype(self.hidden_dtype)
            h = _ln(x, layer.mlp_norm, self.eps, self.hidden_dtype)
            a, g = mx.split(_lin(h, layer.mlp.Wi, cdt), 2, axis=-1)
            x = x + _lin(nn.gelu(a) * g, layer.mlp.Wo, cdt).astype(self.hidden_dtype)
        return _ln(x, self.encoder.final_norm, self.eps, self.hidden_dtype)

    # ------------------------------------------------------------------ decision head
    def _head_layer(self, x, hl, keymask, cdt, train):
        B, L, D = x.shape
        H = max(1, D // 64)
        dh = D // H
        h = _ln(x, hl.norm1, 1e-5, self.hidden_dtype)
        w, b = hl.self_attn.in_proj_weight, hl.self_attn.in_proj_bias
        qkv = h.astype(cdt) @ w.astype(cdt).T + b.astype(cdt)
        q, k, v = (t.reshape(B, L, H, dh).transpose(0, 2, 1, 3) for t in mx.split(qkv, 3, axis=-1))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=dh ** -0.5, mask=keymask)
        o = _lin(o.transpose(0, 2, 1, 3).reshape(B, L, D), hl.self_attn.out_proj, cdt).astype(self.hidden_dtype)
        if train:
            o = _dropout(o, self.head_dropout)
        x = x + o
        h = _ln(x, hl.norm2, 1e-5, self.hidden_dtype)
        f = nn.relu(_lin(h, hl.linear1, cdt))
        if train:
            f = _dropout(f, self.head_dropout)
        f = _lin(f, hl.linear2, cdt).astype(self.hidden_dtype)
        if train:
            f = _dropout(f, self.head_dropout)
        return x + f

    def __call__(self, ids, att, mpos, mmask, qtype, cdt=mx.bfloat16, train=False):
        h = self.encode(ids, att, cdt)
        h = h + self.type_emb(qtype)[:, None, :]
        keymask = mx.where(att.astype(mx.bool_)[:, None, None, :], mx.array(0.0, cdt), mx.array(-float('inf'), cdt))
        for hl in self.head.layers:
            h = self._head_layer(h, hl, keymask, cdt, train)
        m = mx.take_along_axis(h, mpos[:, :, None], axis=1)
        s = _ln(m, self.scorer[0], 1e-5, self.hidden_dtype)
        s = nn.gelu(_lin(s, self.scorer[1], cdt).astype(self.hidden_dtype))
        logits = _lin(s, self.scorer[3], cdt).astype(mx.float32).squeeze(-1)
        logits = mx.where(mmask, logits, -1e4)
        p = mx.stop_gradient(mx.softmax(logits, axis=-1))
        k = mx.maximum(mmask.sum(-1), 2).astype(mx.float32)
        ent = -(p * mx.log(mx.maximum(p, 1e-9))).sum(-1) / mx.log(k)
        ps = mx.sort(p, axis=-1)
        top1, top2 = ps[:, -1], ps[:, -2]
        feats = mx.stack([top1, top1 - top2, ent, k / 255.0], axis=-1)
        pooled = mx.stop_gradient(h[:, 0])
        a = _lin(mx.concatenate([pooled, feats], axis=-1), self.act_head[0], mx.float32)
        act = _lin(nn.gelu(a), self.act_head[2], mx.float32)
        return logits, act


def _dropout(x, p):
    keep = mx.random.bernoulli(1 - p, x.shape)
    return mx.where(keep, x / (1 - p), 0.0)


# ---------------------------------------------------------------------- I/O
def load_laya(model_dir):
    enc_cfg = json.load(open(os.path.join(model_dir, "encoder", "config.json")))
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    model = LayaMLX(enc_cfg, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1)
    weights, dtypes = [], {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), "np") as f:
        for k in f.keys():
            dtypes[k] = f.get_slice(k).get_dtype()
            if k == "temperature":
                continue
            weights.append((k, mx.array(f.get_tensor(k).astype(np.float32))))
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model, cfg, dtypes


def export_laya(model, cfg, base_dir, out_dir, temperature=None, meta=None):
    from mlx.utils import tree_flatten
    os.makedirs(out_dir, exist_ok=True)
    np_dt = {"F16": np.float16, "F32": np.float32}
    with safe_open(os.path.join(base_dir, "model.safetensors"), "np") as f:
        dtypes = {k: f.get_slice(k).get_dtype() for k in f.keys()}
        base_temp = f.get_tensor("temperature")
    out = {}
    for k, v in tree_flatten(model.parameters()):
        if k not in dtypes:
            continue
        out[k] = np.asarray(v.astype(mx.float32)).astype(np_dt[dtypes[k]])
    t = np.asarray(temperature if temperature is not None else np.ones_like(base_temp), dtype=np.float32)
    out["temperature"] = t.astype(np_dt[dtypes["temperature"]])
    missing = set(dtypes) - set(out)
    if missing:
        raise RuntimeError(f"export is missing tensors: {sorted(missing)[:5]}")
    tmp = os.path.join(out_dir, "model.safetensors.tmp")
    save_file(out, tmp)
    os.replace(tmp, os.path.join(out_dir, "model.safetensors"))
    for sub in ("encoder", "tokenizer"):
        if not os.path.exists(os.path.join(out_dir, sub)):
            shutil.copytree(os.path.join(base_dir, sub), os.path.join(out_dir, sub))
    cfg = dict(cfg)
    cfg["model_name"] = "xDecision"
    cfg["temperature"] = t.tolist()
    cfg.pop("temperature_by_options", None)
    cfg.pop("calibration", None)
    json.dump(cfg, open(os.path.join(out_dir, "rl_agent_config.json"), "w"), indent=2, ensure_ascii=False)
    if meta is not None:
        json.dump(meta, open(os.path.join(out_dir, "posttrain_meta.json"), "w"), indent=2, ensure_ascii=False)
