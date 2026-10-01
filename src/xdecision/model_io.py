"""Load a Laya checkpoint directory into a trainable DecisionModel, and write one back out.

The working layout contains encoder/config.json, tokenizer/, model.safetensors and rl_agent_config.json.
Tensor names and the on-disk dtype of the base checkpoint are preserved.
"""
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import build_model

DEFAULT_BASE = os.environ.get("XDECISION_BASE_MODEL", "models/checkpoint")


def load_tokenizer(model_dir):
    tokenizer_dir = os.path.join(model_dir, "tokenizer")
    if not os.path.isdir(tokenizer_dir):
        raise FileNotFoundError(f"tokenizer directory is missing: {tokenizer_dir}; pass --base with an existing checkpoint")
    _fix_tokenizer_config(model_dir)
    return AutoTokenizer.from_pretrained(tokenizer_dir)


def load_config(model_dir):
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        return json.load(f)


def load_model(model_dir, device="cpu"):
    cfg = load_config(model_dir)
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    return model.to(device), cfg


def checkpoint_dtypes(model_dir):
    with safe_open(os.path.join(model_dir, "model.safetensors"), "pt") as f:
        return {k: f.get_slice(k).get_dtype() for k in f.keys()}


_DTYPES = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}


def save_checkpoint(model, cfg, base_dir, out_dir, meta=None):
    """Write `model` in the base checkpoint's layout, tensor names and per-tensor dtypes."""
    os.makedirs(out_dir, exist_ok=True)
    dtypes = checkpoint_dtypes(base_dir)
    sd = {}
    for k, v in model.state_dict().items():
        sd[k] = v.detach().to("cpu", _DTYPES[dtypes[k]]).contiguous()
    tmp = os.path.join(out_dir, "model.safetensors.tmp")
    save_file(sd, tmp)
    os.replace(tmp, os.path.join(out_dir, "model.safetensors"))
    for sub in ("encoder", "tokenizer"):
        dst = os.path.join(out_dir, sub)
        if not os.path.exists(dst):
            shutil.copytree(os.path.join(base_dir, sub), dst)
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    if meta is not None:
        with open(os.path.join(out_dir, "posttrain_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
