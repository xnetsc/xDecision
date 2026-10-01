"""Native MLX inference from a checkpoint or self-contained xDecision GGUF.

Weights remain resident in the requested precision. Q8_0 is decoded once on
load; inference uses dense MLX operations, not quantized matrix multiplication.
Prompt construction and result decoding use the pinned Laya contract.
"""
import json
import math
import os
import threading
from pathlib import Path

import mlx.core as mx
import numpy as np
from laya.agent import Agent
from laya.common import collapsed_options, _reuse_question_tokens
from tokenizers import Tokenizer

from .mlx_model import LayaMLX


class LocalTokenizer:
    """Read-only Rust tokenizer adapter for Laya's exact sequence builder."""
    def __init__(self, text, config):
        self.backend = Tokenizer.from_str(text)
        self.backend.no_padding()
        self.backend.no_truncation()
        for name in ('cls', 'sep', 'mask', 'pad'):
            token = config.get(name + '_token')
            if isinstance(token, dict):
                token = token['content']
            token_id = self.backend.token_to_id(token) if isinstance(token, str) else None
            if token_id is None:
                raise ValueError(f'Missing {name} token')
            setattr(self, name + '_token', token)
            setattr(self, name + '_token_id', token_id)

    def __call__(self, text, *, add_special_tokens=False, truncation=False, max_length=None):
        ids = self.backend.encode(text, add_special_tokens=add_special_tokens).ids
        if truncation:
            ids = ids[:max_length]
        return {'input_ids': ids}


def validate_temperatures(cfg):
    """Use fitted values unchanged; reject invalid values instead of clamping."""
    temperatures = cfg.get('temperature', [1.0, 1.0, 1.0])
    buckets = cfg.get('temperature_by_options', {})
    if len(temperatures) != 3 or any(
            not math.isfinite(float(t)) or float(t) <= 0
            for t in [*temperatures, *buckets.values()]):
        raise ValueError('Temperatures must be finite and positive, with three question types')
    return [float(t) for t in temperatures], {k: float(t) for k, t in buckets.items()}


def read_checkpoint(path, dtype):
    """Return metadata and resident arrays without writing restored weights to disk."""
    path = Path(path)
    if path.is_dir():
        cfg = json.loads((path/'rl_agent_config.json').read_text())
        enc = json.loads((path/'encoder/config.json').read_text())
        tok_text = (path/'tokenizer/tokenizer.json').read_text()
        tok_cfg = json.loads((path/'tokenizer/tokenizer_config.json').read_text())
        weights = mx.load(str(path/'model.safetensors'))
        weights = {k: v.astype(dtype) for k, v in weights.items()}
    else:
        import gguf
        from .gguf_io import _decoded_tensor
        reader = gguf.GGUFReader(path)
        arch = reader.get_field('general.architecture')
        if arch is None or arch.contents() != 'xdecision':
            raise ValueError('Expected a complete xDecision GGUF')
        def field(name):
            value = reader.get_field('xdecision.' + name)
            if value is None:
                raise ValueError(f'Missing embedded {name}')
            return value.contents()
        cfg = json.loads(field('agent_config'))
        enc = json.loads(field('encoder_config'))
        tok_text = field('tokenizer_json')
        tok_cfg = json.loads(field('tokenizer_config'))
        weights = {}
        for tensor in reader.tensors:
            if tensor.name in weights:
                raise ValueError(f'Duplicate tensor {tensor.name}')
            # Match the reference restoration contract, including Q8 -> F16 rounding.
            value = _decoded_tensor(tensor).astype(
                np.float32 if tensor.name == 'temperature' else np.float16)
            weights[tensor.name] = mx.array(value).astype(dtype)
        mx.eval(weights)
    if 'temperature' not in weights or weights['temperature'].shape != (3,):
        raise ValueError('Missing or invalid temperature tensor')
    weights.pop('temperature')  # Calibrated temperatures are taken from the complete JSON metadata.
    return cfg, enc, tok_text, tok_cfg, weights


def collate(items, pad_id):
    n = len(items)
    length = max(len(row['ids']) for row in items)
    count = max(2, max(len(row['markers']) for row in items))
    ids = np.full((n, length), pad_id, dtype=np.int32)
    att = np.zeros((n, length), dtype=np.bool_)
    mpos = np.zeros((n, count), dtype=np.int32)
    mmask = np.zeros((n, count), dtype=np.bool_)
    for i, row in enumerate(items):
        ids[i, :len(row['ids'])] = row['ids']
        att[i, :len(row['ids'])] = True
        mpos[i, :len(row['markers'])] = row['markers']
        mmask[i, :len(row['markers'])] = True
    return ids, att, mpos, mmask, np.array([r['qtype'] for r in items], dtype=np.int32)


class MLXModel:
    def __init__(self, path, device=None, *, dtype='float16', batch_size=16):
        if dtype not in ('float16', 'float32'):
            raise ValueError('MLX inference dtype must be float16 or float32')
        if dtype == 'float32' and os.environ.get('MLX_ENABLE_TF32') != '0':
            raise ValueError('For strict float32, launch Python with MLX_ENABLE_TF32=0 before importing MLX')
        if device not in (None, 'auto', 'gpu', 'metal', 'mps', 'cpu'):
            raise ValueError('MLX requires an Apple GPU or CPU; use backend="torch" for CUDA')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self.device = mx.cpu if device == 'cpu' else mx.gpu
        if self.device == mx.gpu and not mx.metal.is_available():
            raise RuntimeError('MLX Metal device is not available')
        self.dtype = {'float16': mx.float16, 'float32': mx.float32}[dtype]
        self.batch_size = batch_size
        self._lock = threading.RLock()
        with mx.stream(self.device):
            self.cfg, enc, text, tc, weights = read_checkpoint(path, self.dtype)
            self.temperature, self.temperature_by_options = validate_temperatures(self.cfg)
            self.lang_temperatures = {}  # This checkpoint's contract uses type/option-count calibration.
            if self.cfg.get('lang_temperatures'):
                raise ValueError('Language-specific calibration is not supported by this MLX entry point')
            if (enc.get('model_type') != 'modernbert' or enc.get('hidden_activation', 'gelu') != 'gelu'
                    or any(enc.get(k, False) for k in ('norm_bias', 'mlp_bias', 'attention_bias'))):
                raise ValueError('Unsupported encoder configuration')
            if any(v.get('rope_type', 'default') != 'default'
                   for v in enc.get('rope_parameters', {}).values()):
                raise ValueError('Scaled RoPE is not supported')
            self.tok = LocalTokenizer(text, tc)
            self.model = LayaMLX(enc, self.cfg['head_layers'], len(self.cfg.get('act_costs', {}))+1,
                                 hidden_dtype=self.dtype)
            self.model.load_weights(list(weights.items()), strict=True)
            self.model.eval()
            mx.eval(self.model.parameters())

    @_reuse_question_tokens
    def prepare(self, state, questions):
        if not isinstance(questions, dict):
            raise ValueError('questions must be a dictionary')
        for qid, question in questions.items():
            Agent._check_question(qid, question)
        internal = {qid: Agent._to_internal(q) for qid, q in questions.items()}
        items = Agent._encode_state(self, state, list(questions), internal) if questions else []
        return items, internal

    def forward(self, items):
        with mx.stream(self.device):
            batch = [mx.array(v) for v in collate(items, self.tok.pad_token_id)]
            logits, act = self.model(*batch, cdt=self.dtype, train=False)
            probabilities = mx.softmax(act.astype(mx.float32), axis=-1)
            mx.eval(logits, probabilities)
            logits, probabilities = np.asarray(logits), np.asarray(probabilities)
        if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
            raise FloatingPointError('Non-finite MLX result; restart with MLX_ENABLE_TF32=0 and dtype="float32"')
        return logits, probabilities

    def predict(self, state, questions):
        with self._lock:
            if self.model is None:
                raise RuntimeError('Model is closed')
            items, internal = self.prepare(state, questions)
            qids, answers = list(questions), {}
            for start in range(0, len(items), self.batch_size):
                chunk = items[start:start+self.batch_size]
                logits, act = self.forward(chunk)
                answers.update(Agent._decode_answers(self, logits, act, chunk,
                    qids[start:start+self.batch_size], internal, 0))
            usage = {'input_tokens': sum(len(r['ids']) for r in items), 'output_tokens': 0}
            collapsed = collapsed_options(qids, items)
            if collapsed:
                usage['options'] = collapsed
            return {'model': 'xDecision', 'answers': answers, 'usage': usage}

    def close(self):
        with self._lock:
            self.model = None
