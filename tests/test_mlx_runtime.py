import importlib.util
import json
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('mlx'), 'MLX requires Apple Silicon')
class MLXRuntime(unittest.TestCase):
    def test_float32_requires_explicit_precision_policy(self):
        from xdecision.mlx_runtime import MLXModel
        with patch.dict('os.environ', {'MLX_ENABLE_TF32': '1'}):
            with self.assertRaisesRegex(ValueError, 'MLX_ENABLE_TF32=0'):
                MLXModel('unused', dtype='float32')

    def test_temperature_values_are_not_clamped(self):
        from xdecision.mlx_runtime import validate_temperatures
        cfg = {'temperature': [.1, 8., 1.], 'temperature_by_options': {'choice:2': 12.}}
        t, buckets = validate_temperatures(cfg)
        self.assertEqual(t, [.1, 8., 1.])
        self.assertEqual(buckets['choice:2'], 12.)
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                validate_temperatures({'temperature': [value, 1, 1]})

    def test_no_disk_restoration_for_mlx(self):
        from xdecision.runtime import load
        with patch('xdecision.mlx_runtime.MLXModel') as cls, \
             patch('xdecision.runtime.restore_gguf') as restore, \
             patch('xdecision.runtime.tempfile.TemporaryDirectory') as temporary:
            model = load('sample.gguf', backend='mlx')
            cls.assert_called_once_with('sample.gguf', None, dtype='float16', batch_size=16)
            restore.assert_not_called()
            temporary.assert_not_called()
            model.close()
            with self.assertRaises(RuntimeError):
                model.predict('state', {})

    def test_collate_single_and_variable_options(self):
        from xdecision.mlx_runtime import collate
        rows = [dict(ids=[1, 2, 3], markers=[1], qtype=0),
                dict(ids=[1, 2], markers=[0, 1], qtype=2)]
        ids, att, markers, mask, _ = collate(rows, 0)
        self.assertEqual(ids.shape, (2, 3))
        self.assertEqual(markers.shape, (2, 2))
        self.assertEqual(mask.tolist(), [[True, False], [True, True]])
        self.assertEqual(att.tolist(), [[True, True, True], [True, True, False]])

    def test_tiny_model_padding_and_precision(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from xdecision.mlx_model import LayaMLX
        cfg = dict(hidden_size=64, intermediate_size=96, vocab_size=32,
                   num_hidden_layers=2, num_attention_heads=1,
                   layer_types=['full_attention', 'sliding_attention'], local_attention=4)
        mx.random.seed(9)
        ref = LayaMLX(cfg)
        half = LayaMLX(cfg, hidden_dtype=mx.float16)
        half.load_weights([(k, v.astype(mx.float16)) for k, v in tree_flatten(ref.parameters())])
        ids = mx.array([[1, 2, 3, 4, 5]])
        att = mx.ones(ids.shape, dtype=mx.bool_)
        positions = mx.array([[1, 3]])
        mask = mx.ones((1, 2), dtype=mx.bool_)
        qtype = mx.array([0])
        baseline = ref(ids, att, positions, mask, qtype, cdt=mx.float32)
        padded = ref(mx.pad(ids, [(0, 0), (0, 9)]), mx.pad(att, [(0, 0), (0, 9)]),
                     positions, mask, qtype, cdt=mx.float32)
        fp16 = half(ids, att, positions, mask, qtype, cdt=mx.float16)
        for x, p, h in zip(baseline, padded, fp16):
            # Different padded lengths select different SDPA kernels/rounding.
            np.testing.assert_allclose(np.array(x), np.array(p), atol=1e-4, rtol=1e-4)
            np.testing.assert_allclose(np.array(x), np.array(h), atol=.01, rtol=.01)
            self.assertTrue(np.isfinite(np.array(p)).all())

    def test_tokenizer_and_preparation_match_reference(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        from laya.agent import Agent
        from xdecision.mlx_runtime import LocalTokenizer, MLXModel
        from types import SimpleNamespace
        backend = Tokenizer(models.WordLevel({'[UNK]':0, '[CLS]':1, '[SEP]':2, '[MASK]':3,
                                              '[PAD]':4, 'yes':5, 'no':6}, unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        cfg = dict(unk_token='[UNK]', cls_token='[CLS]', sep_token='[SEP]',
                   mask_token='[MASK]', pad_token='[PAD]')
        native = MLXModel.__new__(MLXModel)
        native.tok = LocalTokenizer(backend.to_str(), cfg)
        native.cfg = {'max_len': 64, 'head_max_len': 32}
        ref = SimpleNamespace(tok=PreTrainedTokenizerFast(tokenizer_object=backend, **cfg), cfg=native.cfg)
        q = {'q': {'type':'noul','instructions':'yes','labels': {'false':'no','true':'yes'}}}
        for state in ('yes [MASK] no', '', {'yes': True}, [{'text':'no '*90}, {'text':'yes'}]):
            items, internal = native.prepare(state, q)
            expected = Agent._encode_state(ref, state, ['q'], internal)
            self.assertEqual(items, expected)
        self.assertEqual(native.prepare('yes', {}), ([], {}))
        with self.assertRaises(ValueError):
            native.prepare('yes', {'bad': {'type': 'choice', 'instructions':'yes', 'criteria':['x','x']}})


if __name__ == '__main__':
    unittest.main()
