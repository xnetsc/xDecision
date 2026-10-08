import contextlib
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from xdecision import devices, train


class DeviceSelection(unittest.TestCase):
    def test_cuda_preferred_and_free_memory_selects_index(self):
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2), \
             patch('torch.cuda.mem_get_info', side_effect=[(2, 10), (7, 10)]):
            self.assertEqual(str(devices.pick_device()), 'cuda:1')

    def test_mps_then_cpu(self):
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=True):
            self.assertEqual(str(devices.pick_device('auto')), 'mps')
            self.assertEqual(str(devices.pick_device('cpu')), 'cpu')
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            self.assertEqual(str(devices.pick_device()), 'cpu')
            for name in ['cuda', 'mps']:
                with self.assertRaises(ValueError):
                    devices.pick_device(name)

    def test_explicit_cuda_index_and_invalid_index(self):
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2), \
             patch('torch.cuda.current_device', return_value=1):
            self.assertEqual(str(devices.pick_device('cuda')), 'cuda:1')
            self.assertEqual(str(devices.pick_device('cuda:0')), 'cuda:0')
            with self.assertRaises(ValueError):
                devices.pick_device('cuda:2')

    def test_auto_precision(self):
        with patch('xdecision.devices.supports_bf16', return_value=True):
            self.assertEqual(devices.pick_precision(torch.device('cuda')), 'bf16')
        with patch('xdecision.devices.supports_bf16', return_value=False):
            self.assertEqual(devices.pick_precision(torch.device('cuda')), 'fp16')
            self.assertEqual(devices.pick_precision(torch.device('mps')), 'fp32')
            self.assertEqual(devices.pick_precision(torch.device('cpu')), 'fp32')
            with self.assertRaises(ValueError):
                devices.pick_precision(torch.device('cuda'), 'bf16')
        with self.assertRaises(ValueError):
            devices.pick_precision(torch.device('mps'), 'fp16')

    def test_native_bf16_queries_requested_device(self):
        with patch('torch.cuda.device', return_value=contextlib.nullcontext()) as ctx, \
             patch('torch.cuda.is_bf16_supported', return_value=True) as supported:
            self.assertTrue(devices.supports_bf16(torch.device('cuda:1')))
            ctx.assert_called_once_with(torch.device('cuda:1'))
            supported.assert_called_once_with(including_emulation=False)


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)

    def forward(self, input_ids, attention_mask):
        return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyDecision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.type_emb = torch.nn.Embedding(3, 8)
        self.head = torch.nn.Module()
        self.head.layers = torch.nn.ModuleList([])
        self.scorer = torch.nn.Linear(8, 1)
        self.act_head = torch.nn.Sequential(torch.nn.Linear(12, 2))
        self.register_buffer('temperature', torch.ones(3))


def rows():
    return [dict(ids=[1, 2, 3, 4], markers=[1, 3], target=[1., 0.], qtype=q) for q in range(3)]


class TrainingSteps(unittest.TestCase):
    def step(self, name, precision):
        device = devices.pick_device(name)
        torch.manual_seed(7)
        model = TinyDecision().to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        scaler = torch.amp.GradScaler('cuda', enabled=precision == 'fp16')
        batch = {k: v.to(device) for k, v in train.collate(rows(), 0).items()}
        before = model.scorer.weight.detach().clone()
        with devices.autocast_context(device, precision):
            logits, act = train.forward(model, batch)
        loss, _ = train.loss_fn(logits, act, batch, sigma=.3)
        self.assertTrue(torch.isfinite(loss).item())
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        scaler.step(opt)
        scaler.update()
        devices.synchronize(device)
        self.assertFalse(torch.equal(before, model.scorer.weight))

    def test_cpu_step(self):
        self.step('cpu', 'fp32')

    @unittest.skipUnless(devices.supports_bf16(torch.device('cpu')), 'BF16 is not faster than FP32 on this CPU')
    def test_cpu_bf16_step(self):
        self.assertEqual(devices.pick_precision(torch.device('cpu')), 'bf16')
        self.step('cpu', 'bf16')

    def test_cpu_bf16_follows_the_probe(self):
        probe = devices.probe_cpu_bf16()
        self.assertIs(probe, devices.probe_cpu_bf16())
        self.assertIn('supported', probe)
        for measured, precision in (({'supported': False, 'fp32_ms': 1, 'bf16_ms': 3}, 'fp32'),
                                    ({'supported': False, 'error': 'RuntimeError: no kernel'}, 'fp32'),
                                    ({'supported': True, 'fp32_ms': 3, 'bf16_ms': 1}, 'bf16')):
            with self.subTest(measured=measured), patch('xdecision.devices._CPU_BF16_PROBE', measured):
                self.assertEqual(devices.pick_precision(torch.device('cpu')), precision)
                self.assertEqual(devices.device_info(torch.device('cpu'), precision)['cpu_bf16_probe'], measured)
                if precision == 'fp32':
                    with self.assertRaises(ValueError):
                        devices.pick_precision(torch.device('cpu'), 'bf16')

    def test_probe_failure_falls_back_to_fp32(self):
        with patch('xdecision.devices._CPU_BF16_PROBE', None), \
             patch('torch.autocast', side_effect=RuntimeError('unsupported')):
            probe = devices.probe_cpu_bf16()
            self.assertFalse(probe['supported'])
            self.assertIn('unsupported', probe['error'])

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS hardware required')
    def test_mps_step(self):
        self.step('mps', devices.pick_precision(torch.device('mps')))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware required')
    def test_cuda_steps(self):
        for precision in ['fp32', 'fp16'] + (['bf16'] if devices.supports_bf16(torch.device('cuda')) else []):
            with self.subTest(precision=precision):
                self.step('cuda', precision)

    def test_training_flushes_partial_accumulation(self):
        model = TinyDecision()
        with tempfile.TemporaryDirectory() as out, \
             patch('sys.argv', ['train', '--device', 'cpu', '--max-items', '1', '--accum', '2', '--out', out, '--save-every', '0']), \
             patch('xdecision.train.load_model', return_value=(model, {})), \
             patch('xdecision.train.load_tokenizer', return_value=types.SimpleNamespace(pad_token_id=0)), \
             patch('torch.load', return_value=rows()), \
             patch('xdecision.train.save_checkpoint') as save:
            train.main()
            self.assertEqual(save.call_args.kwargs['meta']['step'], 2)
            self.assertEqual(save.call_args.kwargs['meta']['of'], 2)

    def test_dry_run_does_not_load_model(self):
        with patch('sys.argv', ['train', '--device', 'cpu', '--dry-run']), \
             patch('xdecision.train.load_model') as load:
            train.main()
            load.assert_not_called()

    def test_bench_larger_than_dataset_does_not_save(self):
        with tempfile.TemporaryDirectory() as out, \
             patch('sys.argv', ['train', '--device', 'cpu', '--bench', '99', '--out', out]), \
             patch('xdecision.train.load_model', return_value=(TinyDecision(), {})), \
             patch('xdecision.train.load_tokenizer', return_value=types.SimpleNamespace(pad_token_id=0)), \
             patch('torch.load', return_value=rows()), \
             patch('xdecision.train.save_checkpoint') as save:
            train.main()
            save.assert_not_called()


if __name__ == '__main__':
    unittest.main()
