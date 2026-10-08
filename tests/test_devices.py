import contextlib
import tempfile
import types
import unittest
import unittest.mock
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
        with patch('torch.cuda.is_available', return_value=False), patch('xdecision.devices._xpu', return_value=None), \
             patch('torch.backends.mps.is_available', return_value=True):
            self.assertEqual(str(devices.pick_device('auto')), 'mps')
            self.assertEqual(str(devices.pick_device('cpu')), 'cpu')
        with patch('torch.cuda.is_available', return_value=False), patch('xdecision.devices._xpu', return_value=None), \
             patch('torch.backends.mps.is_available', return_value=False):
            self.assertEqual(str(devices.pick_device()), 'cpu')
            for name in ['cuda', 'mps', 'xpu']:
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


X86_CPUINFO = '''processor\t: 0
vendor_id\t: GenuineIntel
flags\t\t: fpu sse4_2 avx avx2 fma avx512f avx512bw avx512_vnni avx512_bf16 amx_tile amx_bf16
'''
ARM_CPUINFO = '''processor\t: 0
BogoMIPS\t: 50.00
Features\t: fp asimd evtstrm aes asimdhp asimddp sve sve2 i8mm bf16
CPU implementer\t: 0x41
'''


def host(system, machine, flags):
    """cpu_report() for a simulated host; flags=None means the ISA could not be read."""
    with patch('platform.system', return_value=system), patch('platform.machine', return_value=machine), \
         patch('xdecision.devices._read_isa', return_value=(set(flags or ()), 'test')):
        devices.cpu_report.cache_clear()
        try:
            return devices.cpu_report()
        finally:
            devices.cpu_report.cache_clear()


class HardwareDetection(unittest.TestCase):
    def test_architecture_names(self):
        for machine, arch in [('x86_64', 'x86_64'), ('AMD64', 'x86_64'), ('i686', 'x86'), ('aarch64', 'arm64'),
                              ('arm64', 'arm64'), ('armv7l', 'arm'), ('ppc64le', 'ppc64le'),
                              ('riscv64', 'riscv64'), ('', 'unknown')]:
            self.assertEqual(devices.cpu_arch(machine), arch)

    def test_operating_system_feature_lists(self):
        self.assertTrue({'avx512_bf16', 'amx_bf16', 'avx2'} <= devices.parse_cpuinfo(X86_CPUINFO))
        self.assertTrue({'asimd', 'sve2', 'bf16'} <= devices.parse_cpuinfo(ARM_CPUINFO))
        arm = devices.parse_sysctl('hw.optional.arm.FEAT_BF16: 1\nhw.optional.arm.FEAT_I8MM: 0\n'
                                   'hw.optional.neon: 1\nhw.optional.AdvSIMD: 1\n', 'arm')
        self.assertEqual(arm, {'bf16', 'neon', 'advsimd'})
        x86 = devices.parse_sysctl('FPU SSE4.2 AVX1.0 FMA\nAVX2 AVX512F AVX512VNNI\n', 'x86')
        self.assertTrue({'sse4_2', 'avx', 'avx2', 'avx512f', 'avx512_vnni'} <= x86)

    def test_isa_is_read_from_the_running_system(self):
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', unittest.mock.mock_open(read_data=ARM_CPUINFO)):
            flags, source = devices._read_isa('Linux', 'arm')
        self.assertIn('bf16', flags)
        self.assertEqual(source, '/proc/cpuinfo')
        done = types.SimpleNamespace(stdout='hw.optional.arm.FEAT_BF16: 1\n', returncode=0)
        with patch('subprocess.run', return_value=done) as run:
            self.assertEqual(devices._read_isa('Darwin', 'arm'), ({'bf16'}, 'sysctl'))
            self.assertEqual(run.call_args.args[0], ['sysctl', 'hw.optional'])
        self.assertEqual(devices._read_isa('Windows', 'x86'), (set(), None))

    def test_bf16_instruction_set_per_architecture(self):
        cases = [(('Linux', 'x86_64', {'avx2', 'avx512f'}), False),
                 (('Linux', 'x86_64', {'avx2', 'amx_bf16'}), True),
                 (('Linux', 'aarch64', {'asimd', 'sve'}), False),
                 (('Linux', 'aarch64', {'asimd', 'bf16'}), True),
                 (('Darwin', 'arm64', {'neon', 'bf16'}), True),
                 (('Linux', 'riscv64', {'rv64imafdc'}), None),
                 (('Windows', 'AMD64', None), None)]
        for (system, machine, flags), expected in cases:
            with self.subTest(machine=machine, flags=flags):
                report = host(system, machine, flags)
                self.assertIs(report['bf16_isa'], expected)
                self.assertEqual(report['arch'], devices.cpu_arch(machine))

    def test_missing_bf16_instructions_skip_the_probe(self):
        with patch('xdecision.devices.cpu_report', return_value=host('Linux', 'aarch64', {'asimd'})), \
             patch('xdecision.devices.probe_cpu_bf16') as probe:
            self.assertFalse(devices.cpu_bf16()['supported'])
            self.assertEqual(devices.pick_precision(torch.device('cpu')), 'fp32')
            probe.assert_not_called()
        for flags in ({'asimd', 'bf16'}, None):
            with self.subTest(flags=flags), \
                 patch('xdecision.devices.cpu_report', return_value=host('Linux', 'aarch64', flags)), \
                 patch('xdecision.devices.probe_cpu_bf16', return_value={'supported': True}) as probe:
                self.assertEqual(devices.pick_precision(torch.device('cpu')), 'bf16')
                probe.assert_called_once()

    def test_gpu_runtime_is_identified(self):
        with patch('torch.cuda.is_available', return_value=True), patch('torch.version.hip', '6.2'):
            self.assertEqual(devices.cuda_runtime(), 'rocm')
        with patch('torch.cuda.is_available', return_value=True), patch('torch.version.hip', None):
            self.assertEqual(devices.cuda_runtime(), 'cuda')
        with patch('torch.cuda.is_available', return_value=False):
            self.assertIsNone(devices.cuda_runtime())

    def test_intel_xpu_after_cuda(self):
        xpu = types.SimpleNamespace(device_count=lambda: 1)
        with patch('torch.cuda.is_available', return_value=False), patch('xdecision.devices._xpu', return_value=xpu):
            self.assertEqual(str(devices.pick_device()), 'xpu:0')
            self.assertEqual(str(devices.pick_device('xpu')), 'xpu:0')
            with self.assertRaises(ValueError):
                devices.pick_device('xpu:1')
            with self.assertRaises(ValueError):
                devices.pick_precision(torch.device('xpu'), 'fp16')

    def test_backend_choice(self):
        metal = {'installed': True, 'gpu': 'metal'}
        cases = [(('auto', 'auto', None, None, metal), 'mlx'),
                 (('auto', 'auto', 'cuda', None, metal), 'torch'),
                 (('auto', 'auto', None, object(), metal), 'torch'),
                 (('auto', 'mps', None, None, metal), 'torch'),
                 (('auto', 'auto', None, None, {'installed': True, 'gpu': None}), 'torch'),
                 (('auto', 'auto', None, None, {'installed': False, 'gpu': None}), 'torch'),
                 (('torch', 'auto', None, None, metal), 'torch'),
                 (('mlx', 'auto', 'cuda', None, metal), 'mlx')]
        for (requested, device, cuda, xpu, mlx), expected in cases:
            with self.subTest(requested=requested, device=device, cuda=cuda, xpu=xpu, mlx=mlx), \
                 patch('xdecision.devices.cuda_runtime', return_value=cuda), \
                 patch('xdecision.devices._xpu', return_value=xpu), \
                 patch('xdecision.devices.mlx_report', return_value=mlx):
                self.assertEqual(devices.pick_backend(requested, device), expected)
        with patch('xdecision.devices.mlx_report', return_value={'installed': False, 'gpu': None}), \
             self.assertRaises(ValueError):
            devices.pick_backend('mlx')

    def test_train_hands_shared_options_to_mlx(self):
        fake = types.SimpleNamespace(main=unittest.mock.Mock())
        with patch.dict('sys.modules', {'xdecision.mlx_train': fake}), \
             patch('xdecision.train.pick_backend', return_value='mlx'), \
             patch('xdecision.train.mlx_report', return_value={'installed': True, 'gpu': 'metal'}):
            train.main(['--train', 'a.pt', '--equiv', 'b.pt', '--w-consistency', '.5', '--accum', '4',
                        '--reset-act-head', '--out', 'o'])
        argv = fake.main.call_args.args[0]
        self.assertEqual(argv[argv.index('--equiv') + 1], 'b.pt')
        self.assertEqual(argv[argv.index('--w-consistency') + 1], '0.5')
        self.assertIn('--reset-act-head', argv)
        for torch_only in ('--accum', '--device', '--precision', '--grad-ckpt'):
            self.assertNotIn(torch_only, argv)


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
            with self.subTest(measured=measured), patch('xdecision.devices._CPU_BF16_PROBE', measured), \
                 patch('xdecision.devices.cpu_report', return_value=host('Linux', 'x86_64', None)):
                self.assertEqual(devices.pick_precision(torch.device('cpu')), precision)
                self.assertEqual(devices.device_info(torch.device('cpu'), precision)['cpu_bf16'], measured)
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
