import argparse
import importlib.util
import random
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from xdecision import train
from xdecision.training_recipe import grouped_batches, add_recipe_arguments, validate_recipe


def views():
    targets = [[0., 1.], [0., 1.], [0., 0., 1.], [1., 0.], [0., 1.], [1., 0., 0.]]
    return [dict(ids=[1, 2, 3, 4, 5, 6], markers=list(range(len(t))), target=t,
                 qtype=q, equiv_group=7, support_index=s, support_flip=f)
            for t, q, s, f in zip(targets, [0, 2, 1, 0, 2, 1], [1, 1, 0, 0, 1, 0],
                                   [False]*5+[True])]


class Consistency(unittest.TestCase):
    def test_mapping_and_reversed_scale(self):
        b = train.collate(views(), 0)
        u = train.aligned_support(b['target'], b['qtype'], b['marker_mask'],
                                  b['support_index'], b['support_flip'])
        torch.testing.assert_close(u, torch.ones(6))

    def test_groups_are_intact_and_budget_is_enforced(self):
        rows = views()+[dict(r, equiv_group=9) for r in views()]
        batches = grouped_batches(rows, 768, 12, random.Random(1))
        self.assertEqual(sorted(sum(batches, [])), list(range(12)))
        for b in batches:
            for gid in (7, 9):
                self.assertIn(sum(rows[i]['equiv_group'] == gid for i in b), (0, 6))
        for tokens, count in [(300, 12), (768, 5)]:
            with self.assertRaises(ValueError):
                grouped_batches(rows, tokens, count, random.Random(1))

    def test_malformed_or_misaligned_groups_fail(self):
        for key, value in [('equiv_group', -1), ('support_index', 9),
                           ('support_flip', None), ('target', [0.5, 0.5])]:
            rows = views()
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                grouped_batches(rows, 4096, 16, random.Random(1))
        with self.assertRaises(ValueError):
            grouped_batches(views()[:-1], 4096, 16, random.Random(1))

    def test_consistency_gradient_and_no_pair_case(self):
        b = train.collate(views(), 0)
        z = torch.randn(b['target'].shape, requires_grad=True)
        a = torch.zeros((6, 2), requires_grad=True)
        noise = torch.zeros((4,)+z.shape)
        loss0, _ = train.loss_fn(z, a, b, .3, noise=noise)
        loss1, parts = train.loss_fn(z, a, b, .3, w_consistency=1, noise=noise)
        grad, = torch.autograd.grad(loss1-loss0, z)
        self.assertGreater(parts['consistency'], 0)
        self.assertGreater(grad.abs().sum().item(), 0)
        b['equiv_group'].fill_(-1)
        _, parts = train.loss_fn(z, a, b, .3, w_consistency=1, noise=noise)
        self.assertEqual(parts['consistency'], 0)

    def test_uniform_target_trains_escalation_and_soft_target_is_not_binary(self):
        rows = [dict(ids=[1, 2], markers=[0, 1], target=t, qtype=0)
                for t in ([.5, .5], [.8, .2])]
        b = train.collate(rows, 0)
        z = torch.tensor([[3., 0.], [3., 0.]], requires_grad=True)
        a = torch.tensor([[2., -1.], [2., -1.]], requires_grad=True)
        _, p = train.loss_fn(z, a, b, .3, noise=torch.zeros((4, 2, 2)))
        logp = a.log_softmax(-1)
        expected = -(logp[0, 1]+.8*logp[1, 0]+.2*logp[1, 1])/2
        self.assertAlmostEqual(p['act'], expected.item(), places=6)

    def test_recipe_requires_equivalence_data(self):
        p = argparse.ArgumentParser()
        add_recipe_arguments(p)
        with self.assertRaises(ValueError):
            validate_recipe(p.parse_args(['--w-consistency', '1']))
        validate_recipe(p.parse_args(['--equiv', 'groups.pt', '--w-consistency', '.5']))

    def test_cli_uses_grouped_batches(self):
        from test_devices import TinyDecision
        regular = [dict(r, equiv_group=-1) for r in views()]
        actual = train.loss_fn
        calls = []
        def tracked(z, a, b, sigma, **kw):
            calls.append((b['equiv_group'].tolist(), kw))
            return actual(z, a, b, sigma, **kw)
        with tempfile.TemporaryDirectory() as out, \
             patch('sys.argv', ['train', '--device', 'cpu', '--train', 'plain', '--equiv', 'eq',
                   '--w-consistency', '.5', '--equiv-every', '3', '--epochs', '3',
                   '--max-items', '6', '--accum', '1', '--bench', '3', '--out', out]), \
             patch('xdecision.train.load_model', return_value=(TinyDecision(), {})), \
             patch('xdecision.train.load_tokenizer', return_value=types.SimpleNamespace(pad_token_id=0)), \
             patch('torch.load', side_effect=[regular, views()]), \
             patch('xdecision.train.loss_fn', side_effect=tracked):
            train.main()
        self.assertEqual([c[0] for c in calls], [[-1]*6, [-1]*6, [7]*6])
        self.assertTrue(all(c[1]['w_consistency'] == .5 for c in calls))

    @unittest.skipUnless(importlib.util.find_spec('mlx'), 'MLX required for backend parity')
    def test_torch_mlx_loss_and_gradient_parity(self):
        import mlx.core as mx
        from xdecision.mlx_train import loss_fn as mlx_loss
        rows = views()
        rows += [dict(r, equiv_group=8, target=[1/len(r['target'])]*len(r['target'])) for r in views()]
        rows += [dict(r, equiv_group=-1, target=[.8, .2]) for r in views() if len(r['target']) == 2]
        b = train.collate(rows, 0)
        rng = np.random.default_rng(21)
        z = (4*rng.normal(size=b['target'].shape)).astype('float32')
        z[~b['marker_mask'].numpy()] = -1e4
        a = rng.normal(size=(len(rows), 2)).astype('float32')
        noise = rng.normal(size=(4,)+z.shape).astype('float32')
        kw = dict(w_act=.2, w_consistency=.5, w_uncertain=.5, w_overconf=.5, w_selective=.2)
        tz, ta = torch.tensor(z, requires_grad=True), torch.tensor(a, requires_grad=True)
        tl, tp = train.loss_fn(tz, ta, b, .3, noise=torch.tensor(noise), **kw)
        tg = torch.autograd.grad(tl, (tz, ta))
        def objective(mz, ma):
            model = lambda *args, **kwargs: (mz, ma)
            args = [mx.array(b[k].numpy()) for k in ('input_ids', 'attention_mask', 'marker_pos',
                    'marker_mask', 'qtype', 'target', 'equiv_group', 'support_index', 'support_flip')]
            return mlx_loss(model, *args, .3, noise=mx.array(noise), **kw)
        ml, mp = objective(mx.array(z), mx.array(a))
        mg = mx.grad(lambda x, y: objective(x, y)[0], argnums=(0, 1))(mx.array(z), mx.array(a))
        self.assertAlmostEqual(tl.item(), ml.item(), places=5)
        for key, value in zip(('ce','rl','act','acc','consistency','uncertain','overconf','selective'), mp):
            self.assertAlmostEqual(tp[key], value.item(), places=5)
        for t, m in zip(tg, mg):
            np.testing.assert_allclose(t.numpy(), np.array(m), atol=2e-5, rtol=2e-4)


if __name__ == '__main__':
    unittest.main()
