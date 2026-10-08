import json
import random
import unittest

import torch

from xdecision import train
from xdecision.contrast import aligned_support, claim_views, comparison_views
from xdecision.contrast_data import generate
from xdecision.probes import CLAIMS as PROBE_CLAIMS, PAIRS, PROBE_ENTITIES, _cases, summarize
from xdecision.training_recipe import grouped_batches


def encoded(views):
    return [dict(ids=[1]*8, markers=list(range(len(v['target']))), target=v['target'],
                 qtype={'choice': 0, 'score': 1, 'noul': 2}[v['question']['type']],
                 equiv_group=0, support_index=v['support_index'], support_flip=v['support_flip'])
            for v in views]


class ContrastViews(unittest.TestCase):
    def test_every_view_maps_to_reference_support(self):
        for g in (0.0, 0.5, 1.0):
            for lang in ('zh', 'en'):
                for views in (comparison_views(lang, 'Aster', 'Brio', 'x', g),
                              claim_views(lang, 'P', 'not P', g)):
                    for v in views:
                        self.assertAlmostEqual(sum(v['target']), 1)
                        self.assertAlmostEqual(aligned_support(v['question'], v['target'], v['support_index'],
                                                               v['support_flip']), g)
                    b = train.collate(encoded(views), 0)
                    u = train.aligned_support(b['target'], b['qtype'], b['marker_mask'],
                                              b['support_index'], b['support_flip'])
                    torch.testing.assert_close(u, torch.full((len(views),), g))

    def test_less_suitable_selects_the_other_candidate(self):
        views = comparison_views('en', 'Aster', 'Brio', 'x', 1.0)
        less = [v for v in views if v['kind'] == 'choice_less']
        for v in less:
            labels = list(v['question']['criteria'])
            self.assertEqual(labels[v['target'].index(1.0)], 'Brio')

    def test_probe_report_shape(self):
        rows = []
        for category, gid, lang, gold, views in list(_cases())[:6]:
            out = [dict(view=i % 10, kind=v['kind'], polarity=v['polarity'], phrasing=v['phrasing'], order=o,
                        support=aligned_support(v['question'], v.get('target', [1 - gold, gold]),
                                                v['support_index'], v['support_flip']))
                   for i, (_, v, o) in enumerate(views)]
            rows.append(dict(category=category, group=gid, lang=lang, gold=gold, views=out))
        summary = summarize(rows)
        self.assertEqual(summary['suitability_context']['view_accuracy'], 1.0)
        self.assertEqual(summary['suitability_context']['fact_order_flip_rate'], 0.0)


class GeneratedData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups, cls.items = generate(40, 20, 30, 10, 3)

    def test_groups_pass_trainer_validation(self):
        rows = [dict(r, ids=[1]*8, markers=list(range(len(r['target']))),
                     qtype={'choice': 0, 'score': 1, 'noul': 2}[r['question']['type']]) for r in self.groups]
        batches = grouped_batches(rows, 4096, 16, random.Random(0))
        self.assertEqual(sorted(i for b in batches for i in b), list(range(len(rows))))

    def test_training_data_is_disjoint_from_probes(self):
        text = json.dumps(self.groups + self.items, ensure_ascii=False)
        tasks = [t for _, _, _, task, _ in PAIRS for t in task]
        # The claim subject with its trailing space, e.g. '会议 ' or 'Meeting '.
        claims = [c.split('{e}')[0] for _, claim, *_ in PROBE_CLAIMS for c in claim]
        for word in sorted(PROBE_ENTITIES) + tasks + claims:
            self.assertNotIn(word, text)


if __name__ == '__main__':
    unittest.main()
