import unittest

import torch

from xdecision import train
from xdecision.contrast import aligned_support, claim_views, comparison_views
from xdecision.probes import _cases, summarize


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


if __name__ == '__main__':
    unittest.main()
