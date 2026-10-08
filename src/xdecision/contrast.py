"""Equivalent views of one proposition, asked in both polarities.

A group holds every way of asking one reference proposition: "which is more suitable" and
"which is less suitable" in both candidate orders, "A is better than B" and "B is better than
A", a claim and its negation, forward and reversed score scales, and permuted fact orders.
Each view records the alignment that the consistency loss in train.py / mlx_train.py uses:
`support_index` names the candidate whose probability supports the reference proposition and
`support_flip` marks views that ask its opposite. The held-out probes and the training data
use these builders with disjoint entities and domains.

Targets follow the gold support `g` of the reference proposition: 1 (holds), 0 (does not
hold) or 0.5 (the facts do not decide it). Every view of a group maps to the same `g`.
"""
import random

LEVELS = 5

COMPARE = {
    'zh': {
        'more': ['哪个更适合{task}？', '要{task}，选哪个更合适？', '以下哪一个更适合{task}？'],
        'less': ['哪个更不适合{task}？', '要{task}，哪个是更差的选择？', '以下哪一个不太适合{task}？'],
        'gt': ['{x}比{y}更适合{task}。', '如果要{task}，{x}比{y}更合适。'],
        'lt': ['{x}不如{y}适合{task}。', '如果要{task}，选{x}不如选{y}。'],
        'score': ['与{y}相比，{x}用于{task}的适合程度如何？'],
        'levels': ['明显更不适合', '略微更不适合', '差不多', '略微更适合', '明显更适合'],
    },
    'en': {
        'more': ['Which is more suitable for {task}?', 'Which would be the better choice for {task}?',
                 'For {task}, which option fits better?'],
        'less': ['Which is less suitable for {task}?', 'Which would be the worse choice for {task}?',
                 'For {task}, which option fits worse?'],
        'gt': ['{x} is more suitable than {y} for {task}.', 'For {task}, {x} is a better choice than {y}.'],
        'lt': ['{x} is less suitable than {y} for {task}.', 'For {task}, {x} is a worse choice than {y}.'],
        'score': ['Compared with {y}, how suitable is {x} for {task}?'],
        'levels': ['much less suitable', 'somewhat less suitable', 'about equally suitable',
                   'somewhat more suitable', 'much more suitable'],
    },
}

CLAIM = {
    'zh': {
        'noul': ['根据事实判断：{claim}', '{claim}'],
        'choice': ['根据事实，以下说法是否成立：{claim}'],
        'options': ('成立', '不成立'),
        'score': ['根据事实，以下说法的成立程度如何：{claim}'],
        'levels': ['明确不成立', '可能不成立', '无法确定', '可能成立', '明确成立'],
    },
    'en': {
        'noul': ['Based on the facts: {claim}', '{claim}'],
        'choice': ['Based on the facts, does this statement hold: {claim}'],
        'options': ('holds', 'does not hold'),
        'score': ['Based on the facts, how well does this statement hold: {claim}'],
        'levels': ['clearly does not hold', 'probably does not hold', 'cannot be determined',
                   'probably holds', 'clearly holds'],
    },
}


def _pick(rng, options):
    return options[rng.randrange(len(options))] if rng is not None else options[0]


def _choice(ins, labels, target, support_index, kind, polarity):
    return dict(question={'type': 'choice', 'instructions': ins, 'criteria': {k: '' for k in labels}},
                target=target, support_index=support_index, support_flip=False,
                kind=kind, polarity=polarity)


def _noul(ins, g, flip, kind):
    p = 1 - g if flip else g
    return dict(question={'type': 'noul', 'instructions': ins}, target=[1 - p, p],
                support_index=1, support_flip=flip, kind=kind, polarity='neg' if flip else 'pos')


def _score(ins, levels, g, flip, kind):
    if g == 0.5:
        target = [1 / LEVELS] * LEVELS
    else:
        target = [0.0] * LEVELS
        target[-1 if g == 1 else 0] = 1.0
    if flip:
        levels, target = levels[::-1], target[::-1]
    return dict(question={'type': 'score', 'instructions': ins, 'criteria': list(levels)},
                target=target, support_index=0, support_flip=flip, kind=kind,
                polarity='neg' if flip else 'pos')


def comparison_views(lang, x, y, task, g, rng=None, templates=None):
    """Views of "x is more suitable than y for task", whose gold support is g."""
    t = templates or COMPARE[lang]
    fill = lambda s, a=x, b=y: s.format(x=a, y=b, task=task)
    return [
        _choice(fill(_pick(rng, t['more'])), (x, y), [g, 1 - g], 0, 'choice_more', 'pos'),
        _choice(fill(_pick(rng, t['more'])), (y, x), [1 - g, g], 1, 'choice_more', 'pos'),
        # "Less suitable": y is the answer exactly when x is the more suitable one.
        _choice(fill(_pick(rng, t['less'])), (x, y), [1 - g, g], 1, 'choice_less', 'neg'),
        _choice(fill(_pick(rng, t['less'])), (y, x), [g, 1 - g], 0, 'choice_less', 'neg'),
        _noul(fill(_pick(rng, t['gt'])), g, False, 'noul_gt'),
        _noul(fill(_pick(rng, t['gt']), y, x), g, True, 'noul_gt_reversed'),
        _noul(fill(_pick(rng, t['lt'])), g, True, 'noul_lt'),
        _noul(fill(_pick(rng, t['lt']), y, x), g, False, 'noul_lt_reversed'),
        _score(fill(_pick(rng, t['score'])), t['levels'], g, False, 'score'),
        _score(fill(_pick(rng, t['score'])), t['levels'], g, True, 'score_reversed'),
    ]


def claim_views(lang, claim, negated, g, rng=None, templates=None):
    """Views of `claim` and of its negation, whose gold support is g."""
    t = templates or CLAIM[lang]
    yes, no = t['options']
    fill = lambda s, c=claim: s.format(claim=c)
    return [
        _noul(fill(_pick(rng, t['noul'])), g, False, 'noul_claim'),
        _noul(fill(_pick(rng, t['noul']), negated), g, True, 'noul_negated'),
        _choice(fill(_pick(rng, t['choice'])), (yes, no), [g, 1 - g], 0, 'choice_claim', 'pos'),
        _choice(fill(_pick(rng, t['choice'])), (no, yes), [1 - g, g], 1, 'choice_claim', 'pos'),
        # Whether the negated claim holds: "does not hold" supports the reference claim.
        _choice(fill(_pick(rng, t['choice']), negated), (yes, no), [1 - g, g], 1, 'choice_negated', 'neg'),
        _choice(fill(_pick(rng, t['choice']), negated), (no, yes), [g, 1 - g], 0, 'choice_negated', 'neg'),
        _score(fill(_pick(rng, t['score'])), t['levels'], g, False, 'score'),
        _score(fill(_pick(rng, t['score'])), t['levels'], g, True, 'score_reversed'),
    ]


def render_state(facts, rng=None, as_json=False):
    """Facts in a (possibly permuted) order, as text or as {"facts": [...]}."""
    facts = list(facts)
    if rng is not None:
        rng.shuffle(facts)
    if as_json:
        return {'facts': facts}
    return ''.join(facts) if facts and not facts[0].isascii() else ' '.join(facts)


def aligned_support(question, probabilities, support_index, support_flip):
    """Reference-proposition support of one answered view (mirrors train.aligned_support)."""
    p = list(probabilities)
    if question['type'] == 'score':
        value = sum(i * v for i, v in enumerate(p)) / (len(p) - 1)
    else:
        value = p[support_index]
    return 1 - value if support_flip else value


def shuffled(views, rng):
    views = list(views)
    random.Random(rng.random()).shuffle(views)
    return views
