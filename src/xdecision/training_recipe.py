"""Shared continuation options and intact equivalence-group batching."""
import math
from collections import defaultdict

MIN_GROUP, MAX_GROUP = 6, 16


def add_recipe_arguments(parser):
    parser.add_argument('--equiv', default=None, help='tokenized equivalence groups of 6-16 views')
    parser.add_argument('--equiv-every', type=int, default=3, help='one grouped batch per N micro-batches')
    parser.add_argument('--w-consistency', type=float, default=0.0)
    parser.add_argument('--w-uncertain', type=float, default=0.0)
    parser.add_argument('--w-overconf', type=float, default=0.0)
    parser.add_argument('--w-selective', type=float, default=0.0)


def validate_recipe(args):
    for name in ('w_consistency', 'w_uncertain', 'w_overconf', 'w_selective'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if args.equiv_every < 2:
        raise ValueError('--equiv-every must be at least 2')
    if args.w_consistency > 0 and not args.equiv:
        raise ValueError('--w-consistency requires --equiv')


def recipe_metadata(args):
    return {key: getattr(args, key) for key in
            ('equiv_every', 'w_consistency', 'w_uncertain', 'w_overconf', 'w_selective')}


def grouped_batches(items, max_tokens, max_items, rng):
    groups = defaultdict(list)
    for index, item in enumerate(items):
        gid = item.get('equiv_group')
        support_index = item.get('support_index')
        if type(gid) is not int or gid < 0:
            raise ValueError('equiv_group must be a nonnegative integer')
        if type(support_index) is not int or not 0 <= support_index < len(item['target']):
            raise ValueError('support_index must identify an existing candidate')
        if type(item.get('support_flip')) is not bool:
            raise ValueError('support_flip must be explicit boolean')
        groups[gid].append(index)
    if not groups or any(not MIN_GROUP <= len(g) <= MAX_GROUP for g in groups.values()):
        raise ValueError(f'every equivalence group must have {MIN_GROUP} to {MAX_GROUP} views')
    by_len = defaultdict(list)
    for indices in groups.values():
        if {items[i]['qtype'] for i in indices} != {0, 1, 2}:
            raise ValueError('every group must cover choice, noul and score')
        supports = []
        for i in indices:
            row = items[i]
            target = row['target']
            if (len(target) < 2 or any(not math.isfinite(x) or x < 0 for x in target)
                    or not math.isclose(sum(target), 1, abs_tol=1e-5)):
                raise ValueError('invalid equivalence target distribution')
            value = (sum(j*p for j, p in enumerate(target))/(len(target)-1)
                     if row['qtype'] == 1 else target[row['support_index']])
            supports.append(1-value if row['support_flip'] else value)
        if max(supports)-min(supports) > 1e-5:
            raise ValueError('group targets disagree on reference proposition support')
        length = max(len(items[i]['ids']) for i in indices)
        padded = ((length + 63)//64)*64
        if len(indices) > max_items or padded*len(indices) > max_tokens:
            raise ValueError('batch budget cannot fit one intact equivalence group; increase max-items/max-tokens')
        by_len[(padded, len(indices))].append(indices)
    batches = []
    for (length, size), entries in by_len.items():
        rng.shuffle(entries)
        per = min(max_items//size, max_tokens//(length*size))
        for start in range(0, len(entries), per):
            batches.append([i for group in entries[start:start+per] for i in group])
    rng.shuffle(batches)
    return batches
