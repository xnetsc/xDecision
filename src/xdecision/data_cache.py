"""Validate published tokenized JSONL and build a local PyTorch cache.

  python -m xdecision.data_cache --input data/train/continuation.jsonl.gz --output work/train.pt
  python -m xdecision.data_cache --input data/train/equivalence.jsonl.gz:12000 \
      --input data/train/contrast_groups.jsonl.gz --output work/equivalence.pt

`path:N` keeps a seeded sample of about N rows; equivalence groups are sampled whole. Group
ids are renumbered per input so groups from different files never merge.
"""
import argparse
import gzip
import json
import math
import random
from pathlib import Path
import torch


def load_rows(path):
    source=Path(path)
    opener=gzip.open if source.suffix=='.gz' else open
    rows=[]
    with opener(source,'rt',encoding='utf-8') as stream:
        for number,line in enumerate(stream,1):
            row=json.loads(line)
            ids,markers,target=row['ids'],row['markers'],row['target']
            if not ids or any(type(x) is not int or x<0 for x in ids):
                raise ValueError(f'line {number}: invalid token IDs')
            if len(markers)!=len(target) or len(target)<2 or any(type(x) is not int or not 0<=x<len(ids) for x in markers):
                raise ValueError(f'line {number}: invalid markers')
            if any(type(p) not in (int,float) or not math.isfinite(p) or p<0 for p in target) or not math.isclose(sum(target),1,abs_tol=1e-5):
                raise ValueError(f'line {number}: invalid probabilities')
            if row['qtype']!={'choice':0,'score':1,'noul':2}.get(row['t']):
                raise ValueError(f'line {number}: invalid question type')
            rows.append(row)
    if not rows:
        raise ValueError('Empty dataset')
    return rows


def combine(parts, rng):
    """Concatenate (rows, n) parts, sampling n rows (whole groups) and renumbering group ids."""
    out, next_group = [], 0
    for rows, n in parts:
        grouped = 'equiv_group' in rows[0]
        if grouped:
            groups = {}
            for row in rows:
                groups.setdefault(row['equiv_group'], []).append(row)
            units = list(groups.values())
        else:
            units = [[row] for row in rows]
        if n is not None and n < len(rows):
            rng.shuffle(units)
            kept, total = [], 0
            for unit in units:
                if total >= n:
                    break
                kept.append(unit)
                total += len(unit)
            units = kept
        for unit in units:
            if grouped:
                unit = [dict(row, equiv_group=next_group) for row in unit]
                next_group += 1
            out += unit
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True,action='append',help='path or path:N (repeatable)')
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=20261008)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    parts=[]
    for spec in args.input:
        path,sep,count=spec.rpartition(':')
        if not sep or not count.isdigit():
            path,count=spec,None
        parts.append((load_rows(path),int(count) if count else None))
    rows=combine(parts,random.Random(args.seed)) if len(parts)>1 or parts[0][1] is not None else parts[0][0]
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(rows,output)
    print(f'Validated {len(rows)} records')


if __name__=='__main__':
    main()
