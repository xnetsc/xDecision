"""Validate published tokenized JSONL and build a local PyTorch cache."""
import argparse
import gzip
import json
import math
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows=load_rows(args.input)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(rows,output)
    print(f'Validated {len(rows)} records')


if __name__=='__main__':
    main()
