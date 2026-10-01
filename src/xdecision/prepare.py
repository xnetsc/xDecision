"""Validate user-supplied JSONL and encode it for continuation training."""
import argparse
import json
import math
from pathlib import Path
import torch
from laya.common import build_sequence, QTYPES
from .model_io import load_config, load_tokenizer


def prepare(path, checkpoint):
    cfg = load_config(checkpoint)
    tok = load_tokenizer(checkpoint)
    rows = []
    for number, text in enumerate(Path(path).read_text().splitlines(), 1):
        if not text.strip():
            continue
        row = json.loads(text)
        q = row['question']
        kind = q['type']
        if kind not in QTYPES:
            raise ValueError(f'line {number}: invalid question type')
        criteria = q.get('criteria')
        if kind == 'choice' and not isinstance(criteria, dict):
            raise ValueError(f'line {number}: choice criteria must be a mapping')
        if kind == 'score' and not isinstance(criteria, list):
            raise ValueError(f'line {number}: score criteria must be ordered')
        target = row['target']
        if not isinstance(target,list) or len(target)<2 or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in target):
            raise ValueError(f'line {number}: invalid target distribution')
        if not math.isclose(sum(target),1.0,abs_tol=1e-5):
            raise ValueError(f'line {number}: target must sum to 1')
        internal = {'t':kind,'ins':q['instructions'],'crit':criteria}
        ids, positions = build_sequence(tok,row['state'],internal,cfg['max_len'],cfg['head_max_len'])
        full, full_positions = build_sequence(tok,row['state'],internal,16384,8192)
        if ids != full or positions != full_positions:
            raise ValueError(f'line {number}: input would be truncated')
        if len(positions) != len(target):
            raise ValueError(f'line {number}: labels/options differ')
        rows.append(dict(ids=ids,markers=positions,qtype=QTYPES[kind],target=target,
                         src=row.get('source','custom'),lang=row.get('language','en'),t=kind))
        if any(key in row for key in ('equiv_group', 'support_index', 'support_flip')):
            if (type(row.get('equiv_group')) is not int or row['equiv_group'] < 0
                    or type(row.get('support_index')) is not int
                    or not 0 <= row['support_index'] < len(target)
                    or type(row.get('support_flip')) is not bool):
                raise ValueError(f'line {number}: invalid equivalence metadata')
            for key in ('equiv_group', 'support_index', 'support_flip'):
                rows[-1][key] = row[key]
        if 'label' in row:
            if type(row['label']) is not int or not 0 <= row['label'] < len(target):
                raise ValueError(f'line {number}: invalid label')
            rows[-1]['label'] = row['label']
        if 'determinate' in row:
            if type(row['determinate']) is not bool:
                raise ValueError(f'line {number}: determinate must be boolean')
            rows[-1]['determinate'] = row['determinate']
    if not rows:
        raise ValueError('No examples')
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    output=Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    rows=prepare(a.input,a.checkpoint)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(rows,output)
    print(f'Encoded {len(rows)} examples')


if __name__=='__main__':
    main()
