"""Same-machine inference parity and synchronized end-to-end timing.

Run sequentially with no other GPU workload. This checks execution fidelity,
not ground-truth model accuracy. Every output includes all decision heads and
the checkpoint's calibration. No answers or contextual states are cached.
"""
import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .runtime import load


def fixtures():
    questions = {
        'choice': dict(type='choice', instructions='Which service primarily hosts general source code?',
                       criteria={'github':'GitHub', 'hf':'Hugging Face'}),
        'noul': dict(type='noul', instructions='Hugging Face can also store source code.'),
        'score': dict(type='score', instructions='How suitable is GitHub for general code hosting?',
                      criteria=['unsuitable','partly suitable','suitable','best fit']),
    }
    facts = ['GitHub primarily hosts general software source code.',
             'Hugging Face primarily hosts AI models and datasets but can also store code.']
    states = [' '.join(facts), ' '.join(reversed(facts)), '',
              {'facts':facts}, [{'role':'user','content':' '.join(facts)}],
              ' '.join(facts)+' The literal token <mask> is part of this message.',
              'GitHub主要用于通用代码协作；Hugging Face主要发布模型和数据，也能存储代码。',
              'GitHub héberge du code. Hugging Face héberge des modèles et du code.',
              'GitHub aloja código. Hugging Face aloja modelos y también código.',
              'GitHubにはコード、Hugging Faceにはモデルとコードを保存できます。',
              'No information identifies either service.',
              ('Background information is unavailable. '*350)+' '.join(facts)]
    cases = [(f'state-{i}', s, questions) for i, s in enumerate(states)]
    cases += [('reversed-options', states[0], {
        **questions, 'choice': {**questions['choice'], 'criteria': {'hf':'Hugging Face','github':'GitHub'}},
        'score': {**questions['score'], 'criteria': list(reversed(questions['score']['criteria']))}})]
    cases += [('one-option', 'Only one candidate exists.', {'q':dict(type='choice',instructions='Select it.',criteria=['only'])}),
              ('custom-noul', 'The flag is active.', {'q':dict(type='noul',instructions='The flag is active.',labels={'false':'B','true':'A'})}),
              ('many-options', 'The selected number is 17.', {'q':dict(type='choice', instructions='Select the stated number.', criteria=[str(i) for i in range(20)])}),
              ('empty-questions', 'No questions.', {})]
    return cases


def compare(reference, candidate):
    if len(reference) != len(candidate):
        raise AssertionError('Number of cases changed')
    count = agreement = 0
    probability_error = action_error = score_error = confidence_error = 0.
    for ref, got in zip(reference, candidate):
        if ref['usage'] != got['usage'] or ref['answers'].keys() != got['answers'].keys():
            raise AssertionError('Token usage or output keys changed')
        for key, a in ref['answers'].items():
            b = got['answers'][key]
            if a['type'] == 'noul':
                pa, pb = np.array([1-a['noul'], a['noul']]), np.array([1-b['noul'], b['noul']])
            else:
                if list(a['probabilities']) != list(b['probabilities']):
                    raise AssertionError('Candidate order changed')
                pa, pb = np.array(list(a['probabilities'].values())), np.array(list(b['probabilities'].values()))
            probability_error = max(probability_error, float(np.max(np.abs(pa-pb))))
            action_error = max(action_error, abs(a['action']['act_probability']-b['action']['act_probability']))
            confidence_error = max(confidence_error, abs(a['confidence']-b['confidence']),
                                   abs(a['answer_confidence']-b['answer_confidence']))
            if a['type'] == 'score':
                score_error = max(score_error, abs(a['score']-b['score']))
            agreement += int(pa.argmax() == pb.argmax())
            count += 1
    return dict(questions=count, argmax_agreement=agreement, max_probability_error=probability_error,
                max_act_probability_error=action_error, max_score_error=score_error,
                max_confidence_error=confidence_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='models/checkpoint')
    parser.add_argument('--f16', default=None)
    parser.add_argument('--q8', default=None)
    parser.add_argument('--output', required=True)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args()
    if os.environ.get('MLX_ENABLE_TF32') != '0':
        raise ValueError('Launch this comparison with MLX_ENABLE_TF32=0 for strict FP32')
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    if args.iterations < 1 or args.warmup < 1:
        raise ValueError('iterations and warmup must be positive')
    torch.set_num_threads(4)
    cases = fixtures()
    reference = None
    report = dict(model='xDecision', machine=platform.platform(), torch=torch.__version__,
                  processor=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                  memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),
                  threads=4, iterations=args.iterations, warmup=args.warmup,
                  mlx_enable_tf32=os.environ['MLX_ENABLE_TF32'],
                  input_sha256=hashlib.sha256(json.dumps(cases,ensure_ascii=False).encode()).hexdigest(),
                  timing='Sequential backends; synchronized end-to-end predict, excluding model load. No output cache.',
                  limits='One machine/session; agreement is not accuracy. Load and first request are not fresh-process cold starts. Process RSS is cumulative across backends; MLX allocation is separate.',
                  results={})
    paths = [('torch-cpu', args.checkpoint, dict(backend='torch',device='cpu')),
             ('torch-mps', args.checkpoint, dict(backend='torch',device='mps')),
             ('mlx-f32', args.checkpoint, dict(backend='mlx',dtype='float32')),
             ('mlx-f16', args.checkpoint, dict(backend='mlx',dtype='float16'))]
    if not torch.backends.mps.is_available():
        raise RuntimeError('This paired benchmark requires Apple Silicon MPS and MLX')
    if args.f16:
        paths.append(('mlx-gguf-f16',args.f16,dict(backend='mlx')))
    if args.q8:
        paths += [('torch-gguf-q8',args.q8,dict(backend='torch',device='cpu')),
                  ('mlx-gguf-q8',args.q8,dict(backend='mlx'))]
    import mlx.core as mx
    report['mlx'] = mx.__version__
    q8_reference = None
    for name, path, options in paths:
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()
        begin = time.perf_counter()
        with load(path, **options) as model:
            load_s = time.perf_counter()-begin
            begin = time.perf_counter()
            model.predict(cases[0][1], cases[0][2])
            first_ms = 1000*(time.perf_counter()-begin)
            outputs = [model.predict(state,q) for _,state,q in cases]
            if reference is None:
                reference = outputs
            if name == 'torch-gguf-q8':
                q8_reference = outputs
            parity = compare(q8_reference if name == 'mlx-gguf-q8' else reference, outputs)
            if name.startswith('mlx') and (parity['argmax_agreement'] != parity['questions']
                    or parity['max_probability_error'] > .01
                    or parity['max_act_probability_error'] > .01):
                raise AssertionError(f'Backend fidelity check failed: {parity}')
            native = model._model
            if options['backend'] == 'mlx':
                # Verify exact input IDs against the installed reference sequence/tokenizer path.
                from .model_io import load_tokenizer
                from laya.agent import Agent
                from types import SimpleNamespace
                ref = SimpleNamespace(tok=load_tokenizer(args.checkpoint),cfg=native.cfg)
                for _, state, q in cases:
                    rows, internal = native.prepare(state,q)
                    if q and rows != Agent._encode_state(ref,state,list(q),internal):
                        raise AssertionError('Prepared tokens differ')
            timings = {}
            for workload, state, q in [('short-choice',cases[0][1],{'q':cases[0][2]['choice']}),
                                        ('short-three-types',cases[0][1],cases[0][2]),
                                        ('long-three-types',cases[11][1],cases[11][2])]:
                for _ in range(args.warmup):
                    model.predict(state,q)
                samples=[]
                for _ in range(args.iterations):
                    start=time.perf_counter()
                    model.predict(state,q)
                    samples.append(1000*(time.perf_counter()-start))
                timings[workload]=dict(p50_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)),samples_ms=samples)
            repeated=[model.predict(cases[0][1],cases[0][2]) for _ in range(10)]
            if any(x!=repeated[0] for x in repeated):
                raise AssertionError('Repeated outputs differ')
            record=dict(load_s=load_s,first_request_ms=first_ms,parity=parity,timings=timings,
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        mlx_peak_bytes=mx.get_peak_memory() if options['backend']=='mlx' else None)
            report['results'][name]=record
            print(name,json.dumps({k:v for k,v in record.items() if k!='timings'}),flush=True)
            print('latencies',json.dumps({k:round(v['p50_ms'],3) for k,v in timings.items()}),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
