"""Held-out probes for relative suitability, polarity reversal, order stability and cross-format agreement.

  python -m xdecision.probes --checkpoint models/checkpoint --output evaluation/probes.json

Each group asks one reference proposition many ways (see contrast.py) and every answer is
mapped onto the support of that proposition. A group is consistent when all of its views land
on the same side of 0.5; `mean_range` is the spread of aligned support within a group.

The probes use developer platforms (GitHub, Hugging Face, PyPI, Docker Hub, Jira, Slack,
Kaggle, npm), shipment/refund/server/meeting/leave claims and probe-only phrasings. The
training generator (contrast_data.py) uses none of these entities or domains; the test suite
checks the separation.
"""
import argparse
import json
from collections import defaultdict

from .contrast import COMPARE, CLAIM, comparison_views, claim_views, aligned_support, render_state

PURPOSE = {
    'GitHub': ('托管源代码，提供 Git 仓库、Pull Request、Issues 和 CI',
               'hosting source code with Git repositories, pull requests, issues and CI'),
    'Hugging Face': ('托管模型权重和数据集', 'hosting model weights and datasets'),
    'PyPI': ('发布和安装 Python 包', 'publishing and installing Python packages'),
    'Docker Hub': ('存放和分发容器镜像', 'storing and distributing container images'),
    'Jira': ('跟踪任务和软件缺陷', 'tracking tasks and software bugs'),
    'Slack': ('团队即时聊天', 'instant team chat'),
    'Kaggle': ('举办数据科学竞赛和分享数据集', 'running data-science competitions and sharing datasets'),
    'npm': ('发布和安装 JavaScript 包', 'publishing and installing JavaScript packages'),
}
# (id, more suitable, less suitable, task zh/en, capability both share zh/en)
PAIRS = [
    ('github_hf_code', 'GitHub', 'Hugging Face', ('做代码仓库', 'hosting a code repository'),
     ('存放代码文件', 'store code files')),
    ('hf_github_models', 'Hugging Face', 'GitHub', ('托管模型权重', 'hosting model weights'),
     ('存放大文件', 'store large files')),
    ('pypi_dockerhub', 'PyPI', 'Docker Hub', ('发布 Python 包', 'publishing Python packages'),
     ('分发软件', 'distribute software')),
    ('dockerhub_pypi', 'Docker Hub', 'PyPI', ('分发容器镜像', 'distributing container images'),
     ('分发软件', 'distribute software')),
    ('jira_slack', 'Jira', 'Slack', ('跟踪软件缺陷', 'tracking software bugs'),
     ('记录团队讨论', 'record team discussions')),
    ('slack_jira', 'Slack', 'Jira', ('团队即时沟通', 'instant team chat'),
     ('记录团队讨论', 'record team discussions')),
    ('kaggle_npm', 'Kaggle', 'npm', ('举办数据科学竞赛', 'running data-science competitions'),
     ('托管公开文件', 'host public files')),
    ('npm_kaggle', 'npm', 'Kaggle', ('发布 JavaScript 包', 'publishing JavaScript packages'),
     ('托管公开文件', 'host public files')),
]
LANGS = ('zh', 'en')
NO_CONTEXT = ('用户正在为团队选择工具。', 'A user is choosing a tool for their team.')

# Phrasings that the training generator never emits.
HELDOUT_COMPARE = {
    'zh': dict(COMPARE['zh'], more=['{task}的话，哪个是更好的选择？'], less=['{task}的话，哪个更不合适？'],
               gt=['相比{y}，{x}更适合{task}。'], lt=['相比{y}，{x}没那么适合{task}。']),
    'en': dict(COMPARE['en'], more=['Which one should be preferred for {task}?'],
               less=['Which one should be avoided for {task}?'],
               gt=['Compared with {y}, {x} is better suited to {task}.'],
               lt=['Compared with {y}, {x} is worse suited to {task}.']),
}

# (id, claim zh/en, negation zh/en, facts that make it hold, facts that make it fail, irrelevant facts)
CLAIMS = [
    ('shipment', ('订单 {e} 已发货。', 'Order {e} has shipped.'), ('订单 {e} 尚未发货。', 'Order {e} has not shipped yet.'),
     (['仓库系统显示订单 {e} 已于今天上午交给快递公司。', '快递单号已经生成。'],
      ['The warehouse system shows order {e} was handed to the courier this morning.', 'A tracking number has been issued.']),
     (['仓库系统显示订单 {e} 仍在等待拣货。', '目前还没有生成快递单号。'],
      ['The warehouse system shows order {e} is still waiting to be picked.', 'No tracking number has been issued yet.']),
     (['订单 {e} 的收货地址在上海。'], ['Order {e} will be delivered to Shanghai.'])),
    ('refund', ('退款 {e} 已经到账。', 'Refund {e} has been credited.'), ('退款 {e} 还没有到账。', 'Refund {e} has not been credited yet.'),
     (['银行流水显示退款 {e} 已于昨天入账。'], ['The bank statement shows refund {e} was credited yesterday.']),
     (['银行流水中没有退款 {e} 的入账记录。', '商家表示退款 {e} 仍在审核中。'],
      ['The bank statement has no entry for refund {e}.', 'The merchant says refund {e} is still under review.']),
     (['退款 {e} 的金额为 89 元。'], ['Refund {e} is for 89 yuan.'])),
    ('server', ('服务器 {e} 正在运行。', 'Server {e} is running.'), ('服务器 {e} 已停机。', 'Server {e} is down.'),
     (['监控显示服务器 {e} 在过去一小时内持续响应健康检查。'], ['Monitoring shows server {e} answered every health check in the past hour.']),
     (['监控显示服务器 {e} 自凌晨起无法响应健康检查。', '运维已确认服务器 {e} 断电。'],
      ['Monitoring shows server {e} has failed health checks since midnight.', 'Operations confirmed that server {e} lost power.']),
     (['服务器 {e} 位于第三机房。'], ['Server {e} is located in the third data hall.'])),
    ('meeting', ('会议 {e} 已确认召开。', 'Meeting {e} is confirmed.'), ('会议 {e} 已取消。', 'Meeting {e} has been cancelled.'),
     (['所有参会者都已接受会议 {e} 的邀请。', '会议室已预订成功。'],
      ['All attendees accepted the invitation for meeting {e}.', 'The room booking succeeded.']),
     (['组织者发出通知：会议 {e} 取消。'], ['The organizer sent a notice: meeting {e} is called off.']),
     (['会议 {e} 的议题是季度预算。'], ['Meeting {e} is about the quarterly budget.'])),
    ('leave', ('应批准员工 {e} 的请假申请。', "Employee {e}'s leave request should be approved."),
     ('应拒绝员工 {e} 的请假申请。', "Employee {e}'s leave request should be rejected."),
     (['员工 {e} 的请假材料齐全。', '该时段部门没有排班冲突。'],
      ["Employee {e}'s leave paperwork is complete.", 'The team has no scheduling conflict in that period.']),
     (['员工 {e} 在该时段已被安排值班且无人替换。', '公司规定值班期间不得请假。'],
      ['Employee {e} is on duty in that period and nobody can cover.', 'Company policy forbids leave during on-call duty.']),
     (['员工 {e} 在公司工作了三年。'], ['Employee {e} has worked at the company for three years.'])),
]
ENTITIES = ('K-3071', 'M-5520', 'Z-9184')

PROBE_ENTITIES = set(PURPOSE)


def _cases():
    """Yield (category, group id, lang, gold, [(state, view, fact_order)])."""
    for pid, better, worse, task, shared in PAIRS:
        for li, lang in enumerate(LANGS):
            facts = [(f'{better}和{worse}都可以{shared[0]}。' if lang == 'zh'
                      else f'Both {better} and {worse} can {shared[1]}.'),
                     (f'{better}主要用于{PURPOSE[better][0]}。' if lang == 'zh'
                      else f'{better} is mainly used for {PURPOSE[better][1]}.'),
                     (f'{worse}主要用于{PURPOSE[worse][0]}。' if lang == 'zh'
                      else f'{worse} is mainly used for {PURPOSE[worse][1]}.')]
            orders = {'forward': render_state(facts), 'reversed': render_state(facts[::-1])}
            for phrasing, templates in (('trained', COMPARE[lang]), ('heldout', HELDOUT_COMPARE[lang])):
                views = comparison_views(lang, better, worse, task[li], 1.0, templates=templates)
                for v in views:
                    v['phrasing'] = phrasing
                yield ('suitability_context', f'{pid}/{lang}/{phrasing}', lang, 1.0,
                       [(state, v, order) for order, state in orders.items() for v in views])
                yield ('suitability_no_context', f'{pid}/{lang}/{phrasing}', lang, 1.0,
                       [(NO_CONTEXT[li], v, 'none') for v in views])
            feasible = [(f'{worse}可以{shared[0]}。' if lang == 'zh' else f'{worse} can {shared[1]}.', 1.0),
                        (f'{worse}主要用于{task[0]}。' if lang == 'zh' else f'{worse} is mainly used for {task[1]}.', 0.0),
                        (f'{better}主要用于{task[0]}。' if lang == 'zh' else f'{better} is mainly used for {task[1]}.', 1.0)]
            for i, (claim, g) in enumerate(feasible):
                view = dict(question={'type': 'noul', 'instructions': claim}, support_index=1,
                            support_flip=False, kind='noul_feasibility', polarity='pos', phrasing='trained')
                yield ('feasibility', f'{pid}/{lang}/{i}', lang, g, [(orders['forward'], view, 'forward')])
    for cid, claim, negated, holds, fails, irrelevant in CLAIMS:
        for e in ENTITIES:
            for li, lang in enumerate(LANGS):
                for g, facts in ((1.0, holds[li]), (0.0, fails[li]), (0.5, irrelevant[li])):
                    facts = [f.format(e=e) for f in facts]
                    views = claim_views(lang, claim[li].format(e=e), negated[li].format(e=e), g,
                                        templates=CLAIM[lang])
                    for v in views:
                        v['phrasing'] = 'trained'
                    orders = {'forward': render_state(facts), 'reversed': render_state(facts[::-1])}
                    category = 'claim_undetermined' if g == 0.5 else 'claim_negation'
                    yield (category, f'{cid}/{e}/{lang}/{g}', lang, g,
                           [(state, v, order) for order, state in orders.items() for v in views])


def answer_probabilities(answer):
    if answer['type'] == 'noul':
        return [1 - answer['noul'], answer['noul']]
    return list(answer['probabilities'].values())


def run(model, cases):
    rows = []
    for category, gid, lang, gold, views in cases:
        by_state = defaultdict(list)
        vids = {}
        for state, view, order in views:
            vids.setdefault(id(view), len(vids))
            by_state[state].append((view, order))
        out = []
        for state, entries in by_state.items():
            questions = {f'q{i}': v['question'] for i, (v, _) in enumerate(entries)}
            answers = model.predict(state, questions)['answers']
            for i, (v, order) in enumerate(entries):
                p = answer_probabilities(answers[f'q{i}'])
                s = aligned_support(v['question'], p, v['support_index'], v['support_flip'])
                out.append(dict(view=vids[id(v)], kind=v['kind'], polarity=v['polarity'],
                                phrasing=v['phrasing'], order=order, support=round(s, 4)))
        rows.append(dict(category=category, group=gid, lang=lang, gold=gold, views=out))
    return rows


def _mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def summarize(rows):
    def agg(groups):
        views = [(g['gold'], v) for g in groups for v in g['views']]
        det = [(gold, v) for gold, v in views if gold != 0.5]
        ranges = [max(v['support'] for v in g['views']) - min(v['support'] for v in g['views'])
                  for g in groups]
        out = {'groups': len(groups), 'views': len(views)}
        if det:
            correct = [(v['support'] > 0.5) == (gold == 1) for gold, v in det]
            out['view_accuracy'] = _mean(correct)
            out['group_all_correct'] = _mean(all((v['support'] > 0.5) == (g['gold'] == 1) for v in g['views'])
                                             for g in groups if g['gold'] != 0.5)
            by = defaultdict(list)
            for (gold, v), ok in zip(det, correct):
                by[v['kind']].append(ok)
            out['accuracy_by_kind'] = {k: _mean(x) for k, x in sorted(by.items())}
            pol = defaultdict(list)
            for (gold, v), ok in zip(det, correct):
                pol[v['polarity']].append(ok)
            out['accuracy_by_polarity'] = {k: _mean(x) for k, x in sorted(pol.items())}
        else:
            out['mean_abs_deviation_from_0.5'] = _mean(abs(v['support'] - 0.5) for _, v in views)
        if any(len(g['views']) > 1 for g in groups):
            out['mean_range'] = _mean(ranges)
            out['max_range'] = round(max(ranges), 4)
        flips = []
        for g in groups:
            ordered = defaultdict(dict)
            for v in g['views']:
                ordered[v['order']][v['view']] = v['support'] > 0.5
            if set(ordered) == {'forward', 'reversed'}:
                flips += [ordered['forward'][k] != ordered['reversed'][k] for k in ordered['forward']]
        if flips:
            out['fact_order_flip_rate'] = _mean(flips)
        return out

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r['category']].append(r)
    summary = {}
    for cat, groups in sorted(by_cat.items()):
        entry = agg(groups)
        entry['by_lang'] = {lang: agg([g for g in groups if g['lang'] == lang]) for lang in LANGS}
        phr = {p: [dict(g, views=[v for v in g['views'] if v['phrasing'] == p]) for g in groups] for p in ('trained', 'heldout')}
        phr = {p: [g for g in gs if g['views']] for p, gs in phr.items()}
        if phr['heldout']:
            entry['by_phrasing'] = {p: agg(gs) for p, gs in phr.items()}
        summary[cat] = entry
    focus = [r for r in rows if r['group'].startswith('github_hf_code/') and r['category'].startswith('suitability')]
    summary['github_hf_code'] = agg(focus)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='models/checkpoint')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--backend', default='torch', choices=['torch', 'mlx'])
    ap.add_argument('--output', required=True)
    ap.add_argument('--details', action='store_true', help='include every view in the output')
    args = ap.parse_args()
    from .runtime import load
    with load(args.checkpoint, args.device, backend=args.backend) as model:
        rows = run(model, list(_cases()))
    report = {'checkpoint': args.checkpoint, 'summary': summarize(rows)}
    if args.details:
        report['groups'] = rows
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    for cat, s in report['summary'].items():
        print(f"{cat:24s} acc={s.get('view_accuracy')} all={s.get('group_all_correct')} "
              f"range={s.get('mean_range')} max={s.get('max_range')} order_flip={s.get('fact_order_flip_rate')} "
              f"dev0.5={s.get('mean_abs_deviation_from_0.5')}")


if __name__ == '__main__':
    main()
