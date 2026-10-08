"""Generate polarity, comparison and negation equivalence groups for continuation training.

  python -m xdecision.contrast_data --checkpoint models/checkpoint --out-dir data

Writes readable source records to data/source/ and tokenized records to data/train/:

  contrast_groups  equivalence groups (equiv_group / support_index / support_flip): one reference
                   proposition asked as "more" and "less" suitable in both candidate orders,
                   "A is more suitable than B" and "B is more suitable than A", a claim and its
                   negation, forward and reversed score scales; Chinese and English views of
                   the same proposition share a group, and every view gets its own fact order.
  contrast_items   single questions that separate feasibility ("B can also do it") and primary
                   purpose from relative suitability, plus multi-candidate "most suitable" picks.

Domains and entities are disjoint from probes.py (developer platforms; shipment, refund,
server, meeting and leave claims), so the probes measure transfer rather than recall.
"""
import argparse
import gzip
import json
import random
from pathlib import Path

from .contrast import comparison_views, claim_views, render_state

LANGS = ('zh', 'en')
GROUP_SIZE = 12

# family: (kind zh/en, capability every member shares zh/en, purposes [(zh, en)])
FAMILIES = {
    'vehicle': (('车辆', 'vehicle'), ('运输货物', 'transport goods'), [
        ('长途运输大宗货物', 'long-distance bulk freight'),
        ('在城市里配送小包裹', 'delivering small parcels in the city'),
        ('运输冷冻食品', 'transporting frozen food'),
        ('搬运大型家具', 'moving large furniture')]),
    'kitchen': (('厨房电器', 'kitchen appliance'), ('加工食物', 'prepare food'), [
        ('烘焙面包', 'baking bread'),
        ('榨取新鲜果汁', 'making fresh juice'),
        ('快速加热剩饭', 'quickly reheating leftovers'),
        ('慢炖汤品', 'slow-cooking soups')]),
    'venue': (('场馆', 'venue'), ('举办活动', 'host events'), [
        ('举办大型学术会议', 'hosting large academic conferences'),
        ('举办露天音乐节', 'hosting open-air music festivals'),
        ('举办小型婚宴', 'hosting small wedding banquets'),
        ('举办室内篮球比赛', 'hosting indoor basketball games')]),
    'storage': (('仓储设施', 'storage facility'), ('存放物品', 'store items'), [
        ('冷藏疫苗', 'refrigerating vaccines'),
        ('长期保存纸质档案', 'long-term archiving of paper records'),
        ('存放危险化学品', 'storing hazardous chemicals'),
        ('临时寄存行李', 'temporarily storing luggage')]),
    'footwear': (('鞋子', 'shoe'), ('日常穿着', 'be worn every day'), [
        ('跑马拉松', 'running marathons'),
        ('登山徒步', 'mountain hiking'),
        ('出席正式商务场合', 'formal business occasions'),
        ('雨天出行', 'walking on rainy days')]),
    'office_machine': (('办公设备', 'office machine'), ('处理纸质文件', 'handle paper documents'), [
        ('大批量彩色打印', 'high-volume color printing'),
        ('扫描并归档纸质文件', 'scanning and archiving paper documents'),
        ('粉碎机密文件', 'shredding confidential documents'),
        ('装订厚报告', 'binding thick reports')]),
    'finance': (('金融产品', 'financial product'), ('管理资金', 'manage money'), [
        ('日常小额支付', 'everyday small payments'),
        ('长期储蓄养老', 'long-term retirement saving'),
        ('跨境汇款', 'cross-border money transfers'),
        ('短期应急借款', 'short-term emergency borrowing')]),
    'fitness': (('健身器材', 'fitness machine'), ('锻炼身体', 'be used for exercise'), [
        ('提升心肺耐力', 'building cardiovascular endurance'),
        ('增加肌肉力量', 'building muscle strength'),
        ('改善身体柔韧性', 'improving flexibility'),
        ('膝关节康复训练', 'knee rehabilitation training')]),
    'learning': (('学习资源', 'learning resource'), ('帮助学习', 'support learning'), [
        ('查询生词释义', 'looking up word meanings'),
        ('练习外语口语', 'practicing spoken foreign languages'),
        ('备考数学竞赛', 'preparing for math competitions'),
        ('学习弹钢琴', 'learning to play the piano')]),
    'camera': (('拍摄设备', 'camera'), ('拍摄画面', 'capture images'), [
        ('拍摄远处的野生动物', 'photographing distant wildlife'),
        ('拍摄水下画面', 'filming underwater'),
        ('航拍大面积农田', 'aerial photography of large farmland'),
        ('拍摄证件照', 'taking ID photos')]),
    'farm': (('农业机械', 'farm machine'), ('在农田里作业', 'work in the fields'), [
        ('收割小麦', 'harvesting wheat'),
        ('喷洒农药', 'spraying pesticides'),
        ('翻耕土地', 'plowing land'),
        ('灌溉果园', 'irrigating orchards')]),
    'lighting': (('照明设备', 'light'), ('照明', 'provide light'), [
        ('照亮整个体育场', 'lighting a whole stadium'),
        ('床头阅读', 'bedside reading'),
        ('夜间户外露营', 'camping outdoors at night'),
        ('给室内植物补光', 'supplementing light for indoor plants')]),
}

# Common-sense pairs answered without context: (x zh, x en, y zh, y en, task zh, task en); x is better.
WORLD = [
    ('卡车', 'a truck', '自行车', 'a bicycle', '长途运输大量货物', 'carrying a large load over a long distance'),
    ('自行车', 'a bicycle', '卡车', 'a truck', '穿过狭窄的胡同短途出行', 'a short ride through narrow alleys'),
    ('冰箱', 'a refrigerator', '书架', 'a bookshelf', '保存新鲜食物', 'keeping food fresh'),
    ('书架', 'a bookshelf', '冰箱', 'a refrigerator', '摆放书籍', 'displaying books'),
    ('雨伞', 'an umbrella', '太阳镜', 'sunglasses', '在雨中保持干爽', 'staying dry in the rain'),
    ('太阳镜', 'sunglasses', '雨伞', 'an umbrella', '减少强光对眼睛的刺激', 'reducing glare on the eyes'),
    ('锤子', 'a hammer', '螺丝刀', 'a screwdriver', '钉钉子', 'driving nails'),
    ('螺丝刀', 'a screwdriver', '锤子', 'a hammer', '拧螺丝', 'turning screws'),
    ('飞机', 'an airplane', '货轮', 'a cargo ship', '跨洋快速出差', 'a quick business trip across the ocean'),
    ('货轮', 'a cargo ship', '飞机', 'an airplane', '跨洋运输大量集装箱', 'shipping many containers across the ocean'),
    ('电子表格', 'a spreadsheet', '演示文稿', 'a slide deck', '计算年度预算', 'calculating an annual budget'),
    ('演示文稿', 'a slide deck', '电子表格', 'a spreadsheet', '向观众展示要点', 'presenting key points to an audience'),
    ('显微镜', 'a microscope', '望远镜', 'a telescope', '观察细胞', 'observing cells'),
    ('望远镜', 'a telescope', '显微镜', 'a microscope', '观察遥远的星星', 'observing distant stars'),
    ('羽绒服', 'a down jacket', 'T恤', 'a T-shirt', '冬季户外保暖', 'keeping warm outdoors in winter'),
    ('T恤', 'a T-shirt', '羽绒服', 'a down jacket', '在炎热的夏天穿', 'wearing on a hot summer day'),
    ('订书机', 'a stapler', '剪刀', 'scissors', '把几张纸装订在一起', 'fastening sheets of paper together'),
    ('剪刀', 'scissors', '订书机', 'a stapler', '把一张纸剪成两半', 'cutting a sheet of paper in half'),
    ('温度计', 'a thermometer', '卷尺', 'a tape measure', '测量体温', 'measuring body temperature'),
    ('卷尺', 'a tape measure', '温度计', 'a thermometer', '测量房间的长度', 'measuring the length of a room'),
    ('救护车', 'an ambulance', '公交车', 'a city bus', '紧急送病人去医院', 'rushing a patient to hospital'),
    ('公交车', 'a city bus', '救护车', 'an ambulance', '日常通勤', 'everyday commuting'),
    ('保温杯', 'a vacuum flask', '纸杯', 'a paper cup', '让热水保温几个小时', 'keeping water hot for hours'),
    ('纸杯', 'a paper cup', '保温杯', 'a vacuum flask', '在活动中一次性分发饮料', 'handing out drinks at a one-off event'),
    ('字典', 'a dictionary', '小说', 'a novel', '查词义', 'looking up the meaning of words'),
    ('小说', 'a novel', '字典', 'a dictionary', '消遣阅读一个故事', 'reading a story for fun'),
    ('跑鞋', 'running shoes', '拖鞋', 'slippers', '跑马拉松', 'running a marathon'),
    ('拖鞋', 'slippers', '跑鞋', 'running shoes', '在家里放松地走动', 'relaxing around the house'),
]
WORLD_STATE = (['用户需要做出选择。', '用户正在比较{x}和{y}。', ''],
               ['A user needs to make a choice.', 'A user is comparing {x} and {y}.', ''])

# (claim zh/en, negation zh/en, facts that make it hold, facts that make it fail, irrelevant facts)
CLAIMS = [
    (('账户 {e} 已完成实名认证。', 'Account {e} has completed identity verification.'),
     ('账户 {e} 尚未完成实名认证。', 'Account {e} has not completed identity verification.'),
     (['认证系统记录显示账户 {e} 已于昨天通过审核。'], ['The verification log shows account {e} passed review yesterday.']),
     (['账户 {e} 的认证申请因证件照片模糊被驳回。', '用户还没有重新提交材料。'],
      ['The verification request for account {e} was rejected because the ID photo was blurry.',
       'The user has not resubmitted.']),
     (['账户 {e} 注册于 2021 年。'], ['Account {e} was registered in 2021.'])),
    (('图书 {e} 已归还。', 'Library book {e} has been returned.'), ('图书 {e} 仍未归还。', 'Library book {e} has not been returned.'),
     (['借还系统显示图书 {e} 已于周一归还到总馆。'], ['The circulation system shows book {e} was returned to the main branch on Monday.']),
     (['图书 {e} 已逾期十天。', '借阅人表示书还在家里。'], ['Book {e} is ten days overdue.', 'The borrower says the book is still at home.']),
     (['图书 {e} 是一本历史类书籍。'], ['Book {e} is a history title.'])),
    (('航班 {e} 已起飞。', 'Flight {e} has taken off.'), ('航班 {e} 还没有起飞。', 'Flight {e} has not taken off yet.'),
     (['塔台记录显示航班 {e} 于 08:15 离地。'], ['The tower log shows flight {e} left the ground at 08:15.']),
     (['航班 {e} 因天气原因仍在登机口等待。'], ['Flight {e} is still waiting at the gate because of the weather.']),
     (['航班 {e} 由一架窄体客机执飞。'], ['Flight {e} is operated by a narrow-body aircraft.'])),
    (('施工许可 {e} 已获批准。', 'Building permit {e} has been approved.'),
     ('施工许可 {e} 未获批准。', 'Building permit {e} has not been approved.'),
     (['审批系统显示施工许可 {e} 已签发。'], ['The approval system shows permit {e} has been issued.']),
     (['审批部门退回了施工许可 {e}，要求补充消防方案。'],
      ['The authority sent permit {e} back and asked for a fire-safety plan.']),
     (['施工许可 {e} 涉及一栋六层住宅。'], ['Permit {e} concerns a six-storey residential building.'])),
    (('合同 {e} 已签署。', 'Contract {e} has been signed.'), ('合同 {e} 尚未签署。', 'Contract {e} has not been signed.'),
     (['双方已在合同 {e} 上盖章，电子签名记录完整。'], ['Both parties stamped contract {e} and the e-signature record is complete.']),
     (['合同 {e} 仍在法务审阅中，双方都没有签字。'], ['Contract {e} is still under legal review and neither party has signed.']),
     (['合同 {e} 的期限为两年。'], ['Contract {e} runs for two years.'])),
    (('设备 {e} 已通过校准。', 'Instrument {e} has passed calibration.'),
     ('设备 {e} 未通过校准。', 'Instrument {e} has not passed calibration.'),
     (['计量报告显示设备 {e} 的误差在允许范围内。'], ['The metrology report shows instrument {e} is within tolerance.']),
     (['计量报告显示设备 {e} 的误差超出允许范围。'], ['The metrology report shows instrument {e} is out of tolerance.']),
     (['设备 {e} 放在二号实验室。'], ['Instrument {e} is kept in lab two.'])),
    (('候选人 {e} 已接受录用。', 'Candidate {e} has accepted the offer.'),
     ('候选人 {e} 没有接受录用。', 'Candidate {e} has not accepted the offer.'),
     (['候选人 {e} 签回了录用通知书。'], ['Candidate {e} signed and returned the offer letter.']),
     (['候选人 {e} 回信婉拒了这份工作。'], ['Candidate {e} wrote back to decline the job.']),
     (['候选人 {e} 应聘的是数据分析岗位。'], ['Candidate {e} applied for a data analyst role.'])),
    (('房间 {e} 已打扫干净。', 'Room {e} has been cleaned.'), ('房间 {e} 还没有打扫。', 'Room {e} has not been cleaned yet.'),
     (['保洁记录显示房间 {e} 已于中午完成清洁并通过检查。'], ['The housekeeping log shows room {e} was cleaned at noon and passed inspection.']),
     (['房间 {e} 仍在保洁待办列表中。'], ['Room {e} is still on the housekeeping to-do list.']),
     (['房间 {e} 位于五楼。'], ['Room {e} is on the fifth floor.'])),
    (('贷款申请 {e} 应予批准。', 'Loan application {e} should be approved.'),
     ('贷款申请 {e} 应予拒绝。', 'Loan application {e} should be rejected.'),
     (['贷款申请 {e} 的收入证明齐全。', '申请人信用记录良好，没有逾期。'],
      ['Loan application {e} has complete proof of income.', 'The applicant has a clean credit history with no late payments.']),
     (['贷款申请 {e} 的申请人有三笔逾期未还贷款。'], ['The applicant for loan {e} has three overdue unpaid loans.']),
     (['贷款申请 {e} 的金额为二十万元。'], ['Loan application {e} is for 200,000 yuan.'])),
    (('采购单 {e} 应当付款。', 'Purchase order {e} should be paid.'), ('采购单 {e} 不应付款。', 'Purchase order {e} should not be paid.'),
     (['采购单 {e} 的货物已验收合格。', '发票金额与合同一致。'],
      ['The goods on purchase order {e} passed inspection.', 'The invoice amount matches the contract.']),
     (['验收发现采购单 {e} 的货物短缺一半。', '供应商尚未补货。'],
      ['Inspection found half of the goods on purchase order {e} missing.', 'The supplier has not replaced them.']),
     (['采购单 {e} 采购的是办公椅。'], ['Purchase order {e} is for office chairs.'])),
    (('疫苗批次 {e} 可以放行。', 'Vaccine batch {e} can be released.'), ('疫苗批次 {e} 不能放行。', 'Vaccine batch {e} cannot be released.'),
     (['疫苗批次 {e} 的全部质检项目合格。', '冷链温度记录完整且正常。'],
      ['Every quality test for vaccine batch {e} passed.', 'The cold-chain temperature log is complete and normal.']),
     (['冷链记录显示疫苗批次 {e} 曾在 15 摄氏度下存放六小时。'],
      ['The cold-chain log shows vaccine batch {e} sat at 15 degrees Celsius for six hours.']),
     (['疫苗批次 {e} 共有一万剂。'], ['Vaccine batch {e} contains ten thousand doses.'])),
    (('仓库 {e} 的库存充足。', 'Warehouse {e} has sufficient stock.'), ('仓库 {e} 的库存不足。', 'Warehouse {e} is short of stock.'),
     (['仓库 {e} 的现有库存可以满足未来三个月的订单。'], ['Current stock in warehouse {e} covers the next three months of orders.']),
     (['仓库 {e} 的库存只够两天，已低于安全线。'], ['Warehouse {e} holds only two days of stock, below the safety level.']),
     (['仓库 {e} 位于港口附近。'], ['Warehouse {e} is located near the port.'])),
    (('桥梁 {e} 结构安全。', 'Bridge {e} is structurally safe.'), ('桥梁 {e} 存在结构安全隐患。', 'Bridge {e} is structurally unsafe.'),
     (['最新检测显示桥梁 {e} 各项指标正常，没有发现裂缝。'], ['The latest inspection found every indicator of bridge {e} normal and no cracks.']),
     (['检测发现桥梁 {e} 的主梁出现贯穿性裂缝。'], ['Inspection found a through-crack in the main girder of bridge {e}.']),
     (['桥梁 {e} 建于 1998 年。'], ['Bridge {e} was built in 1998.'])),
]

SYLLABLES = ['ver', 'lo', 'ta', 'qui', 'mar', 'nex', 'sol', 'dra', 'vi', 'ken', 'ora', 'lum', 'bel', 'cor',
             'zen', 'fi', 'tro', 'ga', 'ri', 'pol', 'sen', 'nu', 'vel', 'ka', 'tis', 'mo', 'lex', 'bra']
CITIES = (['上海', '成都', '杭州', '武汉', '西安', '青岛'], ['Lisbon', 'Osaka', 'Denver', 'Turin', 'Leeds', 'Perth'])


def entity_name(rng):
    name = ''.join(rng.choice(SYLLABLES) for _ in range(rng.choice((2, 3))))
    return name.capitalize() + rng.choice(['', '', ' Pro', ' X', ' One'])


def code(rng):
    return '%s-%04d' % (rng.choice('ABCDEFGHJLNPRSTUVWY'), rng.randrange(10000))


def _comparison_facts(lang, x, y, kind, shared, px, py, g, rng):
    zh = lang == 'zh'
    facts = []
    shared_stated = rng.random() < 0.8
    if shared_stated:
        facts.append(rng.choice([f'{x}和{y}都可以{shared}。', f'{x}与{y}都能{shared}。']) if zh else
                     rng.choice([f'Both {x} and {y} can {shared}.', f'{x} and {y} are both able to {shared}.']))
    if g != 0.5:
        for name, purpose in ((x, px), (y, py)):
            facts.append(rng.choice([f'{name}主要用于{purpose}。', f'{name}是专门为{purpose}设计的{kind}。',
                                     f'{name}的设计目标是{purpose}。']) if zh else
                         rng.choice([f'{name} is mainly used for {purpose}.',
                                     f'{name} is a {kind} designed specifically for {purpose}.',
                                     f'{name} was built for {purpose}.']))
    year, city = rng.randrange(1995, 2024), rng.choice(CITIES[0 if zh else 1])
    distractors = ([f'{x}于{year}年推出。', f'{y}的总部位于{city}。', f'{x}的价格比{y}略高。'] if zh else
                   [f'{x} launched in {year}.', f'{y} is headquartered in {city}.', f'{x} costs slightly more than {y}.'])
    facts += rng.sample(distractors, 3 if g == 0.5 else rng.randrange(0, 2))
    return facts, shared_stated


def _group(views, rng, size=GROUP_SIZE):
    """A random subset that keeps both languages and all three question types."""
    for _ in range(100):
        pick = rng.sample(views, min(size, len(views)))
        if ({v['question']['type'] for v in pick} == {'choice', 'noul', 'score'}
                and len({v['language'] for v in pick}) == len({v['language'] for v in views})):
            return pick
    raise RuntimeError('could not draw a balanced group')


def comparison_group(rng, families=FAMILIES):
    family = rng.choice(sorted(families))
    (kind, shared, purposes) = families[family]
    (pi, pj) = rng.sample(range(len(purposes)), 2)
    x, y = entity_name(rng), entity_name(rng)
    while y == x:
        y = entity_name(rng)
    g = 0.5 if rng.random() < 0.15 else rng.choice((0.0, 1.0))
    # g = 1: the task is x's purpose; g = 0: the task is y's purpose; g = 0.5: purposes are not stated.
    task_index = pj if g == 0 else pi
    views, extra = [], []
    for li, lang in enumerate(LANGS):
        task = purposes[task_index][li]
        px, py = purposes[pi][li], purposes[pj][li]
        facts, shared_stated = _comparison_facts(lang, x, y, kind[li], shared[li], px, py, g, rng)
        for v in comparison_views(lang, x, y, task, g, rng):
            v['state'] = render_state(facts, rng, as_json=rng.random() < 0.3)
            v['language'] = lang
            views.append(v)
        if g != 0.5:
            extra += feasibility_items(lang, x, y, shared[li], task, g, facts, shared_stated, rng)
    return f'compare/{family}', views, extra


def feasibility_items(lang, x, y, shared, task, g, facts, shared_stated, rng):
    """Feasibility and primary purpose are separate propositions from relative suitability."""
    better, worse = (x, y) if g == 1 else (y, x)
    zh = lang == 'zh'
    claims = []
    if shared_stated:
        claims.append((f'{worse}可以{shared}。' if zh else f'{worse} can {shared}.', 1.0))
    claims += [(f'{worse}主要用于{task}。' if zh else f'{worse} is mainly used for {task}.', 0.0),
               (f'{better}主要用于{task}。' if zh else f'{better} is mainly used for {task}.', 1.0)]
    items = [dict(state=render_state(facts, rng), question={'type': 'noul', 'instructions': c},
                  target=[1 - t, t], language=lang) for c, t in claims]
    ins = f'哪一个是专门为{task}设计的？' if zh else f'Which one is designed specifically for {task}?'
    order = [better, worse] if rng.random() < 0.5 else [worse, better]
    items.append(dict(state=render_state(facts, rng), language=lang,
                      question={'type': 'choice', 'instructions': ins, 'criteria': {k: '' for k in order}},
                      target=[float(k == better) for k in order]))
    return items


def world_group(rng):
    xz, xe, yz, ye, tz, te = rng.choice(WORLD)
    g = rng.choice((0.0, 1.0))
    # Reference proposition alternates direction so "the first-named option wins" is never a shortcut.
    names = {'zh': (xz, yz) if g == 1 else (yz, xz), 'en': (xe, ye) if g == 1 else (ye, xe)}
    views = []
    for li, lang in enumerate(LANGS):
        a, b = names[lang]
        task = (tz, te)[li]
        state = rng.choice(WORLD_STATE[li]).format(x=a, y=b)
        for v in comparison_views(lang, a, b, task, g, rng):
            v['state'] = state
            v['language'] = lang
            views.append(v)
    return 'compare/world', views, []


def claim_group(rng):
    claim, negated, holds, fails, irrelevant = rng.choice(CLAIMS)
    e = code(rng)
    r = rng.random()
    g = 1.0 if r < 0.42 else 0.0 if r < 0.84 else 0.5
    views = []
    for li, lang in enumerate(LANGS):
        if g == 1:
            facts = holds[li] + rng.sample(irrelevant[li], rng.randrange(0, 2))
        elif g == 0:
            facts = fails[li] + rng.sample(irrelevant[li], rng.randrange(0, 2))
        elif rng.random() < 0.5:
            facts = list(irrelevant[li])
        else:
            # Conflicting sources do not settle the claim.
            facts = ([f'记录甲：{holds[li][0]}', f'记录乙：{fails[li][0]}'] if lang == 'zh' else
                     [f'Record A: {holds[li][0]}', f'Record B: {fails[li][0]}'])
        facts = [f.format(e=e) for f in facts]
        for v in claim_views(lang, claim[li].format(e=e), negated[li].format(e=e), g, rng):
            v['state'] = render_state(facts, rng, as_json=rng.random() < 0.3)
            v['language'] = lang
            views.append(v)
    return 'claim', views, []


def most_suitable_item(rng):
    family = rng.choice(sorted(FAMILIES))
    kind, shared, purposes = FAMILIES[family]
    k = rng.choice((3, 4))
    names = []
    while len(names) < k:
        n = entity_name(rng)
        if n not in names:
            names.append(n)
    target = rng.randrange(k)
    items = []
    for li, lang in enumerate(LANGS):
        facts = [f'{n}主要用于{purposes[i][li]}。' if lang == 'zh' else f'{n} is mainly used for {purposes[i][li]}.'
                 for i, n in enumerate(names)]
        facts.append(f'它们都可以{shared[li]}。' if lang == 'zh' else f'All of them can {shared[li]}.')
        task = purposes[target][li]
        ins = f'哪个最适合{task}？' if lang == 'zh' else f'Which is the most suitable for {task}?'
        order = rng.sample(range(k), k)
        items.append(dict(state=render_state(facts, rng), language=lang,
                          question={'type': 'choice', 'instructions': ins,
                                    'criteria': {names[i]: '' for i in order}},
                          target=[float(i == target) for i in order]))
    return items


def generate(n_compare, n_world, n_claim, n_most, seed):
    rng = random.Random(seed)
    groups, items = [], []
    makers = [comparison_group] * n_compare + [world_group] * n_world + [claim_group] * n_claim
    for gid, make in enumerate(makers):
        source, views, extra = make(rng)
        for v in _group(views, rng):
            groups.append(dict(state=v['state'], question=v['question'], target=v['target'],
                               equiv_group=gid, support_index=v['support_index'], support_flip=v['support_flip'],
                               source=f'contrast_{source.split("/")[0]}', language=v['language'], kind=v['kind']))
        items += [dict(row, source='contrast_feasibility') for row in extra]
    for _ in range(n_most):
        items += [dict(row, source='contrast_most_suitable') for row in most_suitable_item(rng)]
    rng.shuffle(items)
    return groups, items


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 keeps the archive byte-identical across regenerations.
    with open(path, 'wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
        for row in rows:
            gz.write((json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='models/checkpoint')
    ap.add_argument('--out-dir', default='data')
    ap.add_argument('--compare', type=int, default=3000, help='fact-based comparison groups')
    ap.add_argument('--world', type=int, default=600, help='common-sense comparison groups without context')
    ap.add_argument('--claims', type=int, default=1800, help='claim/negation groups')
    ap.add_argument('--most', type=int, default=800, help='multi-candidate "most suitable" questions per language')
    ap.add_argument('--seed', type=int, default=20261008)
    args = ap.parse_args()
    from .model_io import load_config, load_tokenizer
    from .prepare import encode_records
    groups, items = generate(args.compare, args.world, args.claims, args.most, args.seed)
    cfg, tok = load_config(args.checkpoint), load_tokenizer(args.checkpoint)
    out = Path(args.out_dir)
    for name, rows in (('contrast_groups', groups), ('contrast_items', items)):
        _write(out/'source'/f'{name}.jsonl.gz', rows)
        encoded = encode_records(enumerate(rows, 1), cfg, tok)
        for i, enc in enumerate(encoded):
            enc['original_row'] = i
        _write(out/'train'/f'{name}.jsonl.gz', encoded)
        print(f'{name}: {len(rows)} rows')


if __name__ == '__main__':
    main()
