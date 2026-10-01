---
license: apache-2.0
base_model:
  - convaiinnovations/laya-multilingual
  - jhu-clsp/mmBERT-base
base_model_relation: finetune
tags:
  - gguf
  - text-classification
  - multilingual
  - decision-model
---

# xDecision

基于 [Laya Multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) 后训练的多语结构化决策模型，编码器源自 [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)。约 322M 参数，支持 `choice`、`noul`、`score`。不是聊天生成模型，也不是从零预训练的模型。

**这是有明确局限的实验性发布。** 简单事实判读和多个留出集有所改善，但相对适配度排序、跨表达稳定性、未见工作流迁移和不确定性控制尚未全部解决，不适合未经验证地自动执行高风险决策。

## 发布内容与兼容性

同时提供原始 checkpoint（用于直接续训）与 GGUF（用于分发和加载）：

| 文件 | 大小（十进制） | 用途 |
|---|---:|---|
| `xDecision-F16.gguf` | 704.10 MB | 原检查点无损打包；推荐继续后训练使用 |
| `xDecision-Q8_0.gguf` | 402.55 MB | 降低下载和磁盘占用，部分矩阵为 Q8_0 |

GGUF 包含全部 170 个张量，以及编码器配置、tokenizer、决策头、scorer、act/escalate 头和温度参数。F16 对当前原始检查点逐张量零误差，还原后的权重 SHA256 也完全一致。Q8_0 量化 101 个矩阵，其他张量保持原精度。

**架构标识为自定义 `xdecision`，不是通用 Llama/BERT GGUF。不能假定 llama.cpp、Ollama 或任意 GGUF 查看器能执行此模型。** 本仓库提供的参考加载器将 GGUF 还原到临时目录，再用 PyTorch/Laya 执行；Q8_0 也会解量化，因此不能把文件变小当作运行显存下降或原生整数推理加速。

GitHub 与 Hugging Face 使用相同的发布目录结构；大文件使用 Git LFS 分发。

原 checkpoint 完整提供。**训练数据仅分发许可和归属已核实的部分，不是完整原混合数据**：最后阶段原有 139,680 行，随包 49,801 行；六视图等价组 36,000 行完整提供。此前阶段、校准和内部留出也分别保存可分发子集；具体缺项见 `data/manifest.json`。不以此子集声称能够精确复现成品成绩。

```text
models/
  checkpoint/       原始权重、编码器配置、tokenizer 与推理配置
  gguf/             xDecision-F16.gguf、xDecision-Q8_0.gguf
data/               训练、校准与测试数据分开存放，附来源和许可说明
configs/            数据配比与训练配置
src/xdecision/      运行、数据转换、训练、评测与导出代码
tests/              代码测试
examples/           最小运行与训练示例
evaluation/         准确率、资源与量化验证汇总
```

根目录保留 README、安装配置和许可证。模型目录不混入实验报告、优化器临时文件或工作记录；数据以经过许可核查的发布清单为准，不将评测数据并入训练集。

## 做了哪些优化

- 多语推断、阅读理解、常识和知识任务的混合后训练；加入实体—属性绑定、反事实与否定样本。
- 增加问法、事实顺序、选项键名及顺序的变化，并补充有序评分和真假判定样本，减轻部分表面形式偏置。
- 补充证据不足样本与不确定性训练目标，训练 act/escalate 头；并按题型及选项数在校准集上拟合温度。
- 保留多语词表，训练期间冻结 token embeddings，并混入通用任务回放。

这些是已实施的训练措施，**不是每项机制都已得到独立消融证明，也不表示对应问题已经普遍解决**。

## 实测效果

下表是本地统一评测器保存的历史结果；完整汇总在 GitHub 的 `evaluation/metrics.json`。数值为准确率。`上游`指本地 Laya Multilingual 对照，`前一版`指较早的后训练检查点；当前仅分发 xDecision 成品。

| 数据集 / 切片 | 题数 | 上游 | 前一版 | xDecision |
|---|---:|---:|---:|---:|
| Typed decisions，整体 | 2,000 | 35.20% | 56.40% | 58.75% |
| 未训练 workflow：agent_trace_observability | 500 | 28.60% | 27.40% | 34.40% |
| 内部分离验证集 | 3,599 | 48.85% | 70.08% | 70.41% |
| Emotion | 1,000 | 54.50% | 55.50% | 56.60% |
| SST-5 | 2,210 | 27.29% | 42.81% | 44.84% |
| XCOPA | 1,100 | 57.64% | 56.55% | 59.09% |
| XWinograd | 4,442 | 52.16% | 62.63% | 62.94% |
| Belebele 子集 | 2,091 | 33.72% | 53.75% | 55.38% |
| XStory | 1,649 | 57.19% | 70.04% | 70.95% |
| Wikidata 派生留出 | 3,377 | 48.47% | 90.55% | 90.94% |
| Wikipedia 派生留出 | 1,500 | 60.73% | 91.60% | 91.47% |
| 提示注入切片 | 116 | 64.66% | 68.10% | 67.24% |

Typed decisions 整体包含训练涉及的 workflow 类型，**58.75% 不是完全未见工作流的零样本成绩**；相应独立 workflow 只有 34.40%。知识类派生留出也不能替代独立通识考试。表中整体准确率以最高概率选项计分，不是将 score 期望值四舍五入；score MAE 等另在指标文件记录。

已改善：上述任务中的实际判读准确率、原有基础定向探针（28/29 通过），以及部分可判定性/校准指标。尚不能声称所有原问题解决。历史基础探针经过迭代开发，不应当作全新盲测。

### 准确度、耗时和资源

同一台 Apple M5、CPU FP32、四线程、短 choice 请求、2 次预热后 20 次计时：

| 指标 | xDecision 原检查点 | NanoJev 官方检查点 |
|---|---:|---:|
| 基础定向探针 | 28/29 | 12/29 |
| 相对程度探针 | 7/16 | 11/16 |
| 暖请求延迟中位数 | 24.52 ms | 120.81 ms |
| P95 延迟 | 24.91 ms | 122.43 ms |
| 加载耗时 | 1.81 s | 5.23 s |
| 整个测试进程峰值 RSS | 3.47 GB | 5.24 GB |

这是同题诊断，不是通用榜单：xDecision 在基础题和资源开销上更好，NanoJev 在相对程度题上更好；两者在部分有事实的 GitHub/HF 评分题上都仍随事实顺序翻转。延迟不是饱和批量吞吐，RSS 包含加载和测试过程，不是纯权重或GPU显存，也不包含本发布包的 GGUF 首次解包时间。

Q8_0 解量化后完成全部 15 个本地评测套件：Typed decisions 58.75%→58.40%，XCOPA 59.09%→58.64%，内部验证 70.41%→70.13%；所有有准确率的套件下降均不超过 1 个百分点。逐张量检查与全套量化对照见 `evaluation/validation.json`，不把量化检查当成核心遗留问题已解决。

上游模型卡另报告 T4 单题 32.8 ms、typed 零样本约 34.2%；硬件、题目、检查点及推理流程不同，只作背景，不拿来计算加速比或直接排名。见[上游模型卡](https://huggingface.co/convaiinnovations/laya-multilingual)。

## 遗留问题

1. **可行不等于最适合。** GitHub/HF 都能存代码，但针对主要用途的最高/最低相对排序仍不可靠；冻结比较探针仅 7/16。
2. **事实与选项顺序敏感仍在。** 基础题的局部改善不代表换表达、换领域或换评分形式也稳定。
3. **choice/noul/score 语义不能随意对齐。** 可行性真假、主要用途和适配度不是同一个命题；已测跨题型支持度仍有明显差距。
4. **置信度与弃权没有全面达标。** 整体 Typed ECE 为 0.1236、act AUROC 为 0.6583；未见工作流 act AUROC 约 0.4926。内部验证集相应为 0.0306、0.7520，不能代替 OOD 验证。注入切片仍有高置信错误。
5. **泛化有限且并非全面优于前一版。** 未见 workflow 34.40%；Wikipedia 和注入切片略有回退。不可把简单删事实样本上的低置信度解释为所有未知输入都能正确弃权。

使用前在自己的领域验证准确率、概率校准、顺序稳定性及风险—覆盖率；涉及不可逆操作时保留人工复核。

## 运行

```bash
git clone https://github.com/xnetsc/xDecision.git
cd xDecision
git lfs pull
python -m pip install -e .
```

```python
import xdecision

with xdecision.load("models/gguf/xDecision-F16.gguf", device="cpu") as model:
    result = model.predict(
        "The delivery arrived yesterday.",
        {"status": {"type": "choice", "instructions": "Select the delivery status.",
                    "criteria": {"arrived": "Delivered", "pending": "Not yet delivered"}}},
    )
    print(result)
```

也可先将 GGUF 还原到本机工作目录：

返回值保留上游语义：`probabilities` 是选项概率，`answer_confidence` 是上游定义的回答置信度，`confidence` 是兼容字段，不可不加区分地当作正确率或重复校准。`action.act_probability` 是单独的执行头输出。参考封装仅将返回的模型名统一为 xDecision，不重新排序、平均或改写概率；这些字段仍有上文列出的校准局限。

```bash
python -m xdecision.gguf_io restore --gguf models/gguf/xDecision-F16.gguf --output work/model
```

无需另找原始权重或 tokenizer。生成的中间文件只用于本机运行/训练，不是额外的发布模型格式。参考运行环境：Python 3.12、PyTorch 2.14、Transformers 5.17、Laya 0.3.21；其他组合需自行验证。当前输入预算 1,024 tokens、问题与选项预算 256 tokens，超长输入应先分段并检查截断。

## 继续训练

仓库包含数据转换、训练、校准、评测和 GGUF 导出代码。它支持从现有权重继续后训练，**不包含原训练优化器/RNG状态，也不保证精确复现原始训练**；示例数据只是验证流程，不是可用的正式训练集。

JSONL 每行使用 `state`、`question`、`target`。target 按 choice 的 criteria 插入顺序、score 刻度顺序或 noul 的 false/true 顺序排列，必须为合法分布且总和为 1。适配度级别与答案置信度不要混为一谈。领域、实体与场景应先拆分训练/校准/测试，不能把同事实的改写分到不同集合后当独立测试。

```bash
python -m xdecision.prepare --input examples/train.jsonl --checkpoint models/checkpoint --output work/train.pt
python -m xdecision.train --base models/checkpoint --train work/train.pt --out work/continued \
  --device cpu --epochs 1 --max-items 2 --max-tokens 512 --accum 1 --save-every 0
```

默认保留已有 act 头，仅显式 `--reset-act-head` 时重新初始化。续训输出会清除过期温度校准，需用新的独立校准集重新拟合，不能直接沿用旧置信度：

```bash
python -m xdecision.prepare --input your-calibration.jsonl --checkpoint work/continued/final --output work/eval/calib.pt
python -m xdecision.prepare --input your-test.jsonl --checkpoint work/continued/final --output work/eval/val.pt
python -m xdecision.evaluate --ckpt work/continued/final --data work/eval --calibrate --suites val --device cpu --threads 4
python -m xdecision.gguf_io export --checkpoint work/continued/final --output work/xDecision-F16.gguf --quantization F16
python -m xdecision.gguf_io verify --checkpoint work/continued/final --gguf work/xDecision-F16.gguf
```

可直接使用 `models/checkpoint` 续训，不必先解包 GGUF。权重与原始检查点逐字节一致；配置仅整理名称和发布所需字段，保留编码器、决策头、温度和 tokenizer。没有历史优化器状态，不能精确断点恢复。不要执行不可信来源的 PyTorch 数据缓存；用仓库的 JSONL 转换器生成自己的缓存。F16 还原后两步续训已完成，但这不是准确率提升证据。

发布的实际分词数据用 `python -m xdecision.data_cache --input data/train/continuation.jsonl.gz --output work/train.pt` 转成本机缓存，然后可用 `python -m xdecision.train @configs/continue.args` 续训。Apple Silicon 可安装 `pip install -e '.[apple]'` 并使用 `xdecision.mlx_train`；该入口保留分组一致性、不确定性与选择性风险目标，通过 `--equiv`、`--equiv-every`、`--w-consistency`、`--w-uncertain`、`--w-overconf`、`--w-selective` 显式设置。普通 PyTorch 入口不含这些额外分组损失，不应将两个入口混称为相同配方。

## 来源与许可

xDecision 是上述上游模型的后训练衍生作品，代码和模型发布许可 Apache-2.0；mmBERT-base 的 MIT 许可与相关归属继续适用。参阅 `LICENSE`、`NOTICE` 和上游许可。数据遵循各自来源许可，不自动适用代码许可证；不能再分发的来源只提供获取说明或构建脚本。文件校验值见 `checksums.sha256`。
