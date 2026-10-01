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

基于 [Laya Multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) 后训练的多语结构化决策模型。约 322M 参数，支持 `choice`、`noul`、`score`，不是聊天生成模型。


## 发布内容与兼容性

同时提供原始 checkpoint（用于直接续训）与 GGUF（用于分发和加载）：

| 文件 | 大小（十进制） | 用途 |
|---|---:|---|
| `xDecision-F16.gguf` | 704.10 MB | 原检查点无损打包；推荐继续后训练使用 |
| `xDecision-Q8_0.gguf` | 402.55 MB | 降低下载和磁盘占用，部分矩阵为 Q8_0 |

GGUF 包含全部 170 个张量，以及编码器配置、tokenizer、决策头、scorer、act/escalate 头和温度参数。F16 对当前原始检查点逐张量零误差，还原后的权重 SHA256 也完全一致。Q8_0 量化 101 个矩阵，其他张量保持原精度。

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

## 做了哪些优化

- 多语推断、阅读理解、常识和知识任务的混合后训练；加入实体—属性绑定、反事实与否定样本。
- 增加问法、事实顺序、选项键名及顺序的变化，并补充有序评分和真假判定样本，减轻部分表面形式偏置。
- 补充证据不足样本与不确定性训练目标，训练 act/escalate 头；并按题型及选项数在校准集上拟合温度。
- 保留多语词表，训练期间冻结 token embeddings，并混入通用任务回放。

## 实测效果

下表是本地统一评测器保存的历史结果；完整汇总在 GitHub 的 `evaluation/metrics.json`。数值为准确率。`上游`指本地 Laya Multilingual 对照，`前一版`指较早的后训练检查点；当前仅分发 xDecision 成品。

| 数据集 / 切片 | 题数 | 上游 | 前一版 | xDecision |
|---|---:|---:|---:|---:|
| Typed decisions，整体 | 2,000 | 35.20% | 58.75% |
| 未训练 workflow：agent_trace_observability | 500 | 28.60% | 34.40% |
| 内部分离验证集 | 3,599 | 48.85% | 70.41% |
| Emotion | 1,000 | 54.50% | 56.60% |
| SST-5 | 2,210 | 27.29% | 44.84% |
| XCOPA | 1,100 | 57.64% | 59.09% |
| XWinograd | 4,442 | 52.16%  | 62.94% |
| Belebele 子集 | 2,091 | 33.72%  | 55.38% |
| XStory | 1,649 | 57.19%  | 70.95% |
| Wikidata 派生留出 | 3,377 | 48.47%  | 90.94% |
| Wikipedia 派生留出 | 1,500 | 60.73%  | 91.47% |
| 提示注入切片 | 116 | 64.66%  | 67.24% |

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
