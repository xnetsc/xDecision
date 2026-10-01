---
license: apache-2.0
language:
  - multilingual
pipeline_tag: text-classification
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

[English](#english) | [中文](#中文)

## English

[切换到中文](#中文)

xDecision is a multilingual structured decision model with approximately 322M parameters, post-trained from [Laya Multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) with an [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) encoder.

- `choice`: select among candidates.
- `noul`: assess whether a statement is true or false.
- `score`: assign a score on an ordered scale.
- Returns candidate probabilities and an act/escalate recommendation.

The release includes the original checkpoint, F16/Q8_0 GGUF files, redistributable training data, and continued-training code for CUDA, Apple MPS, and CPU. Apple Silicon also has a native MLX inference backend.

### Intended use

Classification, workflow routing, statement assessment, and ordered scoring with explicit context and candidates. Inputs are factual context and structured questions; outputs are candidate probabilities and action recommendations. Calibrate and validate on the target domain before deployment, with human review for insufficient evidence or high-risk decisions.

### Quick start

```bash
git clone https://github.com/xnetsc/xDecision.git
cd xDecision
git lfs pull
python -m pip install -e .
```

For NVIDIA hardware, first install a driver-compatible CUDA build from the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

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

The input limit is 1,024 tokens, with a 256-token budget for the question and candidates. `probabilities` contains candidate probabilities; `action.act_probability` is the action-head output. `answer_confidence` and the compatibility field `confidence` retain their upstream definitions and should be interpreted separately.

### Native Apple Silicon inference

Install `pip install -e '.[apple]'` and select the MLX backend:

```python
with xdecision.load("models/gguf/xDecision-F16.gguf", backend="mlx") as model:
    result = model.predict(state, questions)
```

MLX defaults to Apple GPU FP16, with weights resident in memory. It reads checkpoint directories and complete GGUF files directly, without writing restored checkpoints. Q8_0 is dequantized once at load into dense FP16 arrays. `batch_size=8` reduces question-batch memory. The default `backend="torch"` preserves CPU/CUDA/MPS operation.

For strict FP32, launch Python with `MLX_ENABLE_TF32=0` set before importing MLX, then pass `dtype="float32"`. [MLX's default FP32 matrix operations may use reduced internal precision](https://ml-explore.github.io/mlx/build/html/usage/precision.html); changing the array dtype alone is insufficient.

The two backends use the same prompt construction and answer schema. MLX applies stored calibration temperatures unchanged and retains the action head; it does not cache answers or reuse contextual encoder states across questions. Shared state is tokenized once per request. Loading time and first-request initialization are separate from warm inference latency.

Apple M5, 24 GiB, same checkpoint, 5 warm-ups and 30 timed requests per workload; sequential, synchronized end-to-end inference, `MLX_ENABLE_TF32=0`:

| Backend | Short choice median / P95 | Short choice+noul+score median | Three 1,024-token questions median |
|---|---:|---:|---:|
| PyTorch CPU, 4 threads | 25.01 / 26.35 ms | 59.27 ms | 1,033.57 ms |
| PyTorch MPS | 14.11 / 15.09 ms | 28.99 ms | 424.48 ms |
| MLX strict FP32 | 8.65 / 9.25 ms | 27.44 ms | 408.21 ms |
| MLX FP16 | **5.21 / 6.07 ms** | **11.03 ms** | **125.09 ms** |

Across 42 compatibility questions, all backends retained the CPU reference's top option. Strict FP32 outputs matched to four decimal places; FP16's largest candidate-probability difference was 0.003. F16 and Q8_0 GGUF loading also passed; the Q8_0 comparison uses its own PyTorch-restored reference. This measures backend fidelity, not improved model accuracy. MLX peak allocation across these workloads was 1.42 GB for FP16; this excludes other process memory. Checkpoint loading took 0.43 s and F16 GGUF loading 14.32 s in this session; they are excluded from warm latency. These are not fresh-process cold-start measurements. [Full measurements](evaluation/native_mlx.json).

Reproduce from the repository root:

```bash
MLX_ENABLE_TF32=0 python -m xdecision.benchmark_runtime \
  --checkpoint models/checkpoint --f16 models/gguf/xDecision-F16.gguf \
  --q8 models/gguf/xDecision-Q8_0.gguf --output /tmp/xdecision-runtime.json
```

### Training improvements

- Mixed multilingual inference, reading comprehension, commonsense, and knowledge training, including entity–attribute binding, counterfactuals, and negation.
- Fact-order and candidate-order variations, paraphrases, and all three question types.
- Insufficient-evidence examples, grouped consistency and uncertainty objectives, and act/escalate-head training.
- Frozen multilingual token embeddings, general-task replay, and temperature fitting by question type and candidate count.

### Cross-format consistency

The released weights were trained with a shared proposition-support constraint across `choice`, `noul`, and `score`. For equivalent views of the **same proposition**, training maps the positive choice probability, the noul true probability, and the normalized expected score onto a common support scale. Candidate permutations and reversed score scales are aligned before minimizing the mean squared support difference within each group. This is a training loss, not a post-processing rewrite of inference probabilities.

On a small diagnostic set of six fact scenarios with 24 variant groups, the mean cross-format support range decreased from **0.273 to 0.135** after the combined post-training recipe; the final maximum range was **0.430**. This is not an isolated ablation of the consistency loss, and disagreement remains. A score expectation is a semantic-support proxy, not a calibrated probability. Feasibility, primary purpose, and relative suitability are different propositions and must not be forced to have equal support.

### Evaluation

Accuracy measured with the same local evaluator is shown below. Full results are in [evaluation/metrics.json](evaluation/metrics.json). The overall Typed set includes workflow types covered during training; unseen workflows are reported separately. Basic probes are a development regression set.

| Dataset / slice | Items | Laya Multilingual | xDecision |
|---|---:|---:|---:|
| Typed decisions | 2,000 | 35.20% | 58.75% |
| Unseen workflow: agent_trace_observability | 500 | 28.60% | 34.40% |
| Internal validation set | 3,599 | 48.85% | 70.41% |
| Emotion | 1,000 | 54.50% | 56.60% |
| SST-5 | 2,210 | 27.29% | 44.84% |
| XCOPA | 1,100 | 57.64% | 59.09% |
| XWinograd | 4,442 | 52.16% | 62.94% |
| Belebele subset | 2,091 | 33.72% | 55.38% |
| XStory | 1,649 | 57.19% | 70.95% |
| Wikidata-derived holdout | 3,377 | 48.47% | 90.94% |
| Wikipedia-derived holdout | 1,500 | 60.73% | 91.47% |
| Prompt-injection slice | 116 | 64.66% | 67.24% |

Performance setup: Apple M5, CPU FP32, four threads, short choice requests, two warm-up calls and 20 timed calls. Timings use the original checkpoint; peak RSS covers model loading and the full probe suite.

| Metric | xDecision |
|---|---:|
| Basic targeted probes | 28/29 |
| Relative-suitability probes | 7/16 |
| Median / P95 latency | 24.52 / 24.91 ms |
| Model loading time | 1.81 s |
| Peak process RSS | 3.47 GB |

Dequantized Q8_0 was re-evaluated on 15 suites, with a maximum accuracy drop of **0.45 percentage points**. See [quantization validation](evaluation/validation.json).

### Known limitations

- **Relative suitability and order stability:** both GitHub and Hugging Face can host code, but ranking which is more suitable still produces errors. Some evidence-based scoring answers flip when fact order changes, and the same proposition can yield different results across question types.
- **Unseen domains and high-confidence errors:** unseen-workflow accuracy is 34.40%. Typed ECE is 0.1236 and act AUROC is 0.6583; unseen-workflow act AUROC is 0.4926. High-risk uses require independent validation and human review.
- **GGUF execution:** the MLX backend loads the custom `xdecision` architecture directly; the PyTorch backend restores a temporary checkpoint. Both dequantize Q8_0 for dense inference. llama.cpp and Ollama are not currently supported.

### Files

```text
models/checkpoint/  Original weights, encoder config, tokenizer, inference config
models/gguf/        xDecision-F16.gguf, xDecision-Q8_0.gguf
data/              Training, calibration, test data, and source manifest
configs/           Continued-training parameters and data mixture
src/xdecision/     Inference, training, calibration, evaluation, and export
tests/             Code tests
examples/          Minimal examples
evaluation/        Metrics and quantization validation
```

F16 is **704.10 MB**, includes all 170 tensors and the tokenizer, and restores the original checkpoint losslessly. Q8_0 is **402.55 MB**, with 101 matrices quantized. File hashes are listed in `checksums.sha256`.

Data is distributed according to source licenses: **49,801 of 139,680 rows** from the final training stage and all **36,000 rows** of the six-view equivalence groups are included. Per-stage counts and omissions are listed in [data/manifest.json](data/manifest.json); usage terms are in the [data documentation](data/README.md).

### Continue training

Load existing weights from `models/checkpoint` with a newly initialized optimizer. The act head is retained by default; recalibrate temperatures after training.

```bash
python -m xdecision.data_cache --input data/train/continuation.jsonl.gz --output work/train.pt
python -m xdecision.data_cache --input data/train/equivalence.jsonl.gz --output work/equivalence.pt
python -m xdecision.train --dry-run
python -m xdecision.train @configs/continue.args
```

Custom JSONL records use `state`, `question`, and `target`; see `examples/train.jsonl`. Targets follow candidate order and sum to one; noul uses false/true order. Split training, calibration, and test data by domain, entity, or scenario first.

```bash
python -m xdecision.prepare --input your-train.jsonl --checkpoint models/checkpoint --output work/custom.pt
python -m xdecision.train @configs/continue.args --train work/custom.pt
python -m xdecision.prepare --input your-calibration.jsonl --checkpoint work/continued/final --output work/eval/calib.pt
python -m xdecision.prepare --input your-test.jsonl --checkpoint work/continued/final --output work/eval/val.pt
python -m xdecision.evaluate --ckpt work/continued/final --data work/eval --calibrate --suites val
python -m xdecision.gguf_io export --checkpoint work/continued/final --output work/xDecision-F16.gguf --quantization F16
python -m xdecision.gguf_io verify --checkpoint work/continued/final --gguf work/xDecision-F16.gguf
```

### Device selection

Automatic priority is **CUDA → MPS → CPU**. With multiple CUDA GPUs, the device with the most free memory at startup is selected. Use `CUDA_VISIBLE_DEVICES` to limit eligible devices or `--device cuda:0` to select one explicitly. Evaluation and calibration use FP32.

| Training device | Automatic precision |
|---|---|
| CUDA GPU with native BF16 support | BF16 |
| Other CUDA GPUs | FP16 + GradScaler |
| Apple MPS on macOS 14+ | BF16 |
| Older MPS / CPU | FP32 |

Override precision with `--precision fp32`. To reduce memory usage, lower `--max-tokens` / `--max-items` and enable `--grad-ckpt`; use `--accum` to increase the effective batch size. Logs report device, precision, and memory usage. Training steps were verified on CPU/MPS. CUDA selection branches passed tests; validation on physical NVIDIA hardware remains pending.

Apple Silicon also has an MLX entry point: install with `pip install -e '.[apple]'`, then run `python -m xdecision.mlx_train @configs/continue-mlx.args`. Both continuation configurations load `configs/objective.args`: soft CE, RLCD, soft-label act supervision, cross-format consistency, uncertainty, high-confidence-error, and selective-risk objectives. Every third micro-batch contains intact equivalence groups. The shared weights are 0.5 for consistency, uncertainty, and high-confidence-error losses, and 0.2 for selective risk; act loss has weight 0.2 when equivalence data is supplied. Missing or malformed groups raise an error instead of silently disabling consistency. These are explicit continuation settings, not a reconstruction of the historical optimizer state; backend precision and optimizer schedules can differ.

Custom equivalence data must preserve `equiv_group`, `support_index`, and `support_flip`, include all three question types in each six/eight-view group, and represent the same reference proposition. The trainer checks aligned target support and keeps groups in one micro-batch. `--max-items` and `--max-tokens` must fit a complete group. Continue to use independent calibration and evaluation after training.

### License

Code and model are released under Apache-2.0, with the upstream mmBERT-base MIT license and attribution preserved. See `LICENSE`, `NOTICE`, and `THIRD_PARTY_LICENSES`. Datasets retain their individual source licenses, including non-commercial restrictions on some sources.

---

## 中文

[Switch to English](#english)

约 322M 参数的多语结构化决策模型，基于 [Laya Multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) 后训练，使用 [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) 编码器。

- `choice`：在候选项中选择。
- `noul`：判断命题真假。
- `score`：在有序刻度上评分。
- 输出选项概率及 act/escalate 执行建议。

提供原始 checkpoint、F16/Q8_0 GGUF、可分发训练数据和续训代码，支持 CUDA、Apple MPS、CPU；Apple Silicon 另有原生 MLX 推理后端。

### 适用场景

面向有明确上下文和候选项的分类、工作流路由、命题判断及有序评分。输入为事实上下文和结构化问题，输出为候选概率与执行建议。部署前应在目标领域校准并验证，证据不足或高风险决策保留人工复核。

## 快速开始

```bash
git clone https://github.com/xnetsc/xDecision.git
cd xDecision
git lfs pull
python -m pip install -e .
```

NVIDIA 环境先按 [PyTorch 安装页](https://pytorch.org/get-started/locally/) 安装与驱动匹配的 CUDA 版本。

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

输入上限 1,024 tokens，其中问题与选项预算 256 tokens。`probabilities` 为选项概率，`action.act_probability` 为执行头输出；`answer_confidence` 与兼容字段 `confidence` 沿用上游定义，使用时应分别解释。

## Apple Silicon 原生推理

安装 `pip install -e '.[apple]'` 后选择 MLX 后端：

```python
with xdecision.load("models/gguf/xDecision-F16.gguf", backend="mlx") as model:
    result = model.predict(state, questions)
```

MLX 默认使用 Apple GPU FP16，权重常驻内存；直接读取 checkpoint 目录或完整 GGUF，不落盘还原模型。Q8_0 在加载时一次解量化为稠密 FP16 数组。`batch_size=8` 可降低问题批次的内存占用。默认 `backend="torch"` 保留 CPU/CUDA/MPS 路径。

严格 FP32 需在启动 Python、导入 MLX 之前设置 `MLX_ENABLE_TF32=0`，再传入 `dtype="float32"`。[MLX 默认可能降低 FP32 矩阵运算的内部精度](https://ml-explore.github.io/mlx/build/html/usage/precision.html)，仅修改数组类型不够。

两后端采用相同的提示构造和返回字段。MLX 原样应用保存的校准温度，保留 act 头；不缓存答案、不跨问题复用上下文编码结果，同一请求的状态仅分词一次。模型加载、首次请求初始化与预热后推理延迟分别计量。

Apple M5、24 GiB、相同 checkpoint，每组预热 5 次、计时 30 次；串行执行、同步计时、端到端推理，`MLX_ENABLE_TF32=0`：

| 后端 | 短 choice 中位 / P95 | 短 choice+noul+score 中位 | 三道 1,024-token 问题中位 |
|---|---:|---:|---:|
| PyTorch CPU，4 线程 | 25.01 / 26.35 ms | 59.27 ms | 1,033.57 ms |
| PyTorch MPS | 14.11 / 15.09 ms | 28.99 ms | 424.48 ms |
| MLX 严格 FP32 | 8.65 / 9.25 ms | 27.44 ms | 408.21 ms |
| MLX FP16 | **5.21 / 6.07 ms** | **11.03 ms** | **125.09 ms** |

42 道兼容性问题中，各后端最高概率选项均与 CPU 基准一致。严格 FP32 公开输出一致到四位小数；FP16 选项概率最大差 0.003。F16、Q8_0 GGUF 加载路径也通过检查，Q8_0 以同文件还原的 PyTorch 结果为基准。这是后端保真验证，不代表模型准确度提升。上述负载下 FP16 的 MLX 峰值分配为 1.42 GB，不包含进程的其它内存。本次会话 checkpoint 加载 0.43 秒、F16 GGUF 加载 14.32 秒，不计入预热后延迟，也不作为全新进程冷启动结果。[完整测量](evaluation/native_mlx.json)。

在仓库根目录复现：

```bash
MLX_ENABLE_TF32=0 python -m xdecision.benchmark_runtime \
  --checkpoint models/checkpoint --f16 models/gguf/xDecision-F16.gguf \
  --q8 models/gguf/xDecision-Q8_0.gguf --output /tmp/xdecision-runtime.json
```

## 训练优化

- 多语推断、阅读理解、常识与知识混合训练，加入实体—属性绑定、反事实和否定样本。
- 覆盖事实顺序、选项顺序、同义改写及三种题型。
- 引入证据不足样本、分组一致性和不确定性目标，训练 act/escalate 头。
- 冻结多语 token embeddings，保留通用任务回放；按题型和选项数拟合温度。

### 跨题型一致性

发布权重训练时已使用 `choice`、`noul`、`score` 的统一命题支持度约束。对于表达**同一命题**的等价视图，将 choice 正选项概率、noul 的 true 概率和 score 的归一化期望映射到共同标尺；先对齐候选置换与反向刻度，再最小化组内支持度差的均方值。这是训练损失，不是在推理后改写概率。

在六个事实场景、24 个变体组的小型诊断集上，经过组合后训练配方，平均跨题型支持度跨度由 **0.273 降至 0.135**，成品最大跨度仍为 **0.430**。这不是一致性损失的独立消融结果，差异也尚未消除。score 期望只是语义支持度代理，不是校准概率；可行性、主要用途和相对适配度是不同命题，不能强制对齐。

## 评测

统一本地评测器的准确率如下，完整指标见 [evaluation/metrics.json](evaluation/metrics.json)。Typed 整体包含训练涉及的工作流类型，未见工作流单列。基础探针属于开发回归集。

| 数据集 / 切片 | 题数 | Laya Multilingual | xDecision |
|---|---:|---:|---:|
| Typed decisions | 2,000 | 35.20% | 58.75% |
| 未见工作流 agent_trace_observability | 500 | 28.60% | 34.40% |
| 内部验证集 | 3,599 | 48.85% | 70.41% |
| Emotion | 1,000 | 54.50% | 56.60% |
| SST-5 | 2,210 | 27.29% | 44.84% |
| XCOPA | 1,100 | 57.64% | 59.09% |
| XWinograd | 4,442 | 52.16% | 62.94% |
| Belebele 子集 | 2,091 | 33.72% | 55.38% |
| XStory | 1,649 | 57.19% | 70.95% |
| Wikidata 派生留出 | 3,377 | 48.47% | 90.94% |
| Wikipedia 派生留出 | 1,500 | 60.73% | 91.47% |
| 提示注入切片 | 116 | 64.66% | 67.24% |

性能测试：Apple M5、CPU FP32、4 线程、短 choice 请求，预热 2 次、计时 20 次。耗时使用原 checkpoint；峰值 RSS 覆盖模型加载及整套测试。

| 指标 | xDecision |
|---|---:|
| 基础定向探针 | 28/29 |
| 相对程度探针 | 7/16 |
| 延迟中位数 / P95 | 24.52 / 24.91 ms |
| 加载耗时 | 1.81 s |
| 峰值进程 RSS | 3.47 GB |

Q8_0 解量化后完成 15 套复测，准确率最大下降 **0.45 个百分点**，详见 [量化验证](evaluation/validation.json)。

### 已知限制

- **相对适配度与顺序稳定性**：GitHub/HF 都可存代码，但“哪个更适合”的排序仍有错误；部分有事实评分题会随事实顺序翻转，同命题跨题型结果也存在差异。
- **未见领域与高置信错误**：未见工作流准确率 34.40%；Typed ECE 0.1236、act AUROC 0.6583，未见工作流 act AUROC 0.4926。高风险用途需独立验收和人工复核。
- **GGUF 运行**：MLX 直接加载自定义 `xdecision` 架构，PyTorch 路径先还原临时 checkpoint；两者均将 Q8_0 解量化后执行稠密推理。llama.cpp / Ollama 暂不支持。

## 文件

```text
models/checkpoint/  原权重、编码器配置、tokenizer、推理配置
models/gguf/        xDecision-F16.gguf、xDecision-Q8_0.gguf
data/              train、calibration、test、来源清单
configs/           续训参数、数据配比
src/xdecision/     运行、训练、校准、评测、导出
tests/             代码测试
examples/          最小示例
evaluation/        指标与量化验证
```

F16 **704.10 MB**，包含全部 170 个张量及 tokenizer，可无损还原原 checkpoint；Q8_0 **402.55 MB**，量化其中 101 个矩阵。文件校验值见 `checksums.sha256`。

数据按来源许可分发：最后训练阶段提供 **49,801 / 139,680 行**，六视图等价组提供全部 **36,000 行**。各阶段数量与缺项见 [data/manifest.json](data/manifest.json)，使用条件见 [数据说明](data/README.md)。

## 继续训练

从 `models/checkpoint` 加载现有权重开始新一轮训练，优化器重新初始化。默认保留 act 头，续训后重新校准温度。

```bash
python -m xdecision.data_cache --input data/train/continuation.jsonl.gz --output work/train.pt
python -m xdecision.data_cache --input data/train/equivalence.jsonl.gz --output work/equivalence.pt
python -m xdecision.train --dry-run
python -m xdecision.train @configs/continue.args
```

自有 JSONL 使用 `state`、`question`、`target`，格式见 `examples/train.jsonl`。target 按候选项顺序排列，总和为 1；noul 顺序为 false/true。先按领域、实体或场景拆分训练、校准和测试集。

```bash
python -m xdecision.prepare --input your-train.jsonl --checkpoint models/checkpoint --output work/custom.pt
python -m xdecision.train @configs/continue.args --train work/custom.pt
python -m xdecision.prepare --input your-calibration.jsonl --checkpoint work/continued/final --output work/eval/calib.pt
python -m xdecision.prepare --input your-test.jsonl --checkpoint work/continued/final --output work/eval/val.pt
python -m xdecision.evaluate --ckpt work/continued/final --data work/eval --calibrate --suites val
python -m xdecision.gguf_io export --checkpoint work/continued/final --output work/xDecision-F16.gguf --quantization F16
python -m xdecision.gguf_io verify --checkpoint work/continued/final --gguf work/xDecision-F16.gguf
```

### 设备选择

默认优先 **CUDA → MPS → CPU**。多张 CUDA 卡按启动时空闲显存选一张，可通过 `CUDA_VISIBLE_DEVICES` 限定范围或 `--device cuda:0` 指定。评测和校准使用 FP32。

| 训练设备 | 自动精度 |
|---|---|
| 支持原生 BF16 的 CUDA GPU | BF16 |
| 其他 CUDA GPU | FP16 + GradScaler |
| Apple MPS、macOS 14+ | BF16 |
| 较旧 MPS / CPU | FP32 |

`--precision fp32` 可覆盖精度；显存不足时减小 `--max-tokens` / `--max-items`，启用 `--grad-ckpt`，用 `--accum` 增大有效批量。日志输出设备、精度与内存占用。CPU/MPS 已通过训练步验证；CUDA 选择分支测试通过，NVIDIA 实卡验证待完成。

Apple Silicon 另提供 MLX 入口：安装 `pip install -e '.[apple]'` 后运行 `python -m xdecision.mlx_train @configs/continue-mlx.args`。两份续训配置共同加载 `configs/objective.args`，使用 soft CE、RLCD、软标签 act 监督、跨题型一致性、不确定性、高置信错误惩罚和选择性风险目标。每三个微批次插入一个完整等价组批次；一致性、不确定性、高置信错误惩罚的权重均为 0.5，选择性风险为 0.2，提供等价组时 act 损失权重为 0.2。缺失或不合格的等价组直接报错，不会静默关闭一致性训练。这些是明确的续训配置，并非恢复历史优化器状态；不同后端的精度和优化器调度仍可能不同。

自有等价组数据须保留 `equiv_group`、`support_index`、`support_flip`；每组六或八个视图覆盖三种题型，且表达同一个参考命题。训练器校验映射后的目标支持度，并保证整组进入同一微批次；`--max-items`、`--max-tokens` 必须容纳完整一组。续训后仍需独立校准与评测。

## 许可

代码和模型采用 Apache-2.0，上游 mmBERT-base 保留 MIT 许可及归属。见 `LICENSE`、`NOTICE`、`THIRD_PARTY_LICENSES`。数据分别适用来源许可，其中部分限非商业用途。

[English](#english) | [中文](#中文)
