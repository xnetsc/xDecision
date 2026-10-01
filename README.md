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

约 322M 参数的多语结构化决策模型，基于 [Laya Multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) 后训练，使用 [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) 编码器。

- `choice`：在候选项中选择。
- `noul`：判断命题真假。
- `score`：在有序刻度上评分。
- 输出选项概率及 act/escalate 执行建议。

提供原始 checkpoint、F16/Q8_0 GGUF、可分发训练数据和续训代码，支持 CUDA、Apple MPS、CPU。

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

## 训练优化

- 多语推断、阅读理解、常识与知识混合训练，加入实体—属性绑定、反事实和否定样本。
- 覆盖事实顺序、选项顺序、同义改写及三种题型。
- 引入证据不足样本、分组一致性和不确定性目标，训练 act/escalate 头。
- 冻结多语 token embeddings，保留通用任务回放；按题型和选项数拟合温度。

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
- **GGUF 运行**：自定义 `xdecision` 架构由本仓库加载器还原后交给 PyTorch；Q8_0 节省存储，运行时解量化。llama.cpp / Ollama 暂不支持。

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

Apple Silicon 另提供 MLX 入口：安装 `pip install -e '.[apple]'` 后运行 `python -m xdecision.mlx_train`。PyTorch 使用 soft CE、RLCD、act 损失；MLX 额外支持 `--equiv`、`--w-consistency`、`--w-uncertain`、`--w-overconf`、`--w-selective`，通过显式入口选择训练配方。

## 许可

代码和模型采用 Apache-2.0，上游 mmBERT-base 保留 MIT 许可及归属。见 `LICENSE`、`NOTICE`、`THIRD_PARTY_LICENSES`。数据分别适用来源许可，其中部分限非商业用途。
