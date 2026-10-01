# 数据

这是成品实际使用的数据的**可再分发部分**，不是全部原始混合数据，也不是新生成的替代样本。精确数量、来源、许可与未分发来源见 `manifest.json`。

目录按用途划分：`train/` 是训练数据，`calibration/` 用于温度拟合，`test/` 是内部留出。不同训练阶段存在回放与重复，不能把所有文件简单拼接后称为独立样本。历史阶段只用于溯源与按需续训，不包含失败实验样本或训练日志。

格式为 gzip 压缩的 JSONL，保存原 token IDs、选项标记位置、题型和监督分布，避免解码再分词改变实际训练输入。`original_row` 对应原混合数据的位置，保留重复与相对顺序；`src` 的工作流监督名称作了公开整理。tokenizer 位于 `models/checkpoint/tokenizer/`。这些是可解码的训练文本，不是不可逆匿名数据。

## 使用

```bash
python -m xdecision.data_cache --input data/train/continuation.jsonl.gz --output work/train.pt
python -m xdecision.train --base models/checkpoint --train work/train.pt --out work/continued --device cpu
```

只加载自己的或经过核验的 PyTorch 缓存。此发布子集改变原混合比例，不能保证复现成品指标。内部留出及同来源校准集不能冒充全新领域测试。

## 归属与许可

第三方文本由各数据集作者持有权利。每条样本通过 `src` 对应 manifest 中的原仓库、归属和许可；变更包括题面转换、选项顺序、软标签与分词。保留这些归属、原来源链接和以下许可链接。

- Apache-2.0：见仓库 `LICENSE`。
- CC BY-NC 4.0：[许可](https://creativecommons.org/licenses/by-nc/4.0/)，仅非商业使用。
- CC BY-NC 3.0：[许可](https://creativecommons.org/licenses/by-nc/3.0/)，仅非商业使用。
- CC BY-SA 3.0：[许可](https://creativecommons.org/licenses/by-sa/3.0/)，衍生内容保持相同许可。
- CC BY-SA 4.0：[许可](https://creativecommons.org/licenses/by-sa/4.0/)，衍生内容保持相同许可。
- CC BY 4.0：[许可](https://creativecommons.org/licenses/by/4.0/)。
- CC0 1.0：[声明](https://creativecommons.org/publicdomain/zero/1.0/)。

XNLI 的许可依据[作者仓库](https://github.com/facebookresearch/XNLI/blob/main/LICENSE)；WinoGrande 与 OpenBookQA 的 Apache 许可依据 [WinoGrande](https://github.com/allenai/winogrande/blob/master/LICENSE) 和 [OpenBookQA](https://github.com/allenai/OpenBookQA/blob/master/LICENSE)。其它第三方许可依据 manifest 的发布者数据卡。

许可未核实、混合后无法逐条归属、原版权通知尚未完整保存或市场文本再分发条件不明的来源不分发；这不等于断言它们禁止分发。完整数据本地保留。根目录代码许可不覆盖这些第三方限制，也不提供模型可用于任意商业用途的法律保证。
