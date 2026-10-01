# 数据

成品实际使用的训练数据，按已核实的来源许可分发。各阶段数量、来源、许可与缺项见 `manifest.json`。

`train/` 用于训练，`calibration/` 用于温度拟合，`test/` 用于内部验证。不同阶段包含回放和重复，使用时按阶段选择。

格式为 gzip 压缩的 JSONL，保存实际训练的 token IDs、选项位置、题型和监督分布。`original_row` 为原混合数据行号，`src` 对应来源；tokenizer 位于 `models/checkpoint/tokenizer/`，可用于解码文本。

## 使用

```bash
python -m xdecision.data_cache --input data/train/continuation.jsonl.gz --output work/train.pt
python -m xdecision.train @configs/continue.args
```

发布子集的混合比例与完整训练集不同。继续训练后，请用独立领域数据重新评测。

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

许可、归属或版权通知待核实的来源暂未收录，原因逐项列于 manifest。使用与再分发须遵循各来源的非商业、署名及相同方式共享等条件。
