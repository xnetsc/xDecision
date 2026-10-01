# 配置

`continue.args` 是保守的续训起点，不是历史训练命令的逐项复原。先将发布的数据子集转换为 `work/train.pt`，再运行 `python -m xdecision.train @configs/continue.args`。按自己的内存与数据量调整；重新训练后必须独立校准和评测。

`source_mix.json` 保存最初数据源采样配比。它不是成品最后阶段的精确批次顺序，也不是所有来源都已获准再分发；实际随包数据及各阶段数量见 `data/manifest.json`。
