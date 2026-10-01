# 训练配置

`continue.args` 提供续训起点，默认自动选择 CUDA → MPS → CPU 及对应精度。

```bash
python -m xdecision.train @configs/continue.args --dry-run
python -m xdecision.train @configs/continue.args
python -m xdecision.train @configs/continue.args --device cuda:0 --precision fp32
```

命令行后置参数覆盖文件配置。按显存调整 `--max-tokens`、`--max-items` 和 `--accum`；续训完成后使用独立校准集拟合温度。

`source_mix.json` 记录初始数据源采样配比，实际发布数据的数量与来源见 `data/manifest.json`。
