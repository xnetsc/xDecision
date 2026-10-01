# 训练配置

`continue.args` 提供续训起点，默认自动选择 CUDA → MPS → CPU 及对应精度。

先用 `xdecision.data_cache` 将 `data/train/continuation.jsonl.gz` 和
`data/train/equivalence.jsonl.gz` 分别转换为 `work/train.pt` 和 `work/equivalence.pt`。
`continue.args` 与 `continue-mlx.args` 共同加载 `objective.args`：每三个微批次使用一个完整等价组批次，
一致性、不确定性、高置信错误惩罚权重均为0.5，选择性风险为0.2。
这是续训起点配置；两后端目标相同，不承诺优化器轨迹或结果逐位一致。

```bash
python -m xdecision.train @configs/continue.args --dry-run
python -m xdecision.train @configs/continue.args
python -m xdecision.train @configs/continue.args --device cuda:0 --precision fp32
python -m xdecision.mlx_train @configs/continue-mlx.args
```

命令行后置参数覆盖文件配置。按显存调整 `--max-tokens`、`--max-items` 和 `--accum`；续训完成后使用独立校准集拟合温度。

`source_mix.json` 记录初始数据源采样配比，实际发布数据的数量与来源见 `data/manifest.json`。
