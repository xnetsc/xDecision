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

## 极性与相对适配度续训

`contrast.args` 针对“正着问/反着问结论不一致”和相对适配度（如 GitHub 与 HF 哪个更适合做代码仓库）。
训练数据由 `python -m xdecision.contrast_data` 生成并已随仓库提供（`data/train/contrast_*.jsonl.gz`，可读版在 `data/source/`）。
常规批次为续训回放与可行性/主要用途单题，每两个微批次插入一次完整等价组（新的中英文对照组与原等价组）。

```bash
python -m xdecision.data_cache --input data/train/continuation.jsonl.gz:6000 \
    --input data/train/contrast_items.jsonl.gz:3000 --output work/contrast_train.pt
python -m xdecision.data_cache --input data/train/equivalence.jsonl.gz:3600 \
    --input data/train/contrast_groups.jsonl.gz:12000 --output work/contrast_equiv.pt
python -m xdecision.train @configs/contrast.args --dry-run   # 查看检测到的硬件与精度
python -m xdecision.train @configs/contrast.args
python -m xdecision.probes --checkpoint work/contrast/final --output work/contrast/probes.json
```

`--backend auto` 自动选择 CUDA/ROCm/XPU、MLX GPU、MPS 或 CPU；在其它环境接续训练时，把 `--base` 指向上一轮的输出目录即可，优化器会重新初始化。
