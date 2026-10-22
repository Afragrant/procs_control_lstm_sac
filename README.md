# 基于历史窗口编码 SAC 的刚性接触网受电弓主动控制

用 [PyTorch](https://pytorch.org) + [TorchRL](https://github.com/pytorch/rl) 在刚性接触网–受电弓仿真（PROCS，POMDP）上训练
LSTM-SAC（历史窗口 + 无状态 LSTM），控制下框架主动力，降低接触力波动。

用 [pixi](https://pixi.sh) 管理。

## 环境依赖

```bash
pixi install
```

Python 3.12，主要依赖 `torch>=2.13`、`torchrl>=0.13`、`gymnasium`、`pandas`、`matplotlib`。

## Recurrent-Greedy-SAC

独立 LSTM 记历史，proposal 采样后按在线双 Q 留精英，actor 只做这些动作的最大似然。在线控制只跑 actor。日志和模型在 `logs/recurrent_greedy_sac/`、`models/recurrent_greedy_sac/`。

```bash
python runs/run_recurrent_greedy_sac.py train --seed 1
python runs/run_recurrent_greedy_sac.py test --model models/recurrent_greedy_sac/seed_1 --speed 200
```

对比时至少换 `--seed 1`、`2`、`3`。这里的 `batch_size` 是序列条数，不是 Greedy SAC 的转移条数。
