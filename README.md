# 基于历史窗口编码 SAC 的刚性接触网受电弓主动控制

用 [PyTorch](https://pytorch.org) + [TorchRL](https://github.com/pytorch/rl) 在刚性接触网–受电弓仿真（PROCS，POMDP）上训练
LSTM-SAC（历史窗口 + 无状态 LSTM），控制下框架主动力，降低接触力波动。

用 [pixi](https://pixi.sh) 管理。

## 环境依赖

```bash
pixi install
```

Python 3.12，主要依赖 `torch>=2.13`、`torchrl>=0.13`、`gymnasium`、`pandas`、`matplotlib`。
