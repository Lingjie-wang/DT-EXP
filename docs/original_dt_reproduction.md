# 原始 DT：HalfCheetah-medium-replay 原始 / 延迟奖励复现

本轮追查论文中普通 DT 基线约 33 分与本项目 CORL DT 约 40 分的差异。
LPT 的该基线明确引用 Yamagata et al. (2023) 的 QDT 论文；其附录 C.3
规定 DT 为 batch 64、100k updates、每模型 100 个评测 episode、5 个训练 seed。
本轮按用户已有预算偏好先跑 **seed 0**，不宣称复现了五 seed 均值。

- 论文：https://proceedings.mlr.press/v202/yamagata23a/yamagata23a.pdf
- 原始 DT：https://github.com/kzl/decision-transformer
- 固定 commit：`e2d82e68f330c00f763507b3b01d774740bee53f`
- 入口：`algorithms/offline/original_dt_repro.py`。

## 固定设置

| 项目 | 设置 |
|---|---|
| 数据 | HalfCheetah-medium-replay-v2，全部 202 条轨迹 / 202000 transitions |
| 数据审计 | QT 缓存 pickle 与 CORL HDF5 的状态、动作、奖励、终止标记及轨迹边界逐项相等 |
| 模型 | 原始 DT GPT-2 实现，3 层、1 head、hidden 128、ReLU、dropout 0.1 |
| 训练 | 从头训练，batch 64、context 20、100000 次完整更新 |
| 优化器 | 原始 AdamW，lr 1e-4、weight decay 1e-4、warmup 10000、clip 0.25 |
| 采样与损失 | 原始长度加权轨迹采样、均匀起点、左侧 padding、只对有效 token 求动作 MSE |
| dense | 原始逐步奖励；推理每步从 RTG 中减去实际奖励 |
| delayed | 整条轨迹总回报移至最后一步；推理 RTG 保持常量 |
| 评测 | 原始代码的 `HalfCheetah-v3`，每 10000 updates，RTG 12000 和 6000 各 100 局 |
| 评测种子 | 每局 seed `42+i`，两组及所有 checkpoint 共享；保存/恢复训练 RNG |
| 主结果 | 100k 最终 checkpoint；两个 RTG 分开报告，不用最高 checkpoint 代替最终结果 |

封装直接调用未修改的上游 `experiment()`、`get_batch()`、`SequenceTrainer.train_step()`
和 `evaluate_episode_rtg()`，只加入审计、保存、种子控制与日志。前 16 个 batch
保存不包含奖励的采样哈希，两组保存初始模型哈希。checkpoint 包含模型、优化器、
scheduler、RNG 和状态归一化；完成更新数从 1 计数。

当前兼容运行时使用 `third_party/qt-env`：Python 3.9、Torch 1.11+cu113、
Transformers 4.11.3、Gym 0.23、MuJoCo 2.1。它与上游 2021 年的依赖锁定不同，
各任务保存实际版本和 GPU。历史实验的精确 seed、软件环境与每次 RTG 选择未公开，
本轮属于原始代码及论文公开配置复现。GPU 不同也可能导致数值路径不同。

## 启动与记录

```bash
git clone https://github.com/kzl/decision-transformer third_party/decision-transformer
git -C third_party/decision-transformer checkout e2d82e68f330c00f763507b3b01d774740bee53f
third_party/qt-env/bin/python algorithms/offline/original_dt_repro.py prepare \
  --root results/original-dt-hcmr-seed0-20261004-v2 \
  --source third_party/decision-transformer \
  --dataset third_party/QT/D4RL/halfcheetah-medium-replay-v2.pkl \
  --hdf5 /labmount/users/202615385/.d4rl/datasets/halfcheetah_medium_replay-v2.hdf5
```

准备阶段冻结源代码、协议和数据哈希，之后使用 campaign 内的脚本快照：

```bash
sbatch results/original-dt-hcmr-seed0-20261004-v2/source/run.sbatch \
  "$PWD/results/original-dt-hcmr-seed0-20261004-v2" dense
sbatch results/original-dt-hcmr-seed0-20261004-v2/source/run.sbatch \
  "$PWD/results/original-dt-hcmr-seed0-20261004-v2" delayed
```

每组申请任意 NVIDIA 单卡、2 CPU、8GB、24 小时。保留已有任务顺序。
登录节点执行 `source/sync_results.py --root ROOT --arm ARM --job JOBID`，
向 `2820402607-shandong-university/CORL-DDR` 同步指标和逐 episode 数值表，
不上传代码、数据或 checkpoint。W&B 名称为
`OriginalDT-HCMR-{dense,delayed}-seed0-100k-b64`。同步器读回核验起始配置、
全部评测点和最终分数，结束后退出，不提交任何新任务。

## 验收

```bash
PYTHONPATH=algorithms/offline third_party/qt-env/bin/python -m unittest discover \
  -s tests -p 'test_original_dt_repro.py' -v
```

测试直接调用上游代码，检查延迟评测 RTG 恒定、dense RTG 正确递减、
padding 动作不影响有效 token 的梯度、评测不消耗训练 RNG。
Slurm 任务先在分配到的 GPU 节点使用完整 batch 64 做两次真实更新，
并在每个目标上跑完整 1000 步评测；预检成功才从头启动正式 100k 训练。
源码快照与数据哈希在预检和正式训练启动前分别校验。

登录节点 CPU smoke 因缺少 `GL/osmesa.h` 在模拟器导入阶段失败，未做模型更新。
该尝试保留于 `results/original-dt-hcmr-seed0-20261004`；正式任务使用独立的
`-v2` campaign，沿用已有 GPU 节点可用的 MuJoCo 运行时，不修改历史环境配置。
