# QT 官方代码原样训练：HalfCheetah-medium-replay-v2 / delayed / seed 0

日期：2026-09-19。

这是 **官方实现核查与复跑**，不是已经成功复现论文 Table 4 的声明。
已向用户说明末端奖励处理问题；用户明确选择：
“先保持官方代码原样跑，接受该实现问题并单独标注”。

## 源码与隔离

- 作者仓库：<https://github.com/charleshsc/QT>。
- 固定提交：`cb9e1a4873449b3467f6bf5586e010180d90614c`。
- 服务器源码：`/labmount/users/202615385/code/DT-EXP/third_party/QT`。
- 独立 venv：`third_party/qt-env`，使用 `--system-site-packages` 复用
  `adt-delayed` 的 PyTorch 1.11.0+cu113、MuJoCo、Gym、D4RL。
- 没有重新安装 PyTorch，没有修改原 `adt-delayed` 环境。
- 保留全部官方训练器、模型、采样器源文件；启动前校验提交和 tracked diff。
- 入口通过 AST 仅略过两个 HalfCheetah 路径未使用的 import：
  `mjrl.utils.gym_env.GymEnv`、`torch.utils.tensorboard.SummaryWriter`。
  官方代码令 `writer=None`，本实验也不使用 TensorBoard。
- 其他兼容依赖：`transformers==4.11.3`、`tokenizers==0.10.3`、
  `huggingface-hub==0.0.19`、`sacremoses==0.1.1`、
  `python-dateutil==2.9.0.post0`；实际运行版本和源码哈希写入 config。
- 原 CORL DT、v2、v3-high、v4 等入口没有修改。

`third_party/` 本来就在 Git ignore 中。Git 保存独立 adapter、单元测试、启动脚本
和本文；第三方源码通过仓库 URL + 固定提交恢复，checkpoint 和数据不上传 Git。

## 已确认但按用户选择不修复的问题

官方 `ql_trainer.py` 的 `k_rewards + use_discount` 路径有：

```python
rewards[:, -1] = 0.
```

critic loss 又只计算窗口 `[:, :-1]`。这对一般非终止窗口相当于把末尾作为
bootstrap state；但在 terminal-sum delayed 数据中，唯一非零奖励一旦进入窗口，
必然位于其最后一个有效位置，也就是左 padding 后的最后一列，因此被清零。
该实现没有另外训练“末端 Q = 末端奖励”的目标。当前 HCMR pickle 的 terminals
全部为 0（时间上限是切段边界，不在该字段中），这个问题不能靠 done mask 解决。

真实 upstream trainer 的单步机制测试确认：相同权重、RTG、状态、动作和随机数下，
最后一列奖励从 0 改为 1000，全部已记录 loss、actor/critic 更新后权重逐位相同；
将奖励放在倒数第二列则改变 critic loss 和权重。

注意：不是说整个 QT 不接收任何回报信息。RTG 仍包含轨迹总回报，影响 actor 及其
bootstrap 动作。结论是 **这个 Q target 的显式 reward 项丢掉了末端奖励**。
不据此否定论文算法，也不将此次结果当作论文正确实现的最终结论。

运行时累计记录：

- `audit/nonzero_rewards_before_cumulative`
- `audit/nonzero_rewards_after_cumulative`

## 配置：保留已发布的 HCMR 脚本参数

仓库 `run.sh` 没有单独发布 HCMR delayed 配置。这里明确使用其 HCMR 参数，
**不擅自假定这就是论文 sparse 表所用超参数**，也不按照本次评测调参。

| 项目 | 本次设置 |
| --- | --- |
| 数据 | 同一缓存 `halfcheetah-medium-replay-v2.pkl`，202 条完整轨迹 / 202000 transitions |
| reward | 官方 `mode=delayed`：每条轨迹奖励移到末尾，RTG gamma=1 |
| 初始化 | seed 0，随机初始化，不读取旧 50k checkpoint |
| 更新数 | 100000 actor + 100000 critic updates |
| 网络 | hidden 256，4 layers，4 heads，dropout 0.1 |
| context / batch | K=5 / 256 |
| actor / critic lr | 3e-4 / 3e-4，官方 Adam |
| actor weight decay | 1e-4 |
| Q loss 系数 | eta=5.0，eta2=1.0，沿用发布的 HCMR 脚本 |
| discount / tau | 0.99 / 0.005 |
| 梯度裁剪 | 15.0 |
| Bellman 路径 | k_rewards=True，use_discount=True；原样保留 |
| 学习率调度 | 每 1000 updates 一次 cosine；T_max=500 iterations，保留官方调度长度 |
| 100k 停止 | max_iters=500，early_stop=True，early_epoch=99，消除原循环的预算歧义 |
| EMA | 原样保留：1000 步后开始，每 5 步，系数 0.995 |
| 评测 | 每 5000 updates；eval seed 42；每主口径 100 episodes |

已逐项核对 pickle 与 CORL HDF5：全部 observations、actions、rewards、terminals
数值完全相同，按 terminals/timeouts 划分的全部 202 条轨迹边界也完全相同。

官方 `bc_loss` 实际是动作 MSE **加 next-state prediction MSE**，不是纯 CORL DT
动作 MSE；本实验保留这一实现。官方 `warmup_steps` 参数虽被解析，但没有实际参与
optimizer 调度；本实验没有补上 warmup。

因此，相同环境、奖励、训练步数、评测初始状态不代表所有设置都与 CORL DT 一致：
原 CORL 配置为 hidden128 / 3 layers / 1 head / context20 / batch4096。
该次结果是跨方法原配方对比，不是“只增加一个 Q loss”的公平单因素消融。

## 评测口径：避免混淆 QT 选动作与固定 RTG

| W&B 前缀 | 实际策略与奖励信息 | 用途 |
| --- | --- | --- |
| `eval/qt_strict_delayed` | 官方 `get_action`：候选 RTG `[12000,9000,6000]`、50 个候选、Q-softmax 抽样；环境仅在 episode 结束给出总奖励 | 主 QT 分数 |
| `eval/actor_12000` | 同一个已训练 QT actor；RTG=12000 固定；单次 forward，不用 Q 选动作；严格 delayed feedback | 更接近 DT 的固定条件推理口径，但不是完整 QT |
| `eval/actor_6000` | 同上，RTG=6000 | 同上 |
| `eval/qt_upstream_feedback` | 官方 rollout 原样，底层环境仍返回 dense reward；外部 RTG 虽固定，内部 Q-derived candidate 仍可读取上一步 reward | 仅 100k 额外评测 10 episodes，不能作为严格 delayed 主结果 |

主口径通过环境 wrapper 把中间 reward 置 0、最终返回总和。官方训练代码和
`get_action` 不变，也没有对其 Q-to-RTG 换算或候选噪声另外做修正。
附加 actor-only forward 与 upstream `infer_no_q=True` 返回的第一个候选经测试一致
（数值容差 1e-6）；避免无意义地计算剩余 49 个候选。

四类指标都使用 D4RL normalized score **乘 100**。不同口径不能混合取 best。
评测前保存、评测后恢复训练 RNG；同一口径每次重设环境 seed=42，与已有 DT 一致。
相比 upstream 没有 RNG 隔离的执行过程，整条训练随机数轨迹不保证逐步相同；但每次
train_step 的训练数学、采样方法与参数更新逻辑没有改变。

## 记录、快照与运行

W&B：`2820402607-shandong-university/CORL-DDR`。
名称：`QT-Official-Uncorrected-HCMR-delayed-seed0-100k`。
group：`QT-Official-Uncorrected-HCMR-delayed`。

每 100 步记录 actor/critic/BC/QL loss、target Q、梯度范数、奖励清零诊断。
本地保存初始、10k、20k、50k、75k、100k full snapshot，包含 actor、critic、
target critic、EMA、两套 optimizer / scheduler、RNG、状态归一化和配置。
另外保留原入口按主 QT 分数保存的 best actor/target critic checkpoint。
不上传模型、原始状态、动作或数据；W&B 仅配置、哈希和标量。

```bash
# 在项目根目录，先准备固定版本第三方源码和上述依赖。
PYTHONPATH=algorithms/offline third_party/qt-env/bin/python -m unittest discover \
  -s tests -p test_qt_official_runner.py -v
sbatch --nodelist=gn7 --time=00:20:00 \
  scripts/dt_experiments/run_qt_official_hcmr_seed0.sbatch smoke
# 短跑通过后再正式启动：
sbatch --nodelist=gn7 scripts/dt_experiments/run_qt_official_hcmr_seed0.sbatch
```

Smoke 使用完整模型与 batch，20 updates，四种口径各 1 episode，offline W&B。
不创建周期监控。

## 启动记录

5 个机制测试已通过，包括 RNG 隔离、末端奖励 wrapper、真实 upstream reward
排除行为，以及固定 RTG 单次 forward 与 upstream 第一个候选的数值一致性。

首次 smoke `9382` 在 import 阶段发现缺失 `python-dateutil`，未开始训练；补齐独立
环境依赖后，smoke `9383` 在 gn7 / RTX 4090 上完成，耗时 30 秒、exit 0。
20 次更新及四种评测各 1 episode 均完成；checkpoint_0 / checkpoint_20 可正常加载，
actor 和 critic 权重全部有限。该次采样含 27 个非零 delayed 奖励，经 upstream
原样训练路径清零后剩 0 个，与静态分析及机制测试一致。

正式 job id 与 W&B 链接在启动后补充。
