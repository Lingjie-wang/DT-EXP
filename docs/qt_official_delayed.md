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

正式作业：`9384`，GPUNorm / gn7 / RTX 4090，2026-09-19 12:55 左右启动。
W&B：[QT-Official-Uncorrected-HCMR-delayed-seed0-100k](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/ylm6sxr2)。
结果目录：`checkpoints/qt-official/job-9384/`。
实现提交：`a17c68141f71fbe20faa6f93323355518b3a3e58`，已推送且服务器 Git 已同步。
启动前本地使用与 GitHub CI 相同的 Ruff 0.0.278 对完整工作目录检查通过。

这里记录的是启动状态，不是训练已经完成，也不是已经得到有效提升。

## 2026-10-03：末端奖励修正 + eta=0.01 重跑

用户明确要求修复末端奖励处理并把 eta 改为 0.01，重新运行 HCMR sparse。
新实验名称：`QT-TerminalCorrected-eta0.01-HCMR-delayed-seed0-100k`。
这是修正版实验，eta=0.01 是用户指定值，不宣称它就是论文 HCMR sparse 的参数。

只改变训练边界/Q target 处理与 eta；保留 seed 0 随机初始化、原始数据、K=5、
batch=256、网络、学习率、gamma=0.99、EMA、100k 更新数，以及每 5k 更新进行
100 episodes 的三种主评测口径。100k 的 upstream-feedback 诊断仍为 10 episodes。
因此这次与旧 run 的比较同时涉及两项变化，不能单独归因于其中一项。

修复位于 `algorithms/offline/qt_terminal_correction.py`，通过显式
`--correct-terminal-rewards --eta 0.01` 启用：

1. 采样器根据 `si + 实际窗口长度 == 轨迹长度` 标记 episode end，覆盖原始
   terminals 未标记的 time limit。复制 dones 后才修改，原缓存不变；零回报轨迹
   也按边界判断，不通过奖励是否非零推断结束。
2. 终止窗口保留末端奖励，最后一步监督 `Q(s_last, a_last) = r_last`，向前按
   `y_t = r_t + gamma * y_(t+1)` 传播；终止处不 bootstrap。
3. 非终止窗口维持原有语义：最后一个观测作为 bootstrap state，其动作的 reward
   不进入该窗口目标，critic 不监督这个末尾位置。padding 不进入 critic loss。
4. 原版 checkout 不修改，仍校验固定提交和 clean diff。通过带结构检查的 AST
   只替换 get_batch 的边界标记和 train_step 的 Q target / critic loss；actor loss、
   两套 optimizer、EMA、target 更新、推理代码沿用 upstream。

这里把每条离线轨迹（包括 1000 步 time limit）当作 delayed 回报的有限 episode，
在其边界发放奖励并停止 bootstrap；不是无限时域 time-limit bootstrap 设定。
原版未修正模式仍可由旧脚本启动。

除原来的奖励计数外，新 run 记录：

- `audit/episode_end_samples_cumulative`
- `audit/terminal_reward_targets_cumulative`
- `audit/terminal_target_abs_error_max`（必须为 0）

结果目录保存 adapter 源码、修复源码、展开后的 corrected experiment/train_step
以及全部配置和哈希，以便审阅运行时的实际实现。

验证包括 15 个测试：手算折扣目标、无终止 bootstrap 泄漏、零/负末端奖励、
单个有效 token、真实 upstream sampler 的完整/左 padding 边界、非终止更新与
upstream 一致、末端奖励实际改变 critic 更新，以及原有评测/RNG 测试。

启动脚本：`scripts/dt_experiments/run_qt_corrected_eta001_hcmr_seed0.sbatch`。
同一 Slurm allocation 内先执行测试，再用完整模型和 batch 跑 20 updates 的 offline
smoke 和四种评测（各 1 episode）。验证奖励保留计数=末端监督计数>0、目标误差=0、
全部 loss/权重有限且 checkpoint 可加载后，才从头启动 online W&B 的 100k 训练。
smoke 失败则脚本立即退出，不启动正式训练。

本地 15 个机制测试已全部通过，改动文件的 Ruff 0.0.278、Python 编译和 shell 语法
检查通过。全仓库 Ruff 仍有本任务之外的既有问题，未修改其他实验。
登录节点导入完整 MuJoCo 环境会尝试构建 CPU 后端，因缺少 `GL/osmesa.h` 失败；
未安装或替换依赖。GPU 完整 smoke 留在作业内执行，不能把机制测试当作 GPU smoke
已通过。正式提交使用独立源码快照和 SHA-256 清单，后续工作区改动不影响排队任务。

提交记录：2026-10-03 16:19:55（Asia/Shanghai），Slurm job **12045**，GPUNorm，
1 GPU / 6 CPU / 32 GiB / 24 小时。提交后状态为 `PENDING (AssocMaxJobsLimit)`，
表示账号并行作业数已达上限，尚未开始 GPU smoke 或正式训练，尚无新 W&B run URL。
当前已有作业不变；获得配额后自动执行上述 smoke → 校验 → 正式训练流程。

- 提交配置：`checkpoints/qt-terminal-corrected-eta001/submission-20261003-161928/submission.json`
- 源码快照：同目录 `source/`；SHA-256 清单：同目录 `source.sha256`
- 日志：`logs/qt-corrected-12045.out`
- smoke：`checkpoints/qt-terminal-corrected-eta001/job-12045/smoke/`
- 正式结果：`checkpoints/qt-terminal-corrected-eta001/job-12045/train/`
- 正式 W&B 链接由训练入口写入正式结果目录的 `wandb_run.json`。

## 2026-10-04：作业 12045 的网络失败与独立重试入口

Slurm 确认 12045 于 11:29:45 在 gn7 启动，11:32:08 以 `FAILED / 1:0` 退出
（Asia/Shanghai）。15 个机制测试和 20 updates 的 GPU smoke 已通过，四种评测
各 1 episode 完成；27 个非零末端奖励全部保留并进入 critic 监督，末端目标误差
为 0，初始和 20 步 checkpoint 的 actor/critic/target/EMA 权重均有限。

正式阶段尚未开始更新：入口在 `wandb.init()` 中报 `ProxyError` 并于 90 秒超时。
具体原因是作业继承了提交端的 `HTTP_PROXY/HTTPS_PROXY`（及小写变量），指向
`127.0.0.1:17897`；计算节点不存在这个代理，日志为 `Connection refused`。
因此旧 train 目录只有初始化配置，没有训练指标或 checkpoint。

新增独立入口
`scripts/dt_experiments/run_qt_corrected_eta001_hcmr_seed0_direct.sbatch`：

- 只在重试进程内清除大小写 HTTP/HTTPS/ALL proxy 变量，不修改系统代理。
- 先探测到 W&B API 的直接 HTTPS 连接；未认证 GET `/graphql` 返回 405 属正常。
- 再执行原冻结启动脚本，完整保留其 smoke → 校验 → 正式训练流程。
- 重试沿用 2026-10-03 已验证的训练源码快照；训练、数据、eta、种子、预算与
  评测设置不变。原作业、源码快照和结果全部保留，新作业使用独立 job 目录。

登录节点直连 W&B API 检查已通过；计算节点的连接结果和正式 W&B run 是否创建，
须以重试日志为准。

重试于 2026-10-04 15:10:36 提交为 Slurm job **12177**；提交后查询为
`PENDING (Priority)`，尚未获得节点。没有取消或调整其他作业。

- 重试提交记录：`checkpoints/qt-terminal-corrected-eta001/submission-direct-20261004-151031/submission.json`
- 该目录 `source/` 中训练 runner、terminal correction、原启动脚本逐字节匹配
  第一次提交的冻结源码；新 direct 启动脚本与源码一起记录 SHA-256。
- 日志：`logs/qt-corrected-direct-12177.out`
- 结果：`checkpoints/qt-terminal-corrected-eta001/job-12177/{smoke,train}/`
- 验证：shell 语法、六个代理变量不会传入训练子进程、网络失败时不启动训练，
  以及干净 tracked checkout 中与 GitHub 相同的 Ruff 检查均通过。
