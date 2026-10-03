# Decision Diffuser：HalfCheetah-medium-replay-v2 复现

来源：[论文](https://arxiv.org/abs/2211.15657)、[官方代码](https://github.com/anuragajay/decision-diffuser/tree/main/code)。固定提交 `01ce528c30b4733dc59aa6203e46ec165561158d`。

本轮先跑 seed 0 的原始奖励 / 整局延迟奖励两组。复用官方模型、损失、CDF 归一化、200 步扩散采样和逆动力学；两组只有奖励出现时间不同。训练数据仍是全部 202 条完整轨迹，每条 1000 步。延迟组前 999 步奖励为零，最后一步为原始总回报，不是每个训练窗口末端的片段奖励。

## 复现口径

发布仓库只提供 Hopper 示例，并没有本任务专属调参配置。本轮迁移该发布配置并明确记录以下差异，不称为论文五种子精确复现：

| 设置 | 发布代码（本轮采用） | 论文附录 |
|---|---|---|
| 训练预算 | 1M 优化更新，每次累积 2 个 batch 32 | 2M train steps，batch 32 |
| 扩散步数 | 200 | 100 |
| 逆动力学隐藏宽度 | 256 | 512 |
| 状态条件 | 当前状态 | C=20 |
| U-Net embedding MLP 隐藏宽度 | 512 | 256 |

两种预算并不声称数学等价。规划 horizon 100，U-Net dim 128 / mults (1,4,8)，Adam lr 2e-4，条件 dropout 0.25，EMA 0.995（2000 步后每 10 步更新），guidance 1.2，采样噪声标准差乘 0.5。使用现有 MuJoCo 2.1 / Gym / Torch 1.13 隔离运行时，不是官方旧环境的逐版本复制。

## 回报条件

训练条件按官方 accessor 计算：`sum(gamma**j * reward[start+j]) / 400`，一直到整局结束，而非规划窗口结束。两组 gamma 均为 0.99，scale 均为 400。推理条件固定 0.9，每步重新规划并只执行第一个动作，不读取真实中间奖励来更新条件。动作输出先从 CDF 空间反归一化。

**解释限制：**延迟组起点条件等于 `gamma**(999-start) * episode_return / 400`。因此早期条件很小；固定 0.9 可能在相应状态下外推。保留这一点以检验未修改原方法，不能将可能失败直接解释为扩散模型不适合延迟奖励。400/0.9 来自发布默认值，未经本任务调参或环境分数选择。后续 gamma=1 或条件校准必须单独标记实验，不混入当前两组。

## 数据与实现核验

`algorithms/offline/decision_diffuser_repro.py` 提供 prepare / smoke / train / summarize。官方数学实现不改，仅绕开 cloud/render 的大包导入。直接 HDF5 加载避免离线准备阶段初始化 MuJoCo。两组共享初始权重、独立于模型 RNG 的均匀打乱轨迹窗口流；与官方 DataLoader 的均匀洗牌分布一致，但不是逐随机数复刻。保留官方 starts 0…899 的边界行为。

测试核对官方 accessor 的状态、动作、条件、窗口边界；终局总回报守恒；同总回报不同中间奖励产生相同 delayed 标签；恢复训练的优化器和随机状态；采样固定起始状态。短测使用完整网络和 200 步采样，两次同种子环境动作检查终局前一步结果一致，不把短测收益当策略分数。

每 10k 保存可恢复 checkpoint（模型、EMA、优化器、随机状态、采样 permutation/cursor）。每 100k（100k、200k……1M）保存独立权重和各 10 个完整评测 episode，环境种子 `271828+i`、采样种子固定 8675309。主报告最终 1M 分数，中间分数只展示学习进程。评测使用 EMA，不影响训练 RNG。所有步数是已完成的优化更新。

### 已启动 campaign 的评测频率修订（2026-10-03）

原冻结协议只评测 100k、500k、1M。用户要求每 100k 评测后，新 campaign 默认采用上述完整频率；已运行的两组保留训练代码、原协议、权重更新及优化器状态，在 `evaluation_schedule.json` 单独记录评测修订。

`supplement_evaluations.py amend` 记录已完成步数、未来应评测点和无法补齐的过去点。`run` 模式在原训练 GPU 配额中单独执行：用硬链接保留原子替换前的精确 rolling checkpoint，并核对内部的 completed-update 步数及协议；随后用原入口的 EMA 模型、条件、episode seeds 和评测函数评测。捕获线程独立运行，避免评测阻塞导致后续 checkpoint 被覆盖。训练进程不重启，评测使用自己的随机流；同卡评测可能降低训练吞吐。

本次原始奖励组新增 700k、800k、900k；延迟组修订时已超过 900k，下一次仍为 1M。过去未保留的 checkpoint 明确标记缺失，不插值、不用更晚的权重冒充。W&B 在同一 run 上继续记录，按修订后的各组实际可完成评测点核验完整性；新运行从头训练时则有完整 10 个点。结果 uploader 重启不改变训练进程。

使用任何可用 NVIDIA GPU，两组各 1 卡 / 2 CPU / 16 GB，最长 14 天；实际速度待 GPU 启动后估算。现有片段 DT 作业保留。原论文原始奖励 HCMR 得分 39.3±4.1 为五种子标准误；本轮不能用单种子声称复现其统计结论。此前 DT 是历史背景，预算、条件和模型不同，不构成严格配对。

W&B 项目 `CORL-DDR`，独立组 `DecisionDiffuser-HCMR-dense-vs-delayed-seed0-20261002`。仅同步必要超参数、数值训练/评测结果及结果表；源代码、完整协议、数据、归一化器和权重只存本地。

## 从 Git 版本复现

本项目复现适配器、测试和启动脚本记录在 Git。上一段的“源代码只存本地”指 W&B 不收集代码；GitHub 按项目版本管理要求保存代码。数据集、运行时、官方外部 checkout 和生成结果均不提交。

在已配置 `adt-delayed` 环境及 `.runtime/threshold-cu117` 的工作区，执行：

```bash
bash scripts/decision_diffuser/setup_upstream.sh
python -m pip install --no-deps --target .runtime/decision-diffuser-deps \
  -r scripts/decision_diffuser/requirements.txt
export PYTHONPATH="$PWD/.runtime/decision-diffuser-deps:$PWD/.runtime/threshold-cu117"
python -m unittest discover -s tests -p test_decision_diffuser_repro.py -v
python algorithms/offline/decision_diffuser_repro.py prepare --root results/dd-new-campaign
```

`prepare` 拒绝覆盖已有协议。Slurm 脚本接受 campaign 绝对路径和 `cpu-smoke`、`dense`、`delayed` 参数；CPU 短测提交时添加 `--gres=none --time=00:20:00`，通过后再提交两组 GPU 作业。运行时、数据集和集群路径在入口及启动脚本中明确指定，迁移集群需对应调整。

2026-10-02 启动的 campaign 执行 `results/decision-diffuser-hcmr-seed0-20261002/source/` 下哈希固定的源文件；之后为 GitHub CI 整理的格式不替换正在训练的快照。
