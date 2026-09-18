# v3-high：原始方案与可编辑实验工作稿

文档版本：v0.1 · 整理日期：2026-09-19

方法全名：Single-Target Hard-Fork v3-high。

本文首先记录已经实现的原始 v3-high，供后续逐项讨论和修改；不把后来的 v4、扩大 pair 池、动作扰动或 PREFORL 实验混进来。第 1～8 节是原方案，第 9 节说明复现注意事项，第 10 节留给新想法。修改本文不等于已经修改代码或启动实验。

## 1. 一句话概括

先训练一个普通的 delayed-reward DT，再从相似状态和近期历史中构造“高回报轨迹动作优于低回报轨迹动作”的比较。在 RTG = 12000 下，只对尚未满足偏好间隔的样本额外模仿好动作，同时用冻结的原 DT 限制策略偏移。

总损失为：

$$
L = L_{\mathrm{DT}} + 0.05 L_{\mathrm{pref}} + 0.1 L_{\mathrm{ref}}.
$$

三个部分的分工：

| 分支 | 用什么数据 | RTG 条件 | 作用 |
| --- | --- | --- | --- |
| 普通 DT loss | 完整离线数据集的序列 | 该轨迹的 recorded RTG | 保留普通 DT 的行为建模能力 |
| Preference loss | 筛选后的好/坏动作 pair | 固定 12000 | 对仍需纠正的 pair，额外模仿好动作 |
| Reference loss | 同一批 pair 的好轨迹上下文 | 固定 12000 | 限制当前模型偏离冻结的起点模型 |

这不是 Q-learning，不学习 critic，也不做奖励重分配。负动作来自真实低回报轨迹，不是给正动作添加噪声得到的。

## 2. 数据集与 delayed reward

- 数据集：`halfcheetah-medium-replay-v2`。
- 按数据集的 `terminals` / `timeouts` 划分轨迹。
- 一条轨迹的总回报记为 $R(\tau)=\sum_{t=0}^{T-1}r_t$。
- 将奖励集中到最后一步：

$$
\widetilde r_t =
\begin{cases}
0, & t<T-1,\\
R(\tau), & t=T-1.
\end{cases}
$$

使用 $\gamma=1$，因此同一条轨迹内每个有效时间步的 recorded RTG 都是 $R(\tau)$。训练并不使用逐步 dense reward 进行信用分配；筛选轨迹用的是这个总回报。

模型输入的 RTG 乘以 `reward_scale=0.001`，所以文中原始单位的 12000 对应输入值 12。状态按整个数据集的均值和标准差标准化，标准差加 `1e-6`。

## 3. 离线构造 preference pair

### 3.1 先找候选匹配

1. 按轨迹总回报分组：回报不低于第 70 百分位的轨迹作为好轨迹，不高于第 30 百分位的轨迹作为坏轨迹，即近似 top 30% / bottom 30%。
2. 将时间步按宽度 25 分桶，只在同一个桶内匹配。
3. 抽取一个好轨迹状态，在该桶的坏轨迹状态中随机抽 64 个候选，保留标准化状态距离最近的一个。
4. 候选生成预算为 100000；去除重复 pair 后，只保留当前状态距离最小的约 25%。这里的 25% 是候选 pair 的比例，不是再取 top 25% 轨迹。

当前状态距离是标准化状态向量之间的 RMSE：

$$
\delta_i=\sqrt{\frac{1}{d_s}\left\|\bar s_i^+-\bar s_i^-\right\|_2^2}.
$$

### 3.2 再检查近期历史和分叉差异

通过上一步的 pair 还要同时满足：

| 条件 | 原始阈值 |
| --- | --- |
| 可比较的状态历史长度 | 至少 5 步 |
| 最近 5 步状态序列的标准化 RMSE，包含当前状态 | ≤ 0.75 |
| 当前动作之前的最近 4 步动作序列 RMSE | ≤ 0.50 |
| 两端时间步差 | ≤ 5 |
| 当前两个动作的 RMSE | ≥ 0.25 |
| 好轨迹总回报 − 坏轨迹总回报 | ≥ 2000，原始回报单位 |

历史比较是按距各自当前时刻的相对位置对齐，RMSE 在对应时间步和特征维度上一起取平均。动作距离使用原动作尺度，不做状态式的标准化。

满足这些条件的 pair 被标记为 `valid_branch`。原始 seed 0 文件中有 **2124 个**这样的 pair；这是该文件的统计值，不是算法固定要求的数量，其他 seed 使用各自的 pair 文件。

重要区别：训练加载的是全部 `valid_branch`，不是只加载诊断文件中的 `hard_pair`。已经满足偏好的 pair 仍在池中，只是在当次训练不满足 active 条件时，其 preference 梯度为零。

### 3.3 一个 pair 实际给模型什么输入

一个 pair 存储四个索引：`(好轨迹编号, 好时间步, 坏轨迹编号, 坏时间步)`。

训练时：

- 从好轨迹取截至当前状态、最长 20 步的上下文，记为 $h_i^+$；历史动作来自该好轨迹。
- 好轨迹当前动作作为 $a_i^+$；坏轨迹匹配位置的当前动作作为 $a_i^-$。
- 将好上下文中所有有效位置的 RTG 改为 12000，padding 位置仍为零。
- 在这个同一上下文下得到一个预测 $\hat a_i=\pi_\theta(h_i^+,12000)$，分别与 $a_i^+$、$a_i^-$ 比较。

因此，不是分别在好历史和坏历史上各算一个策略分数，也不是整条序列的 likelihood preference。它是“利用序列上下文预测当前动作”的单步动作误差比较。

筛选时检查的 5 步状态 / 4 步动作，不等于 DT 的完整 20 步上下文。DT 使用因果注意力，并从当前 state token 预测动作，不会读取作为标签的当前 action token。

## 4. 损失函数

### 4.1 普通 DT loss：保持 CORL 设置

从完整数据集采样普通 DT batch，轨迹按长度采样，再随机选序列起点。它不限制在 top 30% 轨迹，也不统一改成 RTG = 12000。

设 batch 大小为 $B$，序列长度为 $K$，动作维度为 $d_a$，有效位置 mask 为 $M_{bt}$：

$$
L_{\mathrm{DT}}
=\frac{1}{BKd_a}\sum_{b,t,j}M_{bt}
\left(\hat a^{\mathrm{recorded}}_{btj}-a_{btj}\right)^2.
$$

这对应代码的“逐元素 MSE × mask 后 `.mean()`”；分母包含 padding 位置，不是改成按有效 token 数重新归一化。

### 4.2 Preference loss：负动作作为停止条件

在同一个好上下文、同一个 RTG = 12000 条件下，定义：

$$
d_i^+=\frac{1}{d_a}\|\hat a_i-a_i^+\|_2^2,
\qquad
d_i^-=\frac{1}{d_a}\|\hat a_i-a_i^-\|_2^2.
$$

这里 $d^+$、$d^-$ 是动作 MSE，不是前面的状态 RMSE。$d_i^+<d_i^-$ 表示预测动作更接近好动作。

单个 pair 的损失为：

$$
v_i=d_i^+-\operatorname{stopgrad}(d_i^-)+m,
\qquad
\ell_i=\max(0,v_i),
\qquad m=0.05.
$$

`stopgrad` 对应代码 `.detach()`：负动作距离在本次反向传播中当作常数，但每个训练 step 都会随当前模型重新计算。

- 若 $d_i^- - d_i^+ \geq m$，这个 pair 已满足间隔，preference loss 为零。
- 若 $d_i^- - d_i^+ < m$，它就是当次的 **active pair**，产生朝好动作靠近的梯度。
- active 不只包括“更接近坏动作”的 pair，也包括“已经更接近好动作，但优势不足 0.05”的 pair。

忽略 batch 归一化，在 active 区间内：

$$
\nabla_{\hat a_i}\ell_i=\frac{2}{d_a}(\hat a_i-a_i^+).
$$

所以这个版本本质上是“由负动作控制是否启用的额外正样本模仿”，不是同时拉近正样本、推远负样本的双向梯度。

### 4.3 Active-only normalization

每步抽取 $B_p=256$ 个 pair，令 $n_A=\sum_i\mathbf 1[v_i>0]$。原版启用了优先采样，因此进入 loss 的每个 pair 权重都是 1：

$$
L_{\mathrm{pref}}=
\begin{cases}
\displaystyle\frac{\sum_i\ell_i}{n_A}, & n_A\geq16,\\[6pt]
\displaystyle\frac{\sum_i\ell_i}{256}, & n_A<16.
\end{cases}
$$

通常按 active 数量归一化，避免大量已满足 pair 稀释信号；active 太少时退回整个 batch 的均值，避免少量样本主导更新。没有 active pair 时该损失为零。计数针对抽样后的 batch，重复抽到的 pair 也计入，不是独立 pair 数。

### 4.4 Reference loss：冻结起点策略

载入 50k DT checkpoint 后，在第一次辅助更新前复制一个 reference 模型 $\pi_{\mathrm{ref}}$，之后冻结，不做 EMA 更新，也不跟随当前模型。

在同一批 pair 的好上下文上，单独做一次当前模型 forward：

$$
L_{\mathrm{ref}}
=\frac{1}{B_p d_a}\sum_i
\left\|\pi_\theta(h_i^+,12000)
-\pi_{\mathrm{ref}}(h_i^+,12000)\right\|_2^2.
$$

它约束的是所有抽到的 pair 的当前动作预测，不仅是 active pair；它也不是在完整数据集上、或所有 RTG 上施加约束。

原版使用普通 MSE anchor，没有 tolerance / trust region：任何预测偏移都会计入损失。reference 使用 `eval()`，关闭 dropout；当前模型仍在 `train()`，包括 reference 分支。因而即使刚复制、权重相同，reference loss 也可能因学生的 dropout 而非零。上式中的当前模型输出应理解为带训练时 dropout 的一次前向输出。

## 5. Pair 的两类采样优先级

这是“哪些 pair 更经常被抽到”的机制，不是额外增加一种 loss。

### 5.1 固定优先级：状态匹配可信度 × 初始难度

先根据当前状态匹配距离生成可信度：

$$
c_i=\exp\left(-\frac{\delta_i}{\max(\operatorname{median}(\delta),10^{-6})}\right).
$$

这里的距离中位数来自初始保留的近邻 pair 池，在 `valid_branch` 二次筛选之前计算。

诊断 DT 在好轨迹的 **recorded RTG** 下计算存储的正负动作误差，定义：

$$
q_i^{\mathrm{diag}}=d_{i,\mathrm{diag}}^- - d_{i,\mathrm{diag}}^+,
\qquad
u_i=\sigma\left(\frac{0.05-q_i^{\mathrm{diag}}}{0.05}\right),
$$

$$
w_i=\max(c_i u_i,10^{-6}),
\qquad b_i=\frac{w_i}{\operatorname{mean}(w)},
\qquad P_{\mathrm{base}}(i)=\frac{b_i}{\sum_jb_j}.
$$

$\sigma$ 是 sigmoid；0.05 分别是代码中的固定难度中心和 `preference_hardness_temperature`。更可信、更难的 pair 获得更高固定优先级。

注意：这个静态难度来自离线诊断的 recorded RTG；训练中的 active 判断来自当前模型的 RTG = 12000。二者不是同一个指标，也不能用前者代替后者。

### 5.2 动态优先级：当前模型的 margin 违例

初始化 $z_i=b_i$。每步仅更新本次抽到的 pair；同一个 pair 若被重复抽到，先平均其非负违例，记为 $\bar v_i^+$：

$$
z_i\leftarrow0.9z_i+0.1b_i\left(1+\frac{\bar v_i^+}{0.05}\right),
\qquad
P_{\mathrm{dyn}}(i)=\frac{z_i}{\sum_jz_j}.
$$

用于后续抽样的概率为：

$$
P(i)=0.5P_{\mathrm{base}}(i)+0.5P_{\mathrm{dyn}}(i).
$$

这一机制不是每个 step 全量重评估整个 pair 池，未抽到的 pair 保留旧动态优先级。动态项自身也乘了 $b_i$，所以把 `dynamic_priority_mix` 改成 1，并不等于完全去除固定可信度 / 旧模型难度。

原版是有放回优先采样；可信度已通过采样概率体现，loss 不再乘一次 $b_i$，也没有重要性采样校正回均匀分布。Reference 分支使用同一批抽样上下文，因此它的训练分布也会随采样优先级变化。

## 6. 完整训练流程

1. 用 delayed 数据训练普通 CORL DT，得到 50k checkpoint。
2. 离线生成、诊断 pair，保存 NPZ；v3-high 训练时直接读取，不重新挖掘 pair。
3. 从对应 seed 的 50k checkpoint 恢复当前模型，并加载其中的 optimizer / scheduler 状态；复制、冻结 reference。
4. 每次更新取一个普通 DT batch 和一个 preference batch。
5. 计算 recorded-RTG 普通 DT loss；在 12000 下计算 preference 和 reference loss。
6. 用当前 margin 违例更新抽到的 pair 的动态优先级；总损失反向传播、梯度裁剪、AdamW 更新。
7. 原始实验继续到标记为 75k 的终点，每 5k 评测一次，记录 W&B 指标和最终模型。

推理时只使用训练后的当前 DT；不需要 pair 池、reference 模型或额外检索器，也没有新增推理网络。

## 7. 原始超参数与 v2 的区别

| 设置 | v3-high 原值 |
| --- | --- |
| 普通 DT batch / preference batch | 4096 / 256 |
| Context length | 20 |
| Embedding / Transformer 层数 / heads | 128 / 3 / 1 |
| Attention / residual / embedding dropout | 均为 0.1 |
| AdamW learning rate / betas / weight decay | 0.0008 / (0.9, 0.999) / 0.0001 |
| Warmup / gradient clipping | 10000 / 0.25；续训加载已有 scheduler，不重新 warmup |
| Preference weight / reference weight / margin | 0.05 / 0.1 / 0.05 |
| Active normalization / 最小 active 数 | 开启 / 16 |
| 优先采样 / dynamic mix / EMA | 开启 / 0.5 / 0.9 |
| Frozen-hardness temperature | 0.05 |
| Preference mode / target mode | `hard_fork` / `high_only` |
| Reference anchor | `mse`，非 trust-region |
| 普通 loss 的 RTG / 辅助 loss 的 RTG | recorded / 12000 |
| 原始训练 seed | 0、1、2，分别使用对应 checkpoint 和 pair 文件 |
| GPU 分区 / W&B 项目 | `GPUNorm` / `corl-ddr`，页面显示 `CORL-DDR` |

v3-high 相对 Target-Aligned Hard-Fork v2 的核心改动是辅助分支的 RTG 条件：

| 分支 | v2 | v3-high |
| --- | --- | --- |
| 普通 DT loss | recorded RTG | 不变 |
| Preference loss | 50% recorded，25% 6000，25% 12000 | 100% 12000 |
| Reference loss | 分别约束 6000、12000，再平均 | 只约束 12000 |
| 评测 target | 12000 和 6000 | 不变 |

Pair 构造、one-sided hinge、权重、active normalization 和动态采样均保留。v3-high 不是“整个 DT 只在 12000 上训练”。

## 8. 评测与诊断指标

评测同样使用 delayed 条件：策略不使用中途 dense reward 递减 RTG，整次 rollout 的目标条件保持不变。环境真实总回报仍用于最终统计 normalized score，这不等于把 dense reward 输入给策略。

- 两个 target：12000、6000；每个 target 每次评测 100 episodes。
- `eval_seed=42`，每 5000 个训练 step 评测。
- 得分：`env.get_normalized_score(episode_return) * 100`。
- 12000 对应直接辅助优化的目标；6000 用于检查对另一目标条件的影响。
- 比较时分别列出每个 target 的 last、相同评测窗口内的 best；可辅以续训窗口的平均分，不能用不同窗口的 best 混比。

训练日志重点看：`train_loss`、`train/preference_loss`、`train/reference_loss`、两个加权辅助 loss、`train/preference_active_ratio`、`train/preference_active_count`、`train/active_normalization_fallback`、正负动作误差以及 `train/preference_accuracy`。

其中 preference accuracy 只检查 $d^+<d^-$，不要求达到 margin；即使 accuracy 很高，也可能还有 active pair。优先采样 batch 上的 active 比例，也不能直接当作整个 pair 池的均匀统计。

## 9. 复现边界、代码依据与已知局限

### 9.1 原始实现和后续公平续训协议要分开

原始 seed 0 launcher 使用 `update_steps=75001`、`paired_resume=false`。当 checkpoint 的 `next_step=50000` 时，实际执行标记为 50000～75000 的 25001 次更新；日志中的 50k 评测在第一次续训更新之后，不是未更新的起点。

后来的 `paired_resume=true` 协议改为 completed-update 计数，从 50001 到 75000 恰好更新 25000 次，并对两组恢复相同随机数状态、匹配普通 DT batch 流。那是复现协议改进，不是 v3-high loss 的新版本，详见[新 50k 配对续训说明](new50k_paired_dt_v3_to75k.md)。

原始续训不完整恢复 RNG / DataLoader 队列；仅仅同为 seed 0，不能保证得到同一个模型或同一条训练曲线。以后比较方案时应记录 checkpoint 和 pair 文件哈希，并统一续训入口、预算、评测与随机数处理。

原始脚本最终保存的 `sap_dt_checkpoint.pt` 只有模型权重和状态归一化信息，不是完整的可精确续训快照；不能仅凭配置了输出目录，就认为中间 checkpoint、optimizer 和动态优先级都已保存。

### 9.2 解释结果时不能跳过的局限

- 高回报轨迹里的某个动作不一定比低回报轨迹的匹配动作更优；整条轨迹的结果差异不能证明这个局部动作的因果优势。
- 当前状态及 5 步前缀接近，仍不保证完整 20 步 DT 上下文可互换。
- 固定 12000 是人为指定的高目标，不表示这些正样本轨迹真的实现了 12000。
- Preference 的负样本只提供训练门控，没有显式排斥负动作的梯度；收益可能来自选择性额外模仿。
- Reference 同时带来稳定性和偏移约束，而且原版含学生 dropout 的影响；不能仅看非零 reference loss 就解释为确定性策略漂移。
- 有限 pair 池和优先采样可能使少量样本被反复使用；训练偏好指标改善不等于环境回报必然改善。

这些是后续修改应针对的假设，不代表本文已经验证哪一项就是性能瓶颈。本文记录方法，不把“v3-high 稳定优于普通 DT”写成已确立结论。

### 9.3 实现依据

整理时本地仓库基准提交：`3d7a3fe4bc2b0956762c3c93440150e82b06cb25`。原始运行参数以 launcher 覆盖后的值为准，不要直接把训练类的默认值当作实验值。

- [训练入口、损失和采样](../algorithms/offline/hard_fork_dt.py)
- [原始候选 pair 构造器](../algorithms/offline/sap_dt_one_sided.py)
- [严格 pair 诊断](../algorithms/offline/diagnose_sap_pairs.py)
- [诊断作业参数](../scripts/dt_experiments/run_dt_pair_diagnostic.sbatch)
- [CORL DT 数据处理与 delayed 评测](../algorithms/offline/dt.py)
- [HalfCheetah medium-replay 配置](../configs/offline/dt/halfcheetah/medium_replay_v2.yaml)
- 原始 launcher：[seed 0](../scripts/dt_experiments/run_single_target_hard_fork_v3_high_delayed_hcmr_seed0.sbatch)、[seed 1](../scripts/dt_experiments/run_single_target_hard_fork_v3_high_delayed_hcmr_seed1.sbatch)、[seed 2](../scripts/dt_experiments/run_single_target_hard_fork_v3_high_delayed_hcmr_seed2.sbatch)。

服务器项目根目录：`/labmount/users/202615385/code/DT-EXP`。原始 seed 0 关键文件，相对于这个目录：

```text
起点 checkpoint:
checkpoints/dt-halfcheetah-medium-replay-v2-delayed-seed0-step50000.pt

pair 文件:
results/pair_diagnostics/dt_pair_diagnostic_halfcheetah_medium_replay_delayed_seed0.npz

模型输出目录:
checkpoints/single-target-hard-fork-v3-high-hcmr-delayed-seed0/

W&B name 参数（实际 run 名可能附带环境名和标识）:
SingleTarget-HardFork-v3-high-HCMR-delayed-seed0
```

原始 seed 0 文件的已有校验记录见 [reference 消融说明](v3_reference_ablation_to100k.md)：checkpoint SHA256 为 `b36430ef5e9e29a090514bf934bca2d5b66625ee6ae18da93ed60aea9e38cbbf`，pair SHA256 为 `8ec016c96d81f0cc5b73e0e32521616ee1364f143abc0efccf07dd8f72c8b52a`。这不是本次重新连接服务器校验的结果；今后开跑前应重新校验。

## 10. 后续想法与修改记录（可编辑区）

可以直接修改上面对应章节；建议在本节同时留下“原来是什么、准备改成什么”的记录，避免讨论稿和已运行配置混淆。下列内容尚未提出具体改动，也未执行新实验。

### 10.1 这次想解决的问题

待填写。

### 10.2 具体修改

| 编号 | 对应章节 | 原设置 | 拟修改为 | 预期作用 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 1 | 待填写 | 待填写 | 待填写 | 待填写 | 待讨论 |

### 10.3 需要保持一致的比较条件

- 数据和奖励处理：待确认。
- 起点 checkpoint / 是否从头训练：待确认。
- 普通 DT loss、网络和优化器：待确认。
- Pair 来源与采样：待确认。
- 总更新次数、评测窗口和 target：待确认。
- 最小对照组，以及什么结果支持 / 不支持这次假设：待填写。

### 10.4 文档版本记录

| 日期 | 文档版本 | 修改内容 | 是否改动训练代码 / 启动实验 |
| --- | --- | --- | --- |
| 2026-09-19 | v0.1 | 对照原始 launcher 和实现整理 v3-high，建立可编辑工作稿 | 否 |
