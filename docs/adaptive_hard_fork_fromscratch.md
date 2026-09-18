# 当前模型重评分 + 近期 reference：从头训练实验

日期：2026-09-19。状态：短跑验证已通过，正式三组已提交，详见文末启动记录。

本实验落实 v3-high 工作稿之后的新提案，使用独立训练入口，不修改原 v3-high。
这是完整训练方案的 seed-0 可行性验证，不是对所有新组件的单因素归因实验。

## 1. 三组对照

| 组别 / arm | 实际优化目标 |
| --- | --- |
| A / `dt` | 普通 CORL DT loss |
| B / `recent_ref` | DT + 近期 reference |
| C / `adaptive_pref` | DT + 当前难度优先采样的 preference + 近期 reference |

数据集为 `halfcheetah-medium-replay-v2`，delayed terminal-sum reward，seed 0。
全部从随机初始化连续训练 100000 次 optimizer updates，不加载历史 50k 权重。
模型、标准 DT batch、优化器、数据预处理等沿用 CORL medium-replay 配置。
普通 DT batch 4096、context 20、learning rate 0.0008；辅助 batch 256。

A 用于判断方法是否优于正常训练的 DT；C 与 B 用于判断新增 preference 项的作用。
B 仍使用高/低回报筛出的 pair 池中的好上下文，因此不能称为完全不含偏好信息的基线。

## 2. 时间表与损失

所有 step 表示已完成更新数。设本次更新前已完成 `t` 次更新：

```text
0～10000：仅普通 DT 梯度。
10000～20000：辅助权重线性从 0 增至预设值。
20000～100000：使用完整辅助权重。
reference 在已完成 10000、15000、20000……次更新时刷新。
pair 在已完成 10000、11000、12000……次更新时全池重评分。
```

令 `alpha(t)=clip((t-10000)/10000, 0, 1)`，C 的目标为：

$$
L=L_{DT}+\alpha(t)\left(0.05L_{pref}+0.1L_{ref}\right).
$$

B 去掉 preference 项；A 直接返回普通 DT loss，不添加零权重辅助计算图。
10000 / 20000 是这次预注册的工程起点，不是由当前 seed 调出来的最优切换点。

普通 DT loss 使用原始 recorded delayed RTG，保留 CORL masked-elementwise-MSE 的
`.mean()` 归一化。辅助分支的有效位置 RTG 全部为 12000（乘 0.001 后输入模型）。

Preference 保留 v3 的 one-sided hinge：

$$
d_i^\pm=\operatorname{MSE}(\pi_\theta(h_i^+,12000),a_i^\pm),\qquad
\ell_i=[d_i^+-\operatorname{stopgrad}(d_i^-)+0.05]_+.
$$

batch 中 active 数至少 16 时按 active 数归一化，否则按整个 batch 大小归一化。
negative 只决定门控，不产生直接排斥梯度；当前训练 forward 保留原 DT dropout。

## 3. 当前模型全池重评分，不用旧难度

复用原 seed-0 NPZ 中全部 2124 个 `valid_branch` pair，保持严格的状态、历史、
时间、动作与回报差筛选不变。只读取 `pairs`、`valid_branch`、`pair_confidence`。
不读取旧 `positive_error`、`negative_error`、`hard_pair`，不加载旧模型。

每 1000 步关闭 dropout，在 12000 条件下用当前模型对全池重评分：

$$
v_i=[d_i^+-d_i^-+0.05]_+,\qquad
b_i=\frac{c_i}{\sum_jc_j},\qquad
u_i=c_i\min(v_i/0.05,5).
$$

若 `sum(u)>0`，下一个区间的 preference 采样概率为：

$$
P_i=0.2b_i+0.8\frac{u_i}{\sum_ju_j}.
$$

若全池没有 active pair，则回退为 `P=b`。`c` 是原始状态匹配可信度，保留
20% 可信度覆盖分布；违例倍率上限为 5，避免极端违例无限主导。不是硬取 top-k。
有放回抽样、loss 权重为 1，不做重要性校正；已经通过采样强调困难样本。
倍率封顶仍不能保证有效样本数高，因此每次刷新记录 ESS 和最大采样概率。

在两次全池评分之间，训练 loss 仍以当次当前模型判断 active。
重评分和 loss 计算全程不使用环境评测分数挑 pair 或更换标签。

## 4. 近期 reference 与独立上下文采样

每 5000 步深拷贝当时模型并冻结，在接下来一段训练中保持不动；不每步同步，
不按评测 best 挑 teacher。这是近期策略约束，不是具有单调改进保证的信赖域。

$$
L_{ref}=\mathbb E_{i\sim b}
\operatorname{MSE}\left(\pi_\theta(h_i^+,12000),\pi_{ref}(h_i^+,12000)\right).
$$

关键公平性选择：reference 从固定 `b` 分布独立抽样，不使用动态 preference 分布。
B/C 使用相同独立随机数生成器种子，因此 reference 上下文索引逐步完全一致；
reference 权重本身随各组模型训练而不同，这是干预造成的预期差异。

学生与 teacher 在 anchor forward 都关闭 dropout；学生保留 autograd，teacher
禁止梯度。这个细节有别于原始 v3 的“学生 dropout 开启”，避免把随机前向噪声
混入策略偏移约束。复制当刻的确定性 anchor 应为零，后续更新使其逐渐非零。

近期模型可能跟随坏更新一起退化；这一实验不会把 reference 更新当成提升保证。

## 5. 配对、评测和保存

- 同一代码入口，三组初始化哈希必须一致；使用同型号 GPU。
- 普通 DT DataLoader 使用独立 generator / worker 随机流。
- preference 与 reference 使用独立 NumPy RNG；辅助 dropout 在 `fork_rng` 中运行，
  不改变后续普通 DT forward 的随机数流。
- 评测前保存、评测后恢复训练随机状态；所有组执行相同的诊断 forward / 全池评分。
- 保存普通 DT 抽样的轨迹编号/起点累积哈希、reference 索引累积哈希；抽查实际
  DT batch 和 Torch RNG 哈希。只上传哈希，不上传原始状态或动作。
- 10000 次更新前各组目标相同；首次辅助更新 ramp 为 0，此时模型也应保持一致。
- 每 5000 步评测 12000 / 6000 各 100 episodes，eval seed 42。
- delayed rollout 不用中途 dense reward 更新 RTG；统计环境真实总回报的 D4RL 分数。
- 主指标：100k last、80/85/90/95/100k 平均分；best 为辅助指标，按相同窗口比较。
- 本地保存 0、10k、20k、50k、75k、100k checkpoint，包括模型、优化器、scheduler、
  reference、采样概率和随机数状态。不覆盖历史目录。
- DataLoader 的 worker 状态/预取队列未序列化；入口只支持从头训练，快照供后续复用，
  不承诺直接无缝精确续训。
- W&B 原项目 `corl-ddr`：只记录配置、标量指标和哈希；不上传 checkpoint / 原数据。
- 不设置定时检查或周期监控。

## 6. 验证与启动

入口：[adaptive_hard_fork_dt.py](../algorithms/offline/adaptive_hard_fork_dt.py)。
旧方案：[v3-high 工作稿](v3_high_method.md)，原代码和 launcher 保留。

```bash
PYTHONPATH=algorithms/offline python -m unittest discover -s tests \
  -p test_adaptive_hard_fork_dt.py -v
sbatch --array=0-2 --time=00:20:00 \
  scripts/dt_experiments/run_adaptive_hard_fork_fromscratch_hcmr_seed0.sbatch smoke
python scripts/dt_experiments/audit_adaptive_hard_fork.py \
  checkpoints/adaptive-hf-fromscratch/job-SMOKE_JOB_ID --require-complete
# 只有 smoke 和跨组配对审计通过后，才提交以下正式三组。
sbatch --array=0-2 \
  scripts/dt_experiments/run_adaptive_hard_fork_fromscratch_hcmr_seed0.sbatch
```

Smoke 保留完整网络和 batch，训练 9 步；辅助在第 2 步之后开启，第 4 步后满权重，
每 3 步更换 reference、每 2 步全池评分；每 3 步评测两档各 1 episode。
因此能覆盖真正的损失启用、两次 teacher 更换、重评分和评测 RNG 隔离，而非只验证
前面纯 DT 的几步。Smoke 使用 offline W&B，不污染正式组曲线。

## 7. 启动记录

实现提交：`4555d161dc1bff43a2041501024e8f76b8868a98`，已推送 GitHub。
[codestyle 线上检查通过](https://github.com/Lingjie-wang/DT-EXP/actions/runs/35373495291)。
原 v3-high 文档单独提交为 `ef4dd59`；历史训练入口与 launcher 未更改。

短跑作业 `9346_0/1/2` 全部在 gn7 的 RTX 4090 上以 exit 0 完成，各约 36～37 秒。
8 个机制单元测试通过，三组各完成 9 次更新和 3 次双目标评测。跨组三组审计通过：

- 初始模型 SHA256 均为 `7c137560795f9c673d3a7a93cb0e93925ac93835e6994369adf8ef5912073c64`。
- 实际 batch / Torch RNG 在审计的 1、2、3、4、5、6、9 步一致。
- 辅助启用前及 ramp=0 的首次辅助更新，三组权重和 loss 一致。
- 全部 9 步的 DT 结构抽样流与 reference 抽样流累积哈希一致。
- 各组实际加权 loss 开关及总 loss 等式正确，更新后模型按预期分化。
- 只使用 2124 个原始 strict pair；pair 文件 SHA256 为
  `8ec016c96d81f0cc5b73e0e32521616ee1364f143abc0efccf07dd8f72c8b52a`。

正式数组作业：`9349`，GPUNorm，固定 gn7 / RTX 4090，`--array=0-2%2`。
优先释放 0 和 2，启动后释放 1；没有遗留手动 hold。

| 作业 | 实验 | 提交时状态 / W&B |
| --- | --- | --- |
| `9349_0` | A：DT control | 已启动，[W&B](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/02f6831f-2004-450e-9f17-2cf2f928a9e0) |
| `9349_1` | B：DT + recent reference | 等待数组并发名额；启动后自动创建 W&B run |
| `9349_2` | C：adaptive preference + recent reference | 已启动，[W&B](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/f93e3a23-c3ab-48b4-9985-b8a929e50835) |

W&B group：`AdaptiveHF-FromScratch100k-HCMR-delayed-seed0`。
本地结果根目录：服务器 `checkpoints/adaptive-hf-fromscratch/job-9349/`，
各 arm 内有唯一 run 目录，包含 `config.json`、`metrics.jsonl`、`evaluations.json`、
`pairing_audit.json`、`reference_events.json` 和阶段 checkpoint，结束后写 `summary.json`。

上述链接与状态是启动记录，不是完成结果。未创建定时监控。
