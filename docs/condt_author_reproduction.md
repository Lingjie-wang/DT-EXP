# ConDT 作者公开代码：Hopper-medium 初轮复现

本轮使用第一作者后来发布的源码，在 Hopper-medium-v2 上运行 DT 与完整 ConDT
对照。用户要求结果记录到 W&B，并合理设置评测点。首轮每种方法只用 **1 个
训练种子（0）**；作者评测函数使用的环境种子 `[1, 5, 10]` 不等于三个训练种子。
本轮不宣称已完成论文的多训练种子统计复现。

- 论文：[PMLR 205, Konan et al.](https://proceedings.mlr.press/v205/konan23a.html)
- 作者源码：[SachinKonan/Contrastive-Decision-Transformers](https://github.com/SachinKonan/Contrastive-Decision-Transformers)
- 固定提交：`ac85be4877016724d648ee055854fd14482b5dad`。
- 论文最初链接的 `CORE-Robotics-Lab/ConDT` 目前没有实际训练代码。
- 入口：`scripts/condt/`；所有历史实验、代码、数据和环境保持原样。

## 固定协议

| 项目 | 设置 |
| --- | --- |
| 数据 | Hopper-medium-v2，原始稠密奖励，使用全部转换完成的轨迹 |
| 模型 | 作者 DT / `dt_contrast_simclr_product`，hidden 128、3 层、1 head、K=20、dropout 0.1 |
| 主训练 | 两组各 100,000 次更新，batch 64，10 × 10,000 |
| ConDT 预训练 | 作者代码固定的 100,000 次对比更新，另行计数；总更新 200,000 |
| 训练种子 | 两组各 0；显式设置 Python、NumPy、PyTorch 随机种子 |
| 评测环境 | 作者原始 `Hopper-v3`，最长 1000 步，目标 RTG=3600，缩放 1000 |
| 评测点 | 主训练 0、10k、20k、…、100k，共 11 点 |
| 0 点含义 | DT 是初始化后；ConDT 是完成预训练后、主训练前 |
| 每点评测 | 作者环境种子 1、5、10，每种子连续 100 局，总计 300 局 |
| 主要结果 | 最后 100k 主训练 checkpoint；不用最高中间分数替代 |
| 资源 | 正式每组 1 GPU、4 CPU、16 GB、24 小时，GPUNorm 分区 |

评测频率沿用作者主循环，兼顾学习曲线和开销。评测函数、RTG 更新、动作选择
和种子初始化方式均保留。没有增加额外目标回报扫描或使用较高的测试分数挑结果。

## 忠实性边界：保留作者代码的行为

这是一轮**带独立运行环境和观测扩展的作者代码复现**，不是逐式重写论文。
已确认公开源码和论文存在以下差异，本轮不把它们悄悄修进作者基线：

1. 正样本从同一轨迹片段有放回抽取，可能抽到重复或 padding 位置；没有按论文
   文字先从各回报桶抽样。使用 `pytorch-metric-learning==1.3.0` 的标准 NTXent。
2. ConDT 预训练后，主训练仍优化 `MSE + 0.1 * NTXent`。论文预训练方案描述为
   随后仅优化动作预测目标。源码中 `using_pretrain` 标志没有控制这一分支。
3. 作者压缩维数硬编码为 128；论文 Hopper-medium 的表格写 50。CLI 的
   `--compress_dim 50` 默认值不会覆盖这个硬编码，实际维数需按源码解释。
4. 作者 `LambdaLR` 初始化会降低预训练学习率。真实预检读回嵌入层实际初始
   lr=`6.03217e-7`，调度器 base lr=`0.00603217`。预训练不 step scheduler；
   主训练第一步手动设置嵌入 lr=`1e-4` 后，scheduler 又按原 base lr 更新。
   每次记录实际用于更新的 lr 和下一步 lr，不按参数名推断。
5. 作者过滤条件中的非空字符串使所有模型都过滤长度≤3的轨迹。本数据最短
   145 步，所以实际丢弃 0 条。训练和评测标准化的微小差别也保留。

如果后续需要检验论文公式版本或修正上述算法问题，应另建独立对照，不能覆盖
本轮源码、配置、输出或 W&B run。结果不达标时应先检查这些差异。

## 观测扩展与预检

原始仓库完整冻结在 campaign 的 `source/upstream_original/`。运行副本只加入
明确的观测调用，补丁保存在 `source_patch.diff`：

- 设置已记录的训练种子，记录依赖、硬件、参数、归一化统计和真实学习率。
- 每 100 次更新写 JSONL；每个阶段首末更新也写入。
- 对已有评测返回值保存逐局回报/长度及 checkpoint，不另写评测算法。
- checkpoint 保存模型、优化器、scheduler、Python/NumPy/Torch 随机状态。
- 原生 W&B 关闭，由独立只读桥接进程上传。原生保存代码会重复写初始评测，
  桥接记录的是当前真正返回的结果，避免把旧指标当作新评测。
- 预检有专用环境变量将 ConDT 预训练缩到 2 步；正式进程清除此变量，保留
  完整 100k 预训练。预检、正式训练使用新进程和不同输出目录。

原始数据、代码、执行副本和观测脚本均有哈希检查。已有输出禁止覆盖。测试
验证剥离允许的观测语句后核心流程 AST 不变，观测和保存不消耗训练 RNG，
预训练与主训练的计数及 checkpoint 内容正确。

2026-10-04，CPU 预检作业 **12254（DT）** 与 **12255（ConDT）** 已在 gn3
成功完成：使用真实 Hopper 模拟器、完整 batch 64、DT 两次主更新，ConDT
两次预训练加两次主更新，每点评测环境种子 1/5/10 各一局。输出保存在
`results/condt-author-hopper-medium-seed0-20261004-v2/`。CPU 吞吐较低，正式
改用 GPU 调度；正式作业仍必须先通过同一 GPU 上的短预检才能开始训练。

Ray 1.12 的作者评测任务只声明 CPU 资源，却用 `data_class.device` 创建
评测张量。通过其原生环境开关 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`
保留 Slurm 已分配 GPU 的可见性，不改变作者评测代码；不向未分配的 GPU 派任务。

## 数据与环境

数据来自 D4RL 1.1 指定的 Berkeley 地址，使用同域 HTTPS 下载。

- HDF5：1,000,000 transitions；SHA-256
  `5bdf1bc4a713c82941de44633df669b36c89850b652a25985166796d25cf71a0`。
- 严格按作者 terminal/timeout 分割逻辑转换：2,186 条轨迹、999,906 transitions；
  保留作者丢弃末尾未结束的 94 条 transitions 的行为。
- Pickle SHA-256：`23f6a54d902e066b36279f6e5d8d5c368eb7e287a8e4783d8544722f249a8ff6`。
- 转换结果已与直接执行作者转换代码所得数组和 pickle 字节逐项核验。
- D4RL normalized score 使用参考值 min=`-20.272305`、max=`3234.3`。

新环境为 `.runtime/condt-author-env`，不修改旧环境。复用可用的 MuJoCo 绑定，
使用 Python 3.9.23、Torch 1.11.0+cu113、Gym 0.23.0、NumPy 1.23.1、
Transformers 4.11.3；与作者原环境的具体差异保存在
`environment-differences.json`。Ray 1.12.0、metric-learning 1.3.0、pandas 1.4.1
等按作者版本安装；grpcio 1.43.0 满足旧 Ray 依赖约束。

新环境单独安装未经修改的 D4RL 1.1，以及其 Adroit 强制导入所需的 MJRL：
`aravindr93/mjrl@3871d93763d3b49c4741e6daeaebbc605fe140dc`。这些属于依赖解决，
没有删除作者的 Adroit 导入。包来源、真实版本及安装日志分别留档。

## 启动

```bash
bash scripts/condt/setup_runtime.sh
.runtime/condt-author-env/bin/python scripts/condt/prepare_data.py --download
git clone https://github.com/SachinKonan/Contrastive-Decision-Transformers \
  third_party/condt-author
git -C third_party/condt-author checkout ac85be4877016724d648ee055854fd14482b5dad
python3 scripts/condt/prepare.py \
  --root results/condt-author-hopper-medium-seed0-20261004-gpu \
  --upstream third_party/condt-author --data-dir third_party/condt-data
```

每个 root 必须是新的目录；已有 checkout 可直接核验固定版本，勿覆盖旧 checkout。
使用冻结脚本提交两组，任务内部执行 GPU 预检再开始完整训练：

```bash
sbatch results/condt-author-hopper-medium-seed0-20261004-gpu/source/run.sbatch \
  "$PWD/results/condt-author-hopper-medium-seed0-20261004-gpu" dt
sbatch results/condt-author-hopper-medium-seed0-20261004-gpu/source/run.sbatch \
  "$PWD/results/condt-author-hopper-medium-seed0-20261004-gpu" condt
```

## W&B

拟使用现有 `2820402607-shandong-university/CORL-DDR`，采用新的独立 group
和稳定 run ID。确认目的地后，在登录节点启动各组的冻结桥接脚本；本次已排队
campaign 的安全观测版本单独保存在 `observation-v2/`：

```bash
.runtime/condt-author-env/bin/python CAMPAIGN/observation-v2/sync_results.py \
  --root CAMPAIGN --arm dt --job SLURM_JOB_ID
```

上传白名单内的实验参数、公开源码版本、依赖版本、数据统计，以及动作损失、
对比损失、实际 lr、阶段更新数、原始回报、D4RL 分数、逐 episode 数值表。
启动前的本地 `upload_scope.json` 列出实际配置载荷。预训练与主训练分别设
横轴，同时提供总更新数。评测 std/SE 明确是**三个环境种子的均值之间**的离散
程度，另列 episode std，不作为跨训练种子置信区间。

在线上传范围限于明确列出的实验参数、公开版本信息和训练评测指标；源码、
补丁、安装日志和完整审计文件仅留在本地，不作为 W&B artifact 上传。
桥接只读已有结果和 Slurm 状态，不提交新任务、不改变训练。启动时读回配置，完成时
读回全部 11 个评测点和最终分数；失败或缺失结果不会标记为完成。持续观测
最长 7 天，进程和游标均留在独立 campaign 中。

## 检查

```bash
.runtime/condt-author-env/bin/python -m unittest discover \
  -s tests -p 'test_condt_*.py' -v
```

当前 14 项观测/同步测试与两个真实 CPU 预检通过。提交前另在干净 tracked
导出中运行仓库的 Ruff 0.0.278 检查。正式训练状态与最终成绩以各组
`status.json`、`evaluations/` 和 W&B 读回核验为准。

## 本次提交状态

正式 campaign：`results/condt-author-hopper-medium-seed0-20261004-gpu/`。
2026-10-04 18:36（Asia/Shanghai）提交 GPU 作业 **12258（DT）** 与
**12259（ConDT）**，提交时均为 `PENDING (Priority)`。没有取消或更改历史任务。

最初的 W&B 启动被自动审批拒绝，要求明确确认目标项目及上传范围。已向用户
请求确认现有 `CORL-DDR` 项目和上述限定指标范围；得到确认前不会重试在线上传。
旧的冻结桥接源码保留用于追溯，正式在线桥接将使用独立的 `observation-v2/`
快照，上传白名单配置和指标，不执行旧版 artifact 上传逻辑。
