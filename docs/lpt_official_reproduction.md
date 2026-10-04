# LPT 官方流程重新复现

本次直接执行作者发布的 `scripts/train.py`，不修改模型、Trainer、采样器或评测器。
旧的适配版实验（`algorithms/offline/lpt_repro.py`、`scripts/lpt/` 和历史结果）保留，
与本轮分开命名。项目长期约束见 `AGENTS.md`：论文复现非必要不改官方代码。

## 来源与设置

- 官方仓库：<https://github.com/mingluzhao/Latent-Plan-Transformer>
- 固定提交：`c4e77cb464c6360733a9ce76d8870f076c76d0aa`。
- 任务：HalfCheetah-medium-replay-v2；原始奖励和整条轨迹末步延迟奖励；各一个训练 seed。
- 模型初始化 CLI seed 为 0；**官方 Trainer 未传 seed，因此保留其默认 42**。
  这里不是宣称所有随机源都由 0 控制。
- 训练：2000 epoch；nominal batch 500；202 条完整轨迹使实际 batch 为 202，
  每 epoch 一次更新；不丢最后一批，不做 microbatch 或梯度累积。
- hidden 128、3 层、1 head、4 个 latent、attention window 32 tokens、
  学习率 1e-4、AdamW、weight decay 1e-4、warmup 10%、梯度裁剪 0.25。
- Langevin step size 0.3，训练 3 步，评测 20 步；reward weight 0.25，omega 1。
- 每 500 次更新使用官方评测器评测 10 局；目标总回报 6000，scale 1000；
  Gym `HalfCheetah-v3`、官方常量状态统计，不额外重设环境 episode seed。
- 模型、训练、评测和 checkpoint 保存行为全部沿用发布代码。
  不带入上一版对 PMC 索引、重复写入、动作历史、dropout、随机流的修正。
  官方 checkpoint 不保存 PMC 缓存，本轮不改写其保存协议，也不声称可等价续训。

官方的 batch 参数不是“500 个长度 20 的 DT 窗口”。原始 LPT 训练完整轨迹，
本数据集每条长度 1000。源码使用轨迹总回报作为奖励监督，虽生成 RTG 字段，
模型不使用它。两个奖励版本经过官方 collator 后的状态、动作和总回报标签相同。
因此这两组并不构成不同奖励信息量的学习问题，也不保证独立评测分数完全相同。

## 数据与必要的运行环境差异

只在独立数据准备入口中按官方 `data/process_data.py` 的 HalfCheetah 分支处理 HDF5：
以 `terminal OR timeout` 切分，保留 202 条、202000 transitions，输出官方需要的
Arrow DatasetDict 字段与目录。dense 原始逐步 float32 奖励保持不变；delayed 用
float64 累加原始奖励并放在末步，保证官方 collator 读取 Python float 列表后得到
与 dense 相同的总回报标签。未做轨迹筛选、窗口采样或重新标注。

独立环境 `.runtime/lpt-official-env`：Torch 2.0.1+cu118 与官方版本一致，使用真实
FlashAttention 2.3.6 CUDA 内核。发布环境未完整列出 datasets、accelerate 和
FlashAttention；本轮固定这些依赖。Python 3.9 替代 3.8 以复用已有 MuJoCo 绑定，
Transformers 4.37.0 release 替代未指定提交的 4.37.0.dev0，Gym-v3 使用
MuJoCo 2.1/mujoco-py。不能把这些环境差异描述成逐位一致的原始运行环境。
完整实际包版本写入运行目录。官方源码零修改不等于保证论文分数可复现。

## 执行与审计

```bash
bash scripts/lpt_official/setup_runtime.sh
.runtime/lpt-official-env/bin/python scripts/lpt_official/prepare.py \
  --root results/lpt-official-hcmr-seed0-20261004 \
  --upstream third_party/Latent-Plan-Transformer \
  --hdf5 /labmount/users/202615385/.d4rl/datasets/halfcheetah_medium_replay-v2.hdf5
sbatch scripts/lpt_official/run.sbatch \
  "$PWD/results/lpt-official-hcmr-seed0-20261004" dense
sbatch scripts/lpt_official/run.sbatch \
  "$PWD/results/lpt-official-hcmr-seed0-20261004" delayed
```

每个任务先在独立进程验证真实 FlashAttention 前向/反向，再用独立目录执行官方
`train.py --epochs 1` 做完整 batch 预检，通过后新进程从头训练正式 2000 epoch。
预检不改变正式训练随机状态。若原始 batch 显存不足，保留失败记录并调整硬件，
不自动缩 batch、换 attention 或加累积。

外围进程仅解析标准输出，保存 loss、completed updates、10 局原始回报和官方
normalized score；不导入或挂钩训练模型。冻结并核查上游源码、数据和外围脚本
SHA256，拒绝覆盖既有运行目录。调度资源为单 GPU、4 CPU、16GB、72 小时。
两个任务均使用独立目录，并且不取消其他历史实验。

W&B 同步在登录节点独立进程中执行：

```bash
/labmount/users/202615385/miniforge3/envs/adt-delayed/bin/python \
  scripts/lpt_official/sync_results.py \
  --root results/lpt-official-hcmr-seed0-20261004 --arm dense --job JOB_ID
```

目标项目 `2820402607-shandong-university/CORL-DDR`，独立组与结果目录同名。
名称为 `LPT-Official-Unmodified-HCMR-{dense,delayed}-init0-trainer42`。
只上传配置、分数、loss 和逐 episode 表，不上传代码、数据或 checkpoint；启动后
读回配置，完成后读回所有四个评测点及最终分数。W&B 进程不干预训练或 Slurm。

CPU 验证：

```bash
PYTHONPATH=scripts/lpt_official .runtime/lpt-official-env/bin/python \
  -m unittest discover -s tests -p test_lpt_official.py -v
```

验收包括终止/超时切分无丢帧、真实官方 collator 两种奖励标签一致、标准输出解析、
源码变更检测。GPU 完整 batch 预检与正式训练属于单独运行验收，排队不等于通过。
