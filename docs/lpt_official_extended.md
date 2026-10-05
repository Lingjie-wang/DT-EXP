# LPT 官方原样版：延长训练与多 seed

本轮按用户要求沿用作者公开训练和评测流程，将预算从 2000 次更新延长到
8000 次更新，并使用 CLI seed 0、1、2、3、4。先跑 HalfCheetah-medium-replay-v2
整局延迟奖励，模型仍为基础 LPT，不是 LPT-EI。

## 保留的设置与解释边界

- 上游提交仍为 `c4e77cb464c6360733a9ce76d8870f076c76d0aa`，源码不修改。
- 新入口位于 `scripts/lpt_official_extended/`；历史入口和结果全部保留。
  每个 seed 使用独立源码副本、输出目录、W&B run 和 GPU 任务。
- `--seed` 只按官方实现控制模型初始化等早期随机状态；官方 Trainer 随后仍使用
  默认 seed 42。W&B 同时标记 `initN` 和 `trainer42`，不宣称所有随机源都随 N 改变。
- 训练从头开始，完整 batch 202 条轨迹，无 microbatch/梯度累积；使用真实
  FlashAttention。学习率 1e-4、10% warmup、线性衰减到新的 8000 步预算。
  因而新 run 在第 2000 步的学习率与历史 2000 步 run 不同。
- 每 500 步评测 10 回合，共 16 个评测点；目标回报 6000。沿用官方未固定
  环境 seed 的评测、dropout、动作历史、PMC 索引和重复写入行为，不混入修复版。
- 官方 checkpoint 仍不保存 PMC 缓存，因此不声称可从历史 checkpoint 等价续训。
- 最终以各 run 第 8000 步的均分汇总 5 个 CLI seed 的均值和样本标准差；
  中间最高分只作诊断，不替代最终值，不筛选 seed。
- dense 和 delayed 的状态、动作与总回报训练标签相同，本轮只运行 delayed。
- 运行环境沿用独立 `.runtime/lpt-official-env`；与作者历史环境的差异见
  `docs/lpt_official_reproduction.md`，延长预算不意味着原论文设置已被完全复原。

## 可复现入口

准备程序校验上一轮冻结源码与数据哈希，然后复制到新的独立目录。拒绝覆盖
已有 campaign，每个任务启动前再次检查冻结源码、数据及外围脚本。

```bash
.runtime/lpt-official-env/bin/python scripts/lpt_official_extended/prepare.py \
  --baseline results/lpt-official-hcmr-seed0-20261004 \
  --root results/lpt-official-hcmr-delayed-8k-seeds0to4-20261005 \
  --epochs 8000 --seeds 0 1 2 3 4 --arm delayed

# 每个 seed 分别提交；SEED_ROOT 指向 campaign/seed0 等独立目录。
sbatch "$SEED_ROOT/source/run.sbatch" "$SEED_ROOT" delayed

# 在登录节点用已有 W&B 环境启动只读同步；不将 W&B 导入训练进程。
/labmount/users/202615385/miniforge3/envs/adt-delayed/bin/python \
  "$SEED_ROOT/source/sync_results.py" --root "$SEED_ROOT" --arm delayed --job JOB_ID
```

每个任务申请 1 GPU、4 CPU、16 GB 内存、72 小时。先在独立进程验证 CUDA
FlashAttention 前后向，再用官方 CLI 做 1 epoch 完整 batch 预检，通过后从新进程
启动正式训练。资源不足或源码校验失败时记录错误，不自动缩小 batch 或改算法。

W&B 项目仍为 `2820402607-shandong-university/CORL-DDR`，分组与 campaign 目录同名，
名称 `LPT-Official-Unmodified-HCMR-delayed-init{0..4}-trainer42-8000updates`。
只上传数值、配置和逐回合结果，横轴为 `completed_updates`。排队状态与实际训练
进度分开记录，W&B observer 在线不表示 GPU 训练已开始。

CPU 验证：

```bash
PYTHONPATH=scripts/lpt_official_extended python3 \
  -m unittest discover -s tests -p test_lpt_official_extended.py -v
```

覆盖实际命令中的 seed/预算、独立输出与身份、哈希校验、拒绝覆盖和扩展进度解析。
GPU 运行验收由每个 Slurm 任务的完整 batch 预检负责。
