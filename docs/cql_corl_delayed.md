# CORL CQL：HalfCheetah 延迟奖励实验

用户要求在 HalfCheetah-medium-v2 和 HalfCheetah-medium-replay-v2 上运行 CQL，
采用 CORL 配置及整局延迟奖励。首轮两个任务各 seed 0，独立运行 100 万次更新。
历史算法、配置和结果不修改。代码从 GitHub 最新 main 的独立工作区开始，保留
5090 服务器已提交的 AuctionNet 和奖励研究代码，不向旧 main 强制推送。

## 算法与配置

直接复用 `algorithms/offline/cql.py` 的模型、ReplayBuffer、CQL 损失、优化器更新、
状态归一化和评测函数。原文件不修改；构建网络和优化器的顺序照原训练入口复制。
新的外围入口为 `scripts/cql_delayed/`，新配置为
`configs/offline/cql/halfcheetah/{medium,medium_replay}_delayed_v2.yaml`。

保留两个原 CORL YAML 的全部科学参数：batch 256，discount 0.99，CQL alpha 10，
policy lr 3e-5，critic lr 3e-4，状态归一化，奖励不归一化；buffer capacity 1000 万。
训练 100 万步，每 5000 步按原 eval_actor 评测 10 局，使用训练 seed 0 重设环境。
未改变评测消耗训练随机数的行为。最终报告 100 万步评测，不用中间最高分替代。

## 延迟奖励及边界处理：明确的任务变更

按原 HDF5 的 `terminal OR timeout` 切分完整轨迹，前面奖励为零，最后一步奖励
为原始逐步奖励的未折扣总和。先 float64 求和，再转换为训练所用 float32。
原始 HDF5 只读，记录 SHA256，每条轨迹总回报核验，尾部不完整时拒绝启动。

不能先延迟奖励、再直接调用默认 `d4rl.qlearning_dataset`：它会丢弃 timeout
转移，恰好丢掉这些轨迹的唯一非零奖励。本实验保留全部转移，并将 true terminal
和 timeout 都视为有限回合结束，令该步 Bellman target 不 bootstrap。非结束步
使用原数据 next_observations；缺失时只在同一回合内向后取观测。结束步 next state
设为零占位且 done=1，防止跨回合连接。这是明确记录的 episodic 延迟任务定义，
不同于 CORL 默认丢弃 timeout 转移的逐步奖励数据路径。

保持 gamma=0.99，意味着延迟奖励改变了折扣目标；本轮不额外改成 gamma=1。
评测仍用原 Gym 环境的总回报及 D4RL 归一化：CQL 动作只依赖状态，不读取中间奖励
作为条件，因此评测累计回报与把奖励移到末步相同。

## 执行、日志与验证

使用现有独立 `.runtime/lpt-official-env`（Torch 2.0.1+cu118，Gym 0.23.1，D4RL 1.1）；
不修改其他实验的环境。实际版本写入 runtime.json，不宣称与早期 CORL 环境逐位一致。
登录节点的 MuJoCo 导入检查遇到新版 GCC 的 incompatible-pointer-types 编译错误；
作业脚本沿用现有 GPU 实验的 CFLAGS 兼容选项，仅降低该编译诊断，不改算法源码。
登录节点随后因缺少 `GL/osmesa.h` 无法构建 CPU 渲染扩展，故完整训练与环境评测
预检由 GPU 作业使用现有 EGL 环境执行，提交排队时尚不能声称 GPU 预检通过。

```bash
PYTHONPATH=. /path/to/.runtime/lpt-official-env/bin/python scripts/cql_delayed/prepare.py \
  --root /path/to/results/cql-corl-delayed-hc-seed0-20261006 \
  --medium /path/to/halfcheetah_medium-v2.hdf5 \
  --medium-replay /path/to/halfcheetah_medium_replay-v2.hdf5
sbatch CAMPAIGN/source/scripts/cql_delayed/run.sbatch CAMPAIGN medium
sbatch CAMPAIGN/source/scripts/cql_delayed/run.sbatch CAMPAIGN medium_replay
```

每个任务 1 GPU、2 CPU、16GB、72 小时。先运行独立进程的 100 步完整 batch
预检及 1 局评测，通过后从新进程开始完整训练，预检随机数不影响正式训练。
排队不等于预检通过；失败时保留日志，不自动改 batch 或算法。

训练每 100 步记录数值，每 5000 步保存逐回合评测；每 10 万步附加 checkpoint。
只读 W&B 同步在登录节点独立进程运行，云端横轴为 completed_updates；不将
在线请求放进训练循环。原始数据、模型、环境和凭据均不进入 Git。

- `CQL-CORL-HCM-delayed-seed0-1M`
- `CQL-CORL-HCMR-delayed-seed0-1M`

项目 `2820402607-shandong-university/CORL-DDR`，分组与 campaign 目录同名。
准备程序冻结源文件和数据哈希，并拒绝覆盖历史输出。单元验证涵盖真实结束、
timeout、零总回报、最后一条转移保留、无跨回合 bootstrap 和原数据不变。

```bash
PYTHONPATH=. /path/to/.runtime/lpt-official-env/bin/python \
  -m unittest discover -s tests -p test_cql_delayed_data.py -v
```
