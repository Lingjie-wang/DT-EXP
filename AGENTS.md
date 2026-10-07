# Project version control

## Experiment budgets and scheduling

The user delegates planning for requested experiments: inspect existing learning
curves and both servers' seed reservations before choosing a common, documented
training budget. Pair comparison arms on the same new seeds. Account for GPU
utilization, memory and observed throughput; run concurrently when useful and
otherwise install a durable queue that counts already-running jobs. Record the
chosen seeds, budget, scheduling limits and rationale before new training starts.
Do not silently shorten historical runs or expand the requested methods/datasets.

## Preserve historical experiments

The user requires new experimental work to preserve existing code and results.
Add independent entry points, configurations, and output directories for new
methods or reproductions. Do not overwrite, replace, or repurpose historical
experiment implementations, configurations, checkpoints, or result directories.
Apply a fix to existing experiment code only when the user explicitly requests
that fix; otherwise, implement the change in a separately named version. Keep
unrelated working-tree changes intact. Record this constraint in future handoffs.

## Faithful paper reproductions

论文复现默认保留官方源代码、原始配置、训练流程和评测流程。非必要不修改
上游代码，也不通过 monkey patch 或重写外围流程悄悄改变算法行为。
优先通过独立运行环境解决依赖和硬件兼容问题。确实无法运行时，仅做最小必要
的兼容修改，明确记录原始报错、修改位置及行为影响；未验证等价性的版本不能
称为未经修改的官方复现。疑似算法错误、性能优化和额外“修复”应另建独立对照，
不得混入官方基线。用户要求的任务、数据集和奖励设置变更应单独记录。
复现结果不达标时，先核查与官方的差异，不默认修改算法或用较高分替代忠实性。

## Publish completed changes

The user requires every completed code change in this project to be recorded on
GitHub. After each coherent change and the appropriate validation, create a Git
commit and push it to the project's existing GitHub remote without asking for
confirmation again. This requirement also applies to follow-up fixes.

- Keep each commit scoped to the current task. Preserve unrelated working-tree
  changes and do not include them unless the user requests it.
- Include relevant tests, configuration, scripts, and experiment documentation
  with the implementation so the version can be reproduced.
- Before pushing, run the repository's CI checks against a clean export of the
  tracked files with the proposed changes. Ignored local directories such as
  `wandb/` can otherwise change Ruff's import classification.
- After pushing, inspect GitHub Actions and report the result. A successful push
  and a passing CI check are separate outcomes.
- Keep datasets, checkpoints, generated run artifacts, environments, and secrets
  out of Git, following the repository's ignore rules.
- Use normal pushes; do not force-push or rewrite shared history. If the remote
  has advanced, reconcile the changes while preserving other work.
- Report the commit and whether the push succeeded. If pushing is blocked, state
  the reason explicitly instead of implying the changes reached GitHub.
