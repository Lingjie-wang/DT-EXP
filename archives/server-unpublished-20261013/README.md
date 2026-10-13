# 5090 服务器未发布源码归档（2026-10-13）

本目录保存 `/home/sckd02/workspace/DT-EXP` 中尚未进入 GitHub 的历史草稿，
用于保全源码与审计。这些文件没有作为已验证的实验入口安装到主分支原目录。
归档不包含数据集、模型、运行环境、W&B 凭据或实验结果。

## 已直接发布到 main 的内容

- 官方 AntMaze CQL 基线及独立环境恢复：保留原提交 `0d517fa`、`4dbaa1b`，
  通过正常提交历史进入 main，原 PR #2 随之标记为已合并。
- 服务器本地的三个论文笔记提交 `72dacf6`、`d31d88e`、`50d9c05`：
  保留原历史合并；Contrastive DT 笔记内容已在主分支，新增 Reinformer、
  Elastic Decision Transformer 两份笔记，没有重复替换已有内容。
- 将 `/.runtime/` 纳入忽略规则，并记录用户要求的直接发布到 main 的工作流。

## 归档内容与边界

`sources.tar.gz` 原样保存 37 个文件：35 个未发布历史实验/诊断文件，另有
2 个已被主分支新版本取代的本地副本（State-only C 文档与启动脚本）。
其中包括动作一致性分析、DT 动作模式、片段行为分析、片段延迟 DT、阈值 DT，
以及早期带适配的 LPT 实现。LPT 草稿与已发布的官方源码复现入口分开保存。

核对的 90 个服务器文件中，44 个与当时 GitHub main 内容完全相同，6 个
已存在于 GitHub 可达历史，37 个是主分支不存在的新路径，3 个是不同副本。
上述新路径中 2 份论文笔记已按正常源码路径发布；不同副本中 `.gitignore`
的新增规则已正常合入。其余 37 个文件均在此归档，未覆盖服务器原文件。

`manifest.json` 记录原始路径、权限、Git blob ID、SHA-256 和文件分类，
以及归档文件自身的 SHA-256。解包内容已逐字节与审计快照比较，所有 Python
文件通过语法解析。此检查仅确认完整保存，不代表算法正确或可直接运行。

## 已知问题

`ruff-diagnostics.json` 保留 Ruff 0.0.278 的完整静态检查结果，共 599 条：
469 条长行、56 条分号多语句、40 条冒号单行语句、27 条导入排序问题，
另有未使用导入/注释、lambda 风格问题，以及以下 3 个未定义名称：

- `algorithms/offline/threshold_dt.py:403`：`model`
- `algorithms/offline/threshold_dt.py:404`：`optimizer`
- `algorithms/offline/threshold_dt.py:405`：`scheduler`

原始缺陷如实保留，没有修改历史实验逻辑或把归档标为通过测试。仓库当前
可执行代码的 CI 与这份归档的静态诊断分别报告；CI 通过不能说明归档已修复。
后续若恢复某个实验，应在独立版本中修复并验证，不能直接覆盖较新的入口。

## 恢复到独立目录

在仓库根目录执行；目标目录必须是新目录：

```bash
mkdir restored-server-drafts-20261013
tar -xzf archives/server-unpublished-20261013/sources.tar.gz \
  -C restored-server-drafts-20261013
```

先按清单校验内容并阅读对应协议，再决定要恢复的独立实验。原服务器工作目录
仍保留原状；没有执行强制重置、清理未跟踪文件或覆盖现有运行产物。
