# Project version control

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
