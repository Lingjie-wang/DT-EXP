# Project version control

The user requires every completed code change in this project to be recorded on
GitHub. After each coherent change and the appropriate validation, create a Git
commit and push it to the project's existing GitHub remote without asking for
confirmation again. This requirement also applies to follow-up fixes.

- Keep each commit scoped to the current task. Preserve unrelated working-tree
  changes and do not include them unless the user requests it.
- Include relevant tests, configuration, scripts, and experiment documentation
  with the implementation so the version can be reproduced.
- Keep datasets, checkpoints, generated run artifacts, environments, and secrets
  out of Git, following the repository's ignore rules.
- Use normal pushes; do not force-push or rewrite shared history. If the remote
  has advanced, reconcile the changes while preserving other work.
- Report the commit and whether the push succeeded. If pushing is blocked, state
  the reason explicitly instead of implying the changes reached GitHub.
