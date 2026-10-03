#!/bin/bash
# Fetch the exact upstream implementation; never replace an existing checkout.
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
UPSTREAM="$PROJECT/third_party/decision-diffuser"
REVISION=01ce528c30b4733dc59aa6203e46ec165561158d
if [ ! -e "$UPSTREAM" ]; then
    mkdir -p "$PROJECT/third_party"
    git clone --filter=blob:none --sparse \
        https://github.com/anuragajay/decision-diffuser.git "$UPSTREAM"
    git -C "$UPSTREAM" checkout --detach "$REVISION"
    git -C "$UPSTREAM" sparse-checkout set code
fi
test "$(git -C "$UPSTREAM" rev-parse HEAD)" = "$REVISION"
test -z "$(git -C "$UPSTREAM" status --porcelain --untracked-files=no)"
test -f "$UPSTREAM/code/diffuser/models/diffusion.py"
printf 'Verified Decision Diffuser upstream %s\n' "$REVISION"
