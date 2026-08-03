#!/usr/bin/env python3

from _run_group import run_group

DATASETS = (
    "antmaze-umaze-v2",
    "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",
)


if __name__ == "__main__":
    raise SystemExit(run_group(DATASETS))
