#!/usr/bin/env python3

from _run_group import run_group

DATASETS = (
    "maze2d-umaze-v1",
    "maze2d-medium-v1",
    "maze2d-large-v1",
)


if __name__ == "__main__":
    raise SystemExit(run_group(DATASETS))
