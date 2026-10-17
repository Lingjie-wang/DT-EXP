"""Reuse the established AntMaze Shapley design with only sequence/return inputs."""

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.cql_antmaze_official_5090.common import read
from scripts.cql_antmaze_success_5090.fit import fit

def run(root):
    plan = read(root / "plan.json")
    work = root / "predictor/umaze"
    with np.load(work / "inputs.npz") as stored:
        if set(stored.files) != {"features", "offsets", "returns", "groups"}:
            raise ValueError("Unexpected information in predictor input")
        features = stored["features"]
        part = {key: stored[key] for key in ["offsets", "returns", "groups"]}
    torch.set_num_threads(2)
    fit(features, part, work / "fit", device="cpu", settings=plan["attribution"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root.resolve())
