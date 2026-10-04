"""Prepare only Hopper-medium-v2 with the author's trajectory conversion rules.

The source URL and score references come from D4RL 1.1 ``infos.py``. HTTPS is
used in place of its HTTP URL. The pinned digest identifies the downloaded
October 2026 source bytes; it is not a checksum published by D4RL.
"""

import argparse
import collections
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import urllib.request
from pathlib import Path

import h5py
import numpy as np

DATASET = "hopper-medium-v2"
SOURCE_URL = (
    "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/"
    "hopper_medium-v2.hdf5"
)
SOURCE_SHA256 = "5bdf1bc4a713c82941de44633df669b36c89850b652a25985166796d25cf71a0"
AUTHOR_SHA = "ac85be4877016724d648ee055854fd14482b5dad"
FIELDS = ("observations", "next_observations", "actions", "rewards", "terminals")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_file(temporary, destination):
    """Create a destination atomically; accept identical prior output only."""
    temporary, destination = Path(temporary), Path(destination)
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if sha256(temporary) != sha256(destination):
            raise FileExistsError(f"Refusing to replace different data: {destination}")
    finally:
        temporary.unlink()


def download(destination):
    if destination.exists():
        return
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urllib.request.urlopen(SOURCE_URL.replace("http:", "https:")) as source:
                shutil.copyfileobj(source, stream)
        except BaseException:
            temporary.unlink()
            raise
    if sha256(temporary) != SOURCE_SHA256:
        temporary.unlink()
        raise ValueError("Downloaded HDF5 does not match the pinned source SHA256")
    publish_file(temporary, destination)


def convert(dataset):
    """Mirror gym/data/download_d4rl_datasets.py, including dropping the tail."""
    paths = []
    data = collections.defaultdict(list)
    episode_step = 0
    use_timeouts = "timeouts" in dataset
    for index in range(dataset["rewards"].shape[0]):
        done_bool = bool(dataset["terminals"][index])
        final_timestep = (
            dataset["timeouts"][index] if use_timeouts else episode_step == 1000 - 1
        )
        for key in FIELDS:
            data[key].append(dataset[key][index])
        if done_bool or final_timestep:
            episode_step = 0
            paths.append({key: np.array(value) for key, value in data.items()})
            data = collections.defaultdict(list)
        episode_step += 1
    return paths


def statistics(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def prepare(hdf5, output_dir):
    source_hash = sha256(hdf5)
    if source_hash != SOURCE_SHA256:
        raise ValueError("Input HDF5 does not match the pinned Hopper-medium-v2 SHA256")
    with h5py.File(hdf5, "r") as source:
        dataset = {key: source[key][:] for key in (*FIELDS, "timeouts")}
        inventory = {}

        def visit(name, item):
            if isinstance(item, h5py.Dataset):
                inventory[name] = {"shape": list(item.shape), "dtype": str(item.dtype)}

        source.visititems(visit)
    paths = convert(dataset)
    destination = output_dir / f"{DATASET}.pkl"
    with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as stream:
        # Python 3.8/3.9's pickle.dump default in the author script is protocol 4.
        pickle.dump(paths, stream, protocol=4)
        temporary = Path(stream.name)
    publish_file(temporary, destination)

    lengths = np.array([len(path["rewards"]) for path in paths])
    returns = np.array([path["rewards"].sum() for path in paths])
    short = lengths <= 3
    audit = {
        "dataset": DATASET,
        "source_url_in_d4rl": SOURCE_URL,
        "download_url": SOURCE_URL.replace("http:", "https:"),
        "source_sha256": source_hash,
        "source_bytes": hdf5.stat().st_size,
        "hdf5_inventory": inventory,
        "author_commit": AUTHOR_SHA,
        "conversion_reference": "gym/data/download_d4rl_datasets.py",
        "conversion_rule": "append transition; split at terminal OR timeout; drop tail",
        "conversion_pickle_protocol": 4,
        "pickle_sha256": sha256(destination),
        "pickle_bytes": destination.stat().st_size,
        "raw_transitions": len(dataset["rewards"]),
        "raw_reward_statistics": statistics(dataset["rewards"]),
        "raw_terminal_count": int(dataset["terminals"].sum()),
        "raw_timeout_count": int(dataset["timeouts"].sum()),
        "raw_terminal_and_timeout_count": int(
            np.logical_and(dataset["terminals"], dataset["timeouts"]).sum()
        ),
        "converted_trajectories": len(paths),
        "converted_transitions": int(lengths.sum()),
        "discarded_incomplete_tail_transitions": int(
            len(dataset["rewards"]) - lengths.sum()
        ),
        "trajectory_length_statistics": statistics(lengths),
        "trajectory_return_statistics": statistics(returns),
        "author_prep_data_filter": {
            "rule": "len(observations) > num_samples_simclr (default 3)",
            "note": "The author's always-true condition applies this to DT too.",
            "dropped_trajectories": int(short.sum()),
            "dropped_transitions": int(lengths[short].sum()),
            "retained_trajectories": int((~short).sum()),
            "retained_transitions": int(lengths[~short].sum()),
        },
        "d4rl_normalization": {
            "ref_min_score": -20.272305,
            "ref_max_score": 3234.3,
            "formula_percent": "100 * (return - ref_min) / (ref_max - ref_min)",
            "reference": "d4rl 1.1 infos.py and offline_env.py",
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", dir=output_dir, delete=False) as stream:
        json.dump(audit, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    publish_file(temporary, output_dir / "data_audit.json")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("third_party/condt-data"))
    parser.add_argument("--hdf5", type=Path, help="Read existing verified source HDF5")
    parser.add_argument("--download", action="store_true", help="Download if absent")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hdf5 = args.hdf5 or args.output_dir / "hopper_medium-v2.hdf5"
    if args.download:
        hdf5.parent.mkdir(parents=True, exist_ok=True)
        download(hdf5)
    audit = prepare(hdf5, args.output_dir)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
