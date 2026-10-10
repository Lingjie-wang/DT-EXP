"""Exercise the unchanged official CLI with real W&B and verify remote telemetry."""

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import wandb

from scripts.cql_antmaze_official_5090.common import completed, read, write
from scripts.cql_antmaze_official_5090.queue import create_work, environment

def validate_runtime(root, plan):
    if wandb.__version__ != plan["required_wandb_version"]:
        raise ValueError("Expected the pinned official W&B version")
    if Path(sys.executable).absolute() != Path(plan["runtime_python"]).absolute():
        raise ValueError("Expected the isolated retry environment")
    probe = subprocess.check_output([
        sys.executable, "-c", "import d4rl, wandb; "
        "print(wandb.__version__); print(wandb.__file__)"],
        env=dict(os.environ, PYTHONPATH=f"{root / 'dependencies/d4rl'}"), text=True)
    version, module = probe.strip().splitlines()[-2:]
    isolated = Path(module).resolve().is_relative_to(Path(sys.prefix).resolve())
    if version != plan["required_wandb_version"] or not isolated:
        raise ValueError("Official import order escapes the isolated W&B runtime")
    write(root / "retry_runtime.json", dict(python=sys.executable,
          wandb_module=wandb.__file__, versions={
              name: importlib.metadata.version(name)
              for name in ["wandb", "protobuf", "numpy", "torch", "gym"]}))


def check(root):
    plan = read(root / "plan.json")
    validate_runtime(root, plan)
    for job in plan["jobs"]:
        phase = "online_cpu_preflight"
        work, argv = create_work(root, plan, job, phase, "cpu")
        argv[argv.index("--group") + 1] = root.name + "-validation"
        argv[argv.index("--name") + 1] = job["id"] + "-online-preflight"
        write(work / "command.json", dict(argv=argv, cwd=str(work)))
        env = environment(root, plan, work, phase)
        env.update(WANDB_MODE="online", CUDA_VISIBLE_DEVICES="")
        with (work / "console.log").open("x") as stream:
            result = subprocess.run(argv, cwd=work, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=600)
        if result.returncode:
            raise RuntimeError(f"Online CLI preflight failed: {work}")
        summary = completed(work, 100, 100, 1)
        records = list((work / "wandb").glob("run-*/run-*.wandb"))
        if len(records) != 1:
            raise ValueError("Missing unique native W&B record")
        identity = records[0].stem.removeprefix("run-")
        remote = wandb.Api(timeout=60).run(
            f"{plan['wandb_entity']}/{plan['wandb_project']}/{identity}")
        steps = [int(r["_step"]) for r in remote.scan_history(keys=["_step", "alpha"])]
        if remote.state != "finished" or not steps or max(steps) != 100:
            raise ValueError("Online training telemetry did not reach 100 updates")
        remote_score = remote.summary.get("d4rl_normalized_score")
        if remote_score != summary["final_normalized_score"]:
            raise ValueError("Remote evaluation differs from the completed local CLI")
        write(work / "verification.json", dict(**summary, wandb_verified=True,
              wandb_id=identity, wandb_url=remote.url, wandb_version=wandb.__version__))
        print(job["id"], "ONLINE_PREFLIGHT_VERIFIED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    check(parser.parse_args().root.resolve())
