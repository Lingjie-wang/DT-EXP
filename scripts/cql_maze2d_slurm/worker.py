"""Run one allocated Slurm arm through the unchanged official CQL program."""

import argparse
import importlib.metadata
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch
import wandb

from scripts.cql_antmaze_official_5090.common import completed, read, verify, write
from scripts.cql_maze2d_four_5090.online_check import score_matches
from scripts.cql_maze2d_four_5090.queue import create_work, environment, telemetry

def stop_child(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)


def run(root, index):
    plan = read(root / "plan.json")
    job = plan["jobs"][index]
    status_path = root / job["id"] / "slurm_status.json"
    if status_path.exists():
        raise FileExistsError("Preserve the earlier attempt; recovery must be explicit")
    state = dict(state="validating", slurm_job_id=os.environ["SLURM_JOB_ID"],
                 array_job_id=os.environ["SLURM_ARRAY_JOB_ID"],
                 array_task_id=index, host=socket.gethostname(), pid=os.getpid())

    def status(**extra):
        state.update(extra, updated_unix=time.time())
        write(status_path, state)

    def interrupted(signum, frame):
        raise InterruptedError(f"Slurm worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    status()
    process = None
    try:
        verify(root, plan)
        if (Path(sys.executable).absolute() != Path(plan["runtime_python"])
                or wandb.__version__ != plan["required_wandb_version"]
                or not torch.cuda.is_available()):
            raise ValueError("Unexpected runtime or missing allocated CUDA device")
        work = root / job["id"]
        env = environment(root, plan, work / "audit", "audit")
        probe = subprocess.check_output([sys.executable, "-c",
            "import d4rl, wandb; print(wandb.__version__); print(wandb.__file__)"],
            env=env, text=True, timeout=600).strip().splitlines()
        if (probe[-2] != plan["required_wandb_version"]
                or not Path(probe[-1]).resolve().is_relative_to(Path(sys.prefix))):
            raise ValueError("D4RL import escaped the isolated W&B runtime")
        write(work / "runtime.json", dict(python=sys.executable,
              cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
              gpu=torch.cuda.get_device_name(0), versions={name:
              importlib.metadata.version(name) for name in [
                  "torch", "numpy", "gym", "wandb", "protobuf", "mujoco-py"]}))
        subprocess.run([sys.executable, "-m", "scripts.cql_maze2d_four_5090.audit",
                        "--root", str(root), "--job", job["id"]],
                       env=env, check=True, timeout=900)
        status(state="gpu_preflight")
        phase = "slurm_gpu_preflight"
        preflight, argv = create_work(root, plan, job, phase, "cuda")
        argv[argv.index("--group") + 1] = root.name + "-validation"
        argv[argv.index("--name") + 1] = job["id"] + "-slurm-gpu-preflight"
        env = environment(root, plan, preflight, phase)
        env["WANDB_MODE"] = "online"
        with (preflight / "console.log").open("x") as stream:
            subprocess.run(argv, cwd=preflight, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=900)
        summary = completed(preflight, 100, 100, 1)
        records = list((preflight / "wandb").glob("run-*/run-*.wandb"))
        if len(records) != 1:
            raise ValueError("Missing unique native W&B preflight record")
        identity = records[0].stem.removeprefix("run-")
        remote = wandb.Api(timeout=60).run(
            f"{plan['wandb_entity']}/{plan['wandb_project']}/{identity}")
        steps = [int(r["_step"]) for r in remote.scan_history(keys=["_step", "alpha"])]
        if (remote.state != "finished" or not steps or max(steps) != 100
                or not score_matches(remote.summary.get("d4rl_normalized_score"),
                                     summary["final_normalized_score"])):
            raise ValueError("GPU preflight telemetry is not verified remotely")
        write(preflight / "verification.json", dict(**summary, wandb_verified=True,
                                                    wandb_url=remote.url))
        training, argv = create_work(root, plan, job, "training", "cuda")
        with (training / "console.log").open("x") as stream:
            process = subprocess.Popen(argv, cwd=training,
                env=environment(root, plan, training, "training"), stdout=stream,
                stderr=subprocess.STDOUT)
        status(state="running", trainer_pid=process.pid, started_unix=time.time())
        complete_since = None
        while process.poll() is None:
            status(**telemetry(training, plan))
            try:
                completed(training, job["updates"], job["eval_every"],
                          job["eval_episodes"])
            except ValueError:
                pass
            else:
                complete_since = complete_since or time.time()
                if time.time() - complete_since >= 300:
                    stop_child(process)
                    status(shutdown_after_complete=True)
                    break
            time.sleep(30)
        summary = completed(training, job["updates"], job["eval_every"],
                            job["eval_episodes"])
        verify(root, plan)
        write(training / "completion.json", dict(**summary,
              process_exit=process.returncode, training_artifacts_complete=True))
        status(state="completed", **telemetry(training, plan),
               best_normalized_score=summary["best_normalized_score"],
               process_exit=process.returncode)
    except BaseException as error:
        if process is not None:
            stop_child(process)
        status(state="failed", error=repr(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int, choices=range(4))
    args = parser.parse_args()
    run(args.root.resolve(), args.index)
