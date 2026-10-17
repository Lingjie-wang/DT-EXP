"""Check the official ordinary-DT training step on CUDA using synthetic tensors.

This is a runtime diagnostic, not an AuctionNet experiment or score.
"""

import argparse
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream))
    from network import DecisionTransformer
    from trainner import Trainer

    config = {}
    for name in ("default.yaml", "env/AuctionNet.yaml", "algo/dt.yaml"):
        config.update(yaml.safe_load((args.upstream / "config" / name).read_text()))
    config.update(state_mean=np.zeros(16), state_std=np.ones(16))
    torch.manual_seed(0)  # Confined to this diagnostic process.
    torch.set_num_threads(4)
    model = DecisionTransformer(config).cuda()
    optimizer = torch.optim.AdamW(model.get_decision_transformer_parameters(),
                                  lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    trainer = Trainer(model=model, optimizer=optimizer, batch_size=64,
                      dataset=SimpleNamespace(states=np.ones((2, 16))),
                      writer=None, config=config)
    batch, length = config["batch_size"], config["K"]
    states = torch.randn(batch, length, 16, device="cuda")
    actions = torch.randn(batch, length, 1, device="cuda")
    rewards = torch.rand(batch, length, 1, device="cuda")
    dones = torch.zeros(batch, length, dtype=torch.long, device="cuda")
    rtg = torch.rand(batch, length + 1, 1, device="cuda")
    timesteps = torch.arange(length, device="cuda").repeat(batch, 1)
    mask = torch.ones(batch, length, device="cuda")
    loss = trainer.train_step(states, actions, rewards, dones, rtg, timesteps, mask)
    torch.cuda.synchronize()
    assert np.isfinite(loss)
    assert all(p.grad is None for p in model.get_mmd_parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in model.get_decision_transformer_parameters())
    report = {"python": platform.python_version(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "capability": list(torch.cuda.get_device_capability()),
              "batch": batch, "context": length, "diagnostic_loss": loss,
              "finite_gradients": True, "mmd_gradients_absent": True,
              "purpose": "synthetic runtime diagnostic, not a benchmark score"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
