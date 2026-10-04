"""Require real FlashAttention forward/backward on the allocated NVIDIA GPU."""

import argparse
import importlib.metadata
import json

import torch
from flash_attn import flash_attn_func

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    assert torch.__version__.split("+")[0] == "2.0.1"
    assert torch.cuda.get_device_capability()[0] >= 8
    tensors = [
        torch.randn(
            2, 64, 1, 128, device="cuda", dtype=torch.float16, requires_grad=True
        )
        for _ in range(3)
    ]
    output = flash_attn_func(
        *tensors, dropout_p=0.1, softmax_scale=1.0, causal=True, window_size=(32, -1)
    )
    output.float().square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(t.grad).all() for t in tensors)
    value = dict(
        flash_attention_cuda_forward_backward=True,
        gpu=torch.cuda.get_device_name(),
        memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        versions={
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "flash-attn",
                "transformers",
                "datasets",
                "accelerate",
                "numpy",
                "gym",
                "mujoco-py",
            )
        },
    )
    with open(args.output, "w") as stream:
        json.dump(value, stream, indent=2)
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
