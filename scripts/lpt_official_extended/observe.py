"""Parse official stdout without importing or modifying the training process."""

import ast
import re

def parse_line(line, total_updates=8000):
    episode = re.search(r"Episode (\d+): Return = ([-+\deE.]+), Length = (\d+)", line)
    if episode:
        return "episode", dict(
            episode=int(episode[1]), raw_return=float(episode[2]), length=int(episode[3])
        )
    evaluation = re.search(r"Evaluation metrics at step (\d+): (\{.*\})", line)
    if evaluation:
        return "evaluation", dict(
            completed_updates=int(evaluation[1]), **ast.literal_eval(evaluation[2])
        )
    if line.strip().startswith("{"):
        try:
            value = ast.literal_eval(line.strip())
        except (ValueError, SyntaxError):
            return None
        if isinstance(value, dict) and "loss" in value and "epoch" in value:
            return "training", dict(completed_updates=round(value["epoch"]), **value)
    progress = re.search(rf"\|\s*(\d+)/{total_updates}\s*\[", line)
    if progress:
        return "progress", dict(completed_updates=int(progress[1]))
    return None
