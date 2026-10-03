"""Opt-in terminal-return correction for the pinned QT implementation.

The upstream checkout stays untouched. Only sampler episode-end flags and the
critic target/loss are replaced; actor, optimizers, EMA and inference stay upstream.
Dataset trajectory ends (including time limits) terminate this episodic delayed
objective. A nonterminal window still bootstraps from its final observed state.
"""

import ast
import inspect
import textwrap

import torch

def episode_end_flags(dones, start, length, trajectory_length):
    """Mark the true trajectory boundary without mutating the cached dataset."""
    result = dones.copy()
    if start + length == trajectory_length:
        result[0, -1, 0] = 1
    return result


def patch_episode_ends(tree):
    """Add boundary flags before upstream left-padding, guarded by AST shape."""
    matches = 0
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef) or function.name != "get_batch":
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.For):
                continue
            for index, statement in enumerate(node.body):
                if (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Call)
                    and ast.unparse(statement.value.func) == "timesteps.append"
                ):
                    addition = ast.parse(
                        "d[-1] = episode_end_flags("
                        "d[-1], si, s[-1].shape[1], len(traj['rewards']))"
                    ).body[0]
                    node.body.insert(index, addition)
                    matches += 1
                    break
    if matches != 1:
        raise RuntimeError(f"Expected one upstream sampler boundary site, got {matches}")
    return ast.fix_missing_locations(tree)


@torch.no_grad()
def terminal_multistep_targets(rewards, dones, attention_mask, bootstrap, discount):
    """Return raw-reward Q targets and the positions supervised by critic loss.

    Terminal windows include the final reward and supervise the final Q directly.
    Nonterminal windows exclude the final reward/action from the target, since
    the sampler provides no next state for that action, and bootstrap at that state.
    Left padding is excluded from the loss and never affects a valid target.
    """
    mask = attention_mask > 0
    terminal = dones[:, -1].bool()
    carry = torch.where(terminal, rewards[:, -1], bootstrap)
    targets = torch.empty_like(rewards)
    targets[:, -1] = carry
    for index in range(rewards.shape[1] - 2, -1, -1):
        carry = rewards[:, index] + discount * (1 - dones[:, index]) * carry
        targets[:, index] = carry
    mask = mask.clone()
    mask[:, -1] &= terminal.squeeze(-1)
    return targets, mask


def corrected_trainer_class(original):
    """Replace two guarded AST sites in upstream train_step, preserving the rest."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original.train_step)))
    function = tree.body[0]
    target_sites = loss_sites = 0
    body = []
    for node in function.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == "self.k_rewards":
            body.extend(ast.parse(textwrap.dedent("""
                if not self.k_rewards or not self.use_discount or self.max_q_backup:
                    raise ValueError("Terminal correction requires discounted k_rewards")
                with torch.no_grad():
                    target_q1, target_q2 = self.critic_target(
                        states[:, -1], next_action[:, -1])
                    target_q, critic_mask = terminal_multistep_targets(
                        rewards, dones, attention_mask,
                        torch.minimum(target_q1, target_q2), self.discount)
                self.last_target_q = target_q
                self.last_critic_mask = critic_mask
            """)).body)
            target_sites += 1
        elif (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "critic_loss" for t in node.targets
            )
        ):
            body.extend(ast.parse(textwrap.dedent("""
                critic_loss = (
                    F.mse_loss(current_q1[critic_mask], target_q[critic_mask])
                    + F.mse_loss(current_q2[critic_mask], target_q[critic_mask]))
            """)).body)
            loss_sites += 1
        else:
            body.append(node)
    if (target_sites, loss_sites) != (1, 1):
        raise RuntimeError("Unexpected upstream critic target/loss structure")
    function.body = body
    ast.fix_missing_locations(tree)
    namespace = dict(original.train_step.__globals__)
    namespace["terminal_multistep_targets"] = terminal_multistep_targets
    exec(compile(tree, __file__, "exec"), namespace)
    return type("TerminalCorrectedTrainer", (original,), {
        "train_step": namespace["train_step"],
        "corrected_train_step_source": ast.unparse(tree) + "\n",
    })
