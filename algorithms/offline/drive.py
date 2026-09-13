"""DRIVE adapted to D4RL Gym tasks and the project's delayed-reward protocol.

This implements the three inference components in Cui et al. (2026): a GMM
Decision Transformer, RTG-filtered retrieval over contextual embeddings, and
an IQL double-Q critic which ranks the combined candidate set.  The paper does
not release a delayed-reward D4RL configuration, so the reward transformation
here deliberately follows the existing ``dt.py`` protocol for both the policy
and critic.
"""

import copy
import os
import uuid
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dt import (
    SequenceDataset,
    set_seed,
    TransformerBlock,
    validate_reward_mode,
    wandb_init,
    wrap_env,
)
from iql import asymmetric_l2_loss, soft_update, TwinQ, ValueFunction
from torch.utils.data import DataLoader
from tqdm.auto import trange

@dataclass
class TrainConfig:
    # Tracking
    project: str = "CORL-DDR"
    group: str = "DRIVE-D4RL"
    name: str = "DRIVE"
    # DT backbone: the D4RL configuration reported in DRIVE Appendix A.4.
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 1000
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    max_action: float = 1.0
    # GMM policy (the paper does not publish M; five is a standard compact MDN).
    gmm_components: int = 5
    gmm_log_std_min: float = -5.0
    gmm_log_std_max: float = 2.0
    policy_learning_rate: float = 1e-4
    policy_weight_decay: float = 1e-4
    policy_batch_size: int = 64
    policy_update_steps: int = 100_000
    warmup_steps: int = 10_000
    clip_grad: Optional[float] = 0.25
    # IQL critic: D4RL values reported in DRIVE Appendix A.4.
    critic_learning_rate: float = 3e-4
    critic_batch_size: int = 256
    critic_update_steps: int = 100_000
    critic_hidden_dim: int = 256
    critic_num_hidden: int = 2
    discount: float = 0.99
    expectile: float = 0.7
    target_tau: float = 0.005
    # Candidate generation and direct vector retrieval.
    generated_candidates: int = 32
    retrieval_pool_size: int = 27
    retrieved_candidates: int = 9
    retrieval_zero_rtg_epsilon: float = 1e-8
    index_batch_size: int = 256
    # Dataset/evaluation.  reward_scale matches the existing CORL DT setting.
    env_name: str = "halfcheetah-medium-replay-v2"
    reward_mode: str = "delayed"
    reward_scale: float = 0.001
    num_workers: int = 4
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    eval_episodes: int = 100
    eval_every: int = 5_000
    # General.
    checkpoints_path: Optional[str] = None
    train_seed: int = 0
    eval_seed: int = 42
    deterministic_torch: bool = False
    device: str = "cuda"

    def __post_init__(self):
        validate_reward_mode(self.reward_mode)
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


class GMMDecisionTransformer(nn.Module):
    """The paper's standard DT backbone with a mixture-density action head."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int,
        episode_len: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        embedding_dropout: float,
        max_action: float,
        gmm_components: int,
        log_std_min: float,
        log_std_max: float,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)
        self.out_norm = nn.LayerNorm(embedding_dim)
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.gmm_head = nn.Linear(
            embedding_dim, gmm_components * (1 + 2 * action_dim)
        )
        self.seq_len = seq_len
        self.episode_len = episode_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gmm_components = gmm_components
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def contextual_features(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        time_steps: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = states.shape[:2]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        action_emb = self.action_emb(actions) + time_emb
        return_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb
        sequence = (
            torch.stack([return_emb, state_emb, action_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )
        if padding_mask is not None:
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
        out = self.emb_drop(self.emb_norm(sequence))
        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)
        return self.out_norm(out)[:, 1::3]

    def gmm_parameters(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        time_steps: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.contextual_features(
            states, actions, returns_to_go, time_steps, padding_mask
        )
        raw = self.gmm_head(features)
        logits = raw[..., : self.gmm_components]
        params = raw[..., self.gmm_components :].view(
            *raw.shape[:2], self.gmm_components, 2, self.action_dim
        )
        means = torch.tanh(params[..., 0, :]) * self.max_action
        log_stds = params[..., 1, :].clamp(
            min=self.log_std_min, max=self.log_std_max
        )
        return features, logits, means, log_stds

    def negative_log_likelihood(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        time_steps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        _, logits, means, log_stds = self.gmm_parameters(
            states, actions, returns_to_go, time_steps, padding_mask=~mask.bool()
        )
        distribution = torch.distributions.Normal(means, log_stds.exp())
        component_log_prob = distribution.log_prob(actions.unsqueeze(-2)).sum(dim=-1)
        log_prob = torch.logsumexp(
            F.log_softmax(logits, dim=-1) + component_log_prob, dim=-1
        )
        valid = mask.float()
        return -(log_prob * valid).sum() / valid.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample_last_actions(
        self,
        logits: torch.Tensor,
        means: torch.Tensor,
        log_stds: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        if logits.shape[0] != 1:
            raise ValueError("Evaluation sampling expects one rollout at a time")
        mixture_probs = torch.softmax(logits[0, -1], dim=-1)
        component_ids = torch.multinomial(mixture_probs, count, replacement=True)
        selected_means = means[0, -1, component_ids]
        selected_stds = log_stds[0, -1, component_ids].exp()
        samples = selected_means + selected_stds * torch.randn_like(selected_means)
        return samples.clamp(-self.max_action, self.max_action)


class TransitionBuffer:
    """Delayed-reward transitions derived from the same trajectory split as DT."""

    def __init__(
        self,
        trajectories,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        reward_scale: float,
        device: str,
    ):
        states, actions, rewards, next_states, terminals = [], [], [], [], []
        for trajectory in trajectories:
            observations = (trajectory["observations"] - state_mean) / state_std
            length = observations.shape[0]
            states.append(observations)
            actions.append(trajectory["actions"])
            rewards.append(trajectory["rewards"] * reward_scale)
            next_state = np.empty_like(observations)
            next_state[:-1] = observations[1:]
            next_state[-1] = observations[-1]
            next_states.append(next_state)
            done = np.zeros(length, dtype=np.float32)
            done[-1] = 1.0
            terminals.append(done)
        self.states = np.concatenate(states).astype(np.float32)
        self.actions = np.concatenate(actions).astype(np.float32)
        self.rewards = np.concatenate(rewards).astype(np.float32)
        self.next_states = np.concatenate(next_states).astype(np.float32)
        self.terminals = np.concatenate(terminals).astype(np.float32)
        self.device = device

    @property
    def size(self) -> int:
        return self.states.shape[0]

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        index = np.random.randint(self.size, size=batch_size)
        return tuple(
            torch.as_tensor(array[index], dtype=torch.float32, device=self.device)
            for array in (
                self.states,
                self.actions,
                self.rewards,
                self.next_states,
                self.terminals,
            )
        )


class IQLCritic:
    def __init__(self, state_dim: int, action_dim: int, config: TrainConfig):
        self.qf = TwinQ(
            state_dim, action_dim, config.critic_hidden_dim, config.critic_num_hidden
        ).to(config.device)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(config.device)
        self.vf = ValueFunction(
            state_dim, config.critic_hidden_dim, config.critic_num_hidden
        ).to(config.device)
        self.q_optimizer = torch.optim.Adam(
            self.qf.parameters(), lr=config.critic_learning_rate
        )
        self.v_optimizer = torch.optim.Adam(
            self.vf.parameters(), lr=config.critic_learning_rate
        )
        self.discount = config.discount
        self.expectile = config.expectile
        self.target_tau = config.target_tau

    def update(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        states, actions, rewards, next_states, terminals = batch
        with torch.no_grad():
            target_q = self.q_target(states, actions)
        values = self.vf(states)
        advantage = target_q - values
        value_loss = asymmetric_l2_loss(advantage, self.expectile)
        self.v_optimizer.zero_grad()
        value_loss.backward()
        self.v_optimizer.step()

        with torch.no_grad():
            targets = rewards + (1.0 - terminals) * self.discount * self.vf(next_states)
        q1, q2 = self.qf.both(states, actions)
        q_loss = 0.5 * (F.mse_loss(q1, targets) + F.mse_loss(q2, targets))
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        soft_update(self.q_target, self.qf, self.target_tau)
        return {
            "critic/q_loss": q_loss.item(),
            "critic/value_loss": value_loss.item(),
            "critic/advantage_mean": advantage.mean().item(),
            "critic/q_mean": torch.minimum(q1, q2).mean().item(),
        }

    def state_dict(self) -> Dict[str, Dict]:
        return {
            "qf": self.qf.state_dict(),
            "q_target": self.q_target.state_dict(),
            "vf": self.vf.state_dict(),
        }


class RetrievalIndex:
    def __init__(
        self, embeddings: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor
    ):
        self.embeddings = F.normalize(embeddings, dim=-1)
        self.actions = actions
        self.returns = returns

    @torch.no_grad()
    def retrieve(
        self, query: torch.Tensor, pool_size: int, final_count: int
    ) -> torch.Tensor:
        query = F.normalize(query.reshape(1, -1), dim=-1)
        similarity = torch.mv(self.embeddings, query[0])
        pool_size = min(pool_size, similarity.numel())
        pool = torch.topk(similarity, k=pool_size).indices
        final_count = min(final_count, pool_size)
        best_return = torch.topk(self.returns[pool], k=final_count).indices
        return self.actions[pool[best_return]]


@torch.no_grad()
def build_retrieval_index(
    model: GMMDecisionTransformer, dataset: SequenceDataset, config: TrainConfig
) -> RetrievalIndex:
    """Embed every valid dataset transition with the policy's contextual state."""
    model.eval()
    embeddings, actions, returns = [], [], []
    for trajectory in dataset.dataset:
        observations = (
            trajectory["observations"] - dataset.state_mean
        ) / dataset.state_std
        trajectory_actions = trajectory["actions"]
        trajectory_returns = trajectory["returns"]
        length = observations.shape[0]
        for batch_start in range(0, length, config.index_batch_size):
            times = np.arange(
                batch_start, min(batch_start + config.index_batch_size, length)
            )
            batch_size = len(times)
            states = np.zeros(
                (batch_size, config.seq_len, model.state_dim), dtype=np.float32
            )
            actions_context = np.zeros(
                (batch_size, config.seq_len, model.action_dim), dtype=np.float32
            )
            returns_context = np.zeros((batch_size, config.seq_len), dtype=np.float32)
            time_steps = np.zeros((batch_size, config.seq_len), dtype=np.int64)
            mask = np.zeros((batch_size, config.seq_len), dtype=np.bool_)
            feature_index = []
            for row, timestep in enumerate(times):
                context_start = max(0, timestep - config.seq_len + 1)
                context_length = timestep - context_start + 1
                states[row, :context_length] = observations[
                    context_start : timestep + 1
                ]
                actions_context[row, :context_length] = trajectory_actions[
                    context_start : timestep + 1
                ]
                returns_context[row, :context_length] = (
                    trajectory_returns[context_start : timestep + 1]
                    * config.reward_scale
                )
                time_steps[row, :context_length] = np.arange(
                    context_start, timestep + 1
                )
                mask[row, :context_length] = True
                feature_index.append(context_length - 1)
            features = model.contextual_features(
                torch.as_tensor(states, device=config.device),
                torch.as_tensor(actions_context, device=config.device),
                torch.as_tensor(returns_context, device=config.device),
                torch.as_tensor(time_steps, device=config.device),
                padding_mask=~torch.as_tensor(mask, device=config.device),
            )
            selected = features[
                torch.arange(batch_size, device=config.device),
                torch.as_tensor(feature_index, device=config.device),
            ]
            # DRIVE excludes only zero-RTG entries.  Do not silently discard
            # negative-return transitions: their RTG is still meaningful for
            # a nearest-neighbor pool before its high-RTG ranking step.
            keep = np.abs(trajectory_returns[times]) > config.retrieval_zero_rtg_epsilon
            if keep.any():
                embeddings.append(selected[keep].cpu())
                actions.append(
                    torch.as_tensor(trajectory_actions[times][keep], dtype=torch.float32)
                )
                returns.append(
                    torch.as_tensor(trajectory_returns[times][keep], dtype=torch.float32)
                )
    if not embeddings:
        raise RuntimeError("Retrieval filter removed every transition")
    embedding_tensor = torch.cat(embeddings).to(config.device)
    action_tensor = torch.cat(actions).to(config.device)
    return_tensor = torch.cat(returns).to(config.device)
    model.train()
    print(f"Retrieval index entries: {embedding_tensor.shape[0]}")
    return RetrievalIndex(embedding_tensor, action_tensor, return_tensor)


@torch.no_grad()
def eval_rollout(
    model: GMMDecisionTransformer,
    critic: IQLCritic,
    retrieval_index: RetrievalIndex,
    env: gym.Env,
    target_return: float,
    config: TrainConfig,
) -> Tuple[float, int]:
    states = torch.zeros(
        1,
        model.episode_len + 1,
        model.state_dim,
        dtype=torch.float32,
        device=config.device,
    )
    actions = torch.zeros(
        1,
        model.episode_len,
        model.action_dim,
        dtype=torch.float32,
        device=config.device,
    )
    returns = torch.zeros(
        1, model.episode_len + 1, dtype=torch.float32, device=config.device
    )
    timesteps = torch.arange(model.episode_len, dtype=torch.long, device=config.device)
    timesteps = timesteps.view(1, -1)
    states[:, 0] = torch.as_tensor(env.reset(), device=config.device)
    returns[:, 0] = target_return
    episode_return = 0.0

    for step in range(model.episode_len):
        state_context = states[:, : step + 1][:, -model.seq_len :]
        action_context = actions[:, : step + 1][:, -model.seq_len :]
        return_context = returns[:, : step + 1][:, -model.seq_len :]
        timestep_context = timesteps[:, : step + 1][:, -model.seq_len :]
        features, logits, means, log_stds = model.gmm_parameters(
            state_context, action_context, return_context, timestep_context
        )
        generated = model.sample_last_actions(
            logits, means, log_stds, config.generated_candidates
        )
        retrieved = retrieval_index.retrieve(
            features[0, -1], config.retrieval_pool_size, config.retrieved_candidates
        )
        candidates = torch.cat([generated, retrieved], dim=0)
        state_for_q = states[:, step].expand(candidates.shape[0], -1)
        q_values = critic.qf(state_for_q, candidates)
        selected_action = candidates[q_values.argmax()]
        next_state, reward, done, _ = env.step(selected_action.cpu().numpy())
        actions[:, step] = selected_action
        states[:, step + 1] = torch.as_tensor(next_state, device=config.device)
        conditioning_reward = reward if config.reward_mode == "original" else 0.0
        returns[:, step + 1] = returns[:, step] - conditioning_reward
        episode_return += reward
        if done:
            return episode_return, step + 1
    return episode_return, model.episode_len


def evaluate(
    model: GMMDecisionTransformer,
    critic: IQLCritic,
    retrieval_index: RetrievalIndex,
    eval_env: gym.Env,
    config: TrainConfig,
    step: int,
) -> None:
    model.eval()
    critic.qf.eval()
    critic.vf.eval()
    for target_return in config.target_returns:
        eval_env.seed(config.eval_seed)
        episode_returns, episode_lengths = [], []
        for _ in trange(config.eval_episodes, desc="Evaluation", leave=False):
            result, length = eval_rollout(
                model,
                critic,
                retrieval_index,
                eval_env,
                target_return * config.reward_scale,
                config,
            )
            episode_returns.append(result / config.reward_scale)
            episode_lengths.append(length)
        normalized_scores = (
            eval_env.get_normalized_score(np.asarray(episode_returns)) * 100
        )
        wandb.log(
            {
                f"eval/{target_return}_return_mean": float(np.mean(episode_returns)),
                f"eval/{target_return}_return_std": float(np.std(episode_returns)),
                f"eval/{target_return}_length_mean": float(np.mean(episode_lengths)),
                f"eval/{target_return}_normalized_score_mean": float(
                    np.mean(normalized_scores)
                ),
                f"eval/{target_return}_normalized_score_std": float(
                    np.std(normalized_scores)
                ),
            },
            step=step,
        )
    model.train()
    critic.qf.train()
    critic.vf.train()


def save_checkpoint(
    model: GMMDecisionTransformer,
    critic: IQLCritic,
    dataset: SequenceDataset,
    config: TrainConfig,
) -> None:
    if config.checkpoints_path is None:
        return
    os.makedirs(config.checkpoints_path, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "critic_state": critic.state_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
            "config": asdict(config),
        },
        os.path.join(config.checkpoints_path, "drive_checkpoint.pt"),
    )


@pyrallis.wrap()
def train(config: TrainConfig) -> None:
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    wandb_init(asdict(config))
    dataset = SequenceDataset(
        config.env_name,
        seq_len=config.seq_len,
        reward_scale=config.reward_scale,
        reward_mode=config.reward_mode,
    )
    wandb.log({f"dataset/{key}": value for key, value in dataset.stats.items()}, step=0)
    eval_env = wrap_env(
        gym.make(config.env_name),
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
    )
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    model = GMMDecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
        gmm_components=config.gmm_components,
        log_std_min=config.gmm_log_std_min,
        log_std_max=config.gmm_log_std_max,
    ).to(config.device)
    print(f"DRIVE policy parameters: {sum(p.numel() for p in model.parameters())}")
    policy_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.policy_learning_rate,
        weight_decay=config.policy_weight_decay,
    )
    policy_scheduler = torch.optim.lr_scheduler.LambdaLR(
        policy_optimizer,
        lambda step: min((step + 1) / config.warmup_steps, 1.0),
    )
    if config.checkpoints_path is not None:
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as file:
            pyrallis.dump(config, file)

    train_loader = DataLoader(
        dataset,
        batch_size=config.policy_batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    train_iterator = iter(train_loader)
    for step in trange(config.policy_update_steps, desc="GMM policy training"):
        states, actions, returns, timesteps, mask = [
            value.to(config.device) for value in next(train_iterator)
        ]
        loss = model.negative_log_likelihood(states, actions, returns, timesteps, mask)
        policy_optimizer.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        policy_optimizer.step()
        policy_scheduler.step()
        wandb.log(
            {
                "policy/nll": loss.item(),
                "policy/learning_rate": policy_scheduler.get_last_lr()[0],
            },
            step=step,
        )

    retrieval_index = build_retrieval_index(model, dataset, config)
    wandb.log(
        {"retrieval/index_size": retrieval_index.returns.numel()},
        step=config.policy_update_steps,
    )
    transition_buffer = TransitionBuffer(
        dataset.dataset,
        dataset.state_mean,
        dataset.state_std,
        config.reward_scale,
        config.device,
    )
    critic = IQLCritic(state_dim, action_dim, config)
    print(f"Delayed critic transitions: {transition_buffer.size}")
    for step in trange(config.critic_update_steps, desc="IQL critic training"):
        metrics = critic.update(transition_buffer.sample(config.critic_batch_size))
        global_step = config.policy_update_steps + step + 1
        wandb.log(metrics, step=global_step)
        if (step + 1) % config.eval_every == 0 or step == config.critic_update_steps - 1:
            evaluate(model, critic, retrieval_index, eval_env, config, global_step)

    save_checkpoint(model, critic, dataset, config)
    wandb.finish()


if __name__ == "__main__":
    train()
