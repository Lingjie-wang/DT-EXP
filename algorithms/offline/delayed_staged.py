"""Strict delayed-reward contracts for the corrected staged-preference study."""

import hashlib

import gym
import numpy as np
from late_preference import fingerprint

class TerminalReward(gym.Wrapper):
    """Expose zero reward until termination, then the accumulated episode return."""

    def __init__(self, env):
        super().__init__(env)
        self.episode_return = 0.0

    def reset(self, **kwargs):
        self.episode_return = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.episode_return += float(reward)
        return observation, self.episode_return if done else 0.0, done, info


def audit_delayed_dataset(dataset):
    if dataset.reward_mode != 'delayed':
        raise ValueError('Delayed rewards are mandatory for this study')
    digest = hashlib.sha256()
    transitions = 0
    for trajectory in dataset.dataset:
        rewards = trajectory['rewards']
        returns = trajectory['returns']
        if (not len(rewards) or not np.isfinite(rewards).all()
                or np.any(rewards[:-1] != 0)):
            raise ValueError('Nonterminal training rewards must all be zero')
        if not np.array_equal(returns, np.full_like(returns, rewards[-1])):
            raise ValueError('Delayed RTG must equal terminal return at every step')
        digest.update(rewards.tobytes())
        digest.update(returns.tobytes())
        transitions += len(rewards)
    return {
        'reward_mode': 'delayed',
        'trajectories': len(dataset.dataset), 'transitions': transitions,
        'all_nonterminal_rewards_zero': True, 'all_training_rtgs_constant': True,
        'transformed_rewards_and_rtgs_sha256': digest.hexdigest(),
        'preference_labels': 'terminal trajectory return only',
        'auxiliary_rtg': 'constant initial target return; no dense prefix subtraction',
        'evaluation': 'terminal reward wrapper; constant RTG; no dense reward feedback',
    }


def validate_delayed_parent(current, saved, completed):
    expected_steps = saved['update_steps'] if current.get('validation_mode') else 100000
    if (completed != expected_steps or saved['update_steps'] != expected_steps
            or saved.get('variant') != 'parent_dt'
            or saved.get('reward_mode') != 'delayed'
            or saved.get('preference_weight') != 0
            or saved.get('resume_checkpoint')):
        raise ValueError('Expected a completed ordinary delayed-reward DT parent')
    allowed = {'name', 'group', 'output_dir', 'checkpoints_path', 'variant',
               'preference_weight', 'learning_rate', 'update_steps', 'eval_every',
               'checkpoint_every', 'log_every'}
    for key, expected in saved.items():
        actual = current.get(key)
        if isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
            expected, actual = tuple(expected), tuple(actual)
        if key not in allowed and expected != actual:
            raise ValueError('Delayed parent setting differs: ' + key)


def verify_delayed_provenance(parent, child):
    for key in ('dataset_sha256', 'pairs_sha256', 'packages', 'attention_backend',
                'delayed_reward_audit'):
        if parent[key] != child[key]:
            raise ValueError('Delayed parent provenance differs: ' + key)
    for filename in ('dt.py', 'state_only_preference.py', 'top_return_weighted_dt.py',
                     'delayed_staged.py', 'late_preference.py'):
        key = 'algorithms/offline/' + filename
        if parent['source_sha256'][key] != child['source_sha256'][key]:
            raise ValueError('Delayed shared source differs: ' + filename)


def stage_objective(variant, dt_loss, preference_loss):
    if variant == 'c_only':
        if dt_loss.requires_grad:
            raise ValueError('C-only DT diagnostic must not retain an autograd graph')
        return 0.05 * preference_loss
    if variant == 'dt':
        return dt_loss
    if variant in ('b', 'c'):
        return dt_loss + 0.05 * preference_loss
    raise ValueError('Unknown delayed-stage variant')


def audit_batch_rtg(ordinary_returns, mask, auxiliary_returns, auxiliary_mask,
                    target_return):
    """Check only valid tokens: padding zeros are not RTG observations."""
    r = ordinary_returns.detach().cpu().numpy()
    valid = mask.detach().cpu().numpy().astype(bool)
    for row, selected in zip(r, valid):
        values = row[selected]
        if not len(values) or not np.all(values == values[0]):
            raise ValueError('Ordinary delayed batch contains nonconstant valid RTGs')
    ar = auxiliary_returns.detach().cpu().numpy()
    av = auxiliary_mask.detach().cpu().numpy().astype(bool)
    if not np.all(ar[av] == target_return):
        raise ValueError('Auxiliary delayed RTG is not the constant requested target')
    return {'ordinary_valid_rtgs_constant': True, 'auxiliary_valid_rtgs_constant': True,
            'ordinary_returns_sha256': fingerprint(ordinary_returns),
            'auxiliary_returns_sha256': fingerprint(auxiliary_returns)}
