"""
Curriculum generator: a factored categorical policy over task template parameters.

Each parameter axis (grid_size, num_rooms, etc.) has its own independent
categorical distribution, represented as a learnable logit vector.
The joint policy is the product of these marginals.

Training: REINFORCE with an exponential moving average baseline.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

import config
from envs.template_env import TEMPLATE_SPACE, is_valid_task


class CurriculumGenerator(nn.Module):
    def __init__(self, baseline_alpha: float = 0.05):
        super().__init__()
        self.template = {
            key: list(values)
            for key, values in TEMPLATE_SPACE.items()
        }
        if "goal_type" in self.template:
            self.template["goal_type"] = [
                goal
                for goal in self.template["goal_type"]
                if goal in config.CURRICULUM_GOAL_TYPES
            ]
            if not self.template["goal_type"]:
                raise ValueError("CURRICULUM_GOAL_TYPES must include at least one goal_type")

        # One logit vector per parameter axis, initialised to uniform
        self.logits = nn.ParameterDict({
            key: nn.Parameter(torch.zeros(len(values)))
            for key, values in self.template.items()
        })

        # EMA baseline for variance reduction
        self._baseline = 0.0
        self._baseline_alpha = baseline_alpha
        self._n_updates = 0
        self._task_scores: dict[tuple[tuple[str, object], ...], float] = {}

    def _task_key(self, params: dict) -> tuple[tuple[str, object], ...]:
        return tuple(sorted(params.items()))

    def _accept_task(self, params: dict) -> bool:
        score = self._task_scores.get(self._task_key(params), 0.0)
        if score < config.NEGATIVE_TASK_THRESHOLD:
            return random.random() > config.NEGATIVE_TASK_REJECT_PROB
        return True

    def _task_entropy(self, params: dict) -> torch.Tensor:
        entropy = torch.tensor(0.0)
        for key, values in self.template.items():
            dist = torch.distributions.Categorical(logits=self.logits[key])
            idx = values.index(params[key])
            entropy = entropy + dist.entropy()
        return entropy

    #########################################################################
    # Sampling
    #########################################################################

    def sample(self) -> tuple[dict, torch.Tensor]:
        """
        Sample one task parameter dict.

        Returns:
            params    — dict mapping parameter name → concrete value
            log_prob  — scalar tensor, sum of log probs across all axes
        """
        params: dict = {}
        log_prob = torch.tensor(0.0)

        for key, values in self.template.items():
            dist = torch.distributions.Categorical(logits=self.logits[key])
            idx  = dist.sample()
            log_prob = log_prob + dist.log_prob(idx)
            params[key] = values[idx.item()]

        return params, log_prob

    def sample_valid(self, max_tries: int = 256) -> tuple[dict, torch.Tensor]:
        """
        Sample until a template leaves enough free cells for reset() to succeed.
        """
        for _ in range(max_tries):
            params, log_prob = self.sample()
            if is_valid_task(params) and self._accept_task(params):
                return params, log_prob
        raise RuntimeError("Failed to sample a valid task template")

    def sample_batch(self, n: int) -> tuple[list[dict], list[torch.Tensor]]:
        """Sample n tasks. Returns parallel lists of params and log_probs."""
        params_list, log_prob_list = [], []
        for _ in range(n):
            p, lp = self.sample()
            params_list.append(p)
            log_prob_list.append(lp)
        return params_list, log_prob_list

    #########################################################################
    # Update
    #########################################################################

    def update(
        self,
        params_batch: list[dict],
        log_probs: list[torch.Tensor],
        rewards: list[float],
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """
        REINFORCE update over a batch of (log_prob, reward) pairs.

        Uses an EMA baseline. Gradients are clipped to prevent large steps
        early in training when the reward signal is noisy.

        Returns a dict of training diagnostics.
        """
        rewards_t = torch.tensor(rewards, dtype=torch.float32)

        # Warm-start baseline from first batch mean, then EMA
        batch_mean = rewards_t.mean().item()
        if self._n_updates == 0:
            self._baseline = batch_mean
        else:
            self._baseline = (
                (1 - self._baseline_alpha) * self._baseline
                + self._baseline_alpha * batch_mean
            )
        self._n_updates += 1

        advantages = rewards_t - self._baseline

        policy_loss = torch.stack([
            -lp * adv
            for lp, adv in zip(log_probs, advantages)
        ]).mean()
        entropy_bonus = torch.stack([
            self._task_entropy(params)
            for params in params_batch
        ]).mean()
        loss = policy_loss - config.GENERATOR_ENTROPY_COEF * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        for params, reward in zip(params_batch, rewards):
            key = self._task_key(params)
            prev = self._task_scores.get(key, reward)
            self._task_scores[key] = (
                (1 - config.TASK_SCORE_ALPHA) * prev
                + config.TASK_SCORE_ALPHA * reward
            )

        return {
            "loss":        loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy":     entropy_bonus.item(),
            "baseline":    self._baseline,
            "adv_mean":    advantages.mean().item(),
            "adv_std":     advantages.std().item() if len(rewards) > 1 else 0.0,
            "grad_norm":   grad_norm.item(),
            "reward_mean": batch_mean,
        }

    #########################################################################
    # Diagnostics
    #########################################################################

    def get_distributions(self) -> dict[str, dict]:
        """Return human-readable probability distributions for logging."""
        result = {}
        with torch.no_grad():
            for key, values in self.template.items():
                probs = torch.softmax(self.logits[key], dim=0).tolist()
                result[key] = {str(v): round(p, 3) for v, p in zip(values, probs)}
        return result

    def most_likely_task(self) -> dict:
        """Return the mode of the generator distribution."""
        params = {}
        with torch.no_grad():
            for key, values in self.template.items():
                idx = self.logits[key].argmax().item()
                params[key] = values[idx]
        return params
