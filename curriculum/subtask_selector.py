"""
Subtask selector: a categorical policy over reward-shaping objectives.

Selects which intermediate subtask to reward the agent for during training.
Updated by REINFORCE based on the agent's improvement on the super tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

import config

SUBTASK_TYPES: list[str] = ["none", "pickup_key", "open_door", "explore"]


class SubtaskSelector(nn.Module):
    """
    Single categorical distribution over subtask types.

    Learned by REINFORCE: reward signal = change in agent success on the
    original super tasks after training with the selected subtask shaping.
    Uses the same EMA baseline and gradient clipping as CurriculumGenerator.
    """

    def __init__(self, baseline_alpha: float = 0.05):
        super().__init__()
        self.subtask_types = SUBTASK_TYPES
        self.logits = nn.Parameter(torch.zeros(len(SUBTASK_TYPES)))

        self._baseline = 0.0
        self._baseline_alpha = baseline_alpha
        self._n_updates = 0

    #########################################################################
    # Sampling
    #########################################################################

    def sample(self) -> tuple[str, torch.Tensor]:
        """
        Sample one subtask type.

        Returns:
            subtask   — string key, one of SUBTASK_TYPES
            log_prob  — scalar tensor
        """
        dist = Categorical(logits=self.logits)
        idx  = dist.sample()
        return self.subtask_types[idx.item()], dist.log_prob(idx)

    #########################################################################
    # Update
    #########################################################################

    def update(
        self,
        log_probs: list[torch.Tensor],
        rewards: list[float],
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """REINFORCE update with EMA baseline and gradient clipping."""
        rewards_t  = torch.tensor(rewards, dtype=torch.float32)
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
            -lp * adv for lp, adv in zip(log_probs, advantages)
        ]).mean()
        dist = Categorical(logits=self.logits)
        entropy_bonus = dist.entropy()
        loss = policy_loss - config.SUBTASK_ENTROPY_COEF * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "loss":        loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy":     entropy_bonus.item(),
            "baseline":    self._baseline,
            "adv_mean":    advantages.mean().item(),
            "grad_norm":   grad_norm.item(),
            "reward_mean": batch_mean,
        }

    #########################################################################
    # Diagnostics
    #########################################################################

    def get_distribution(self) -> dict[str, float]:
        """Return probability over subtask types."""
        with torch.no_grad():
            probs = torch.softmax(self.logits, dim=0).tolist()
        return {t: round(p, 3) for t, p in zip(self.subtask_types, probs)}

    def most_likely(self) -> str:
        """Return the mode of the distribution."""
        return self.subtask_types[self.logits.argmax().item()]
