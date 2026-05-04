"""
Minimal PPO agent for discrete-action MiniGrid environments.

Architecture: CNN over the 7x7x3 agent view with direction embedding
               shared trunk wth an actor head and a critic head
Training: clipped surrogate loss + value loss + entropy bonus (GAE)

Designed to be used as the inner loop of the curriculum training:
    agent.collect_rollout(env, n_steps):  rollout data
    agent.update(rollout): radient step(s)
    agent.evaluate(task_params_list, n): success rates on original tasks
"""

from __future__ import annotations

from typing import Callable

import gymnasium.vector
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import config
from envs.template_env import TemplateEnv


#############################################################################
# Network
#############################################################################

class ActorCritic(nn.Module):
    """Shared-trunk CNN policy and value network for MiniGrid image observations."""

    def __init__(self, n_actions: int = 7):
        super().__init__()

        # Input: (B, 7, 7, 3) uint8 — MiniGrid's partial-observation grid
        # Channels last → permute to (B, 3, 7, 7) before conv
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2),   # → (B, 16, 6, 6)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),  # → (B, 32, 5, 5)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),  # → (B, 64, 4, 4)
            nn.ReLU(),
            nn.Flatten(),                       # → (B, 1024)
        )

        # Direction embedding (4 possible values: 0=right,1=down,2=left,3=up)
        self.dir_embed = nn.Embedding(4, 8)

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(1024 + 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.actor  = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small gain on actor output for a near-uniform policy at init
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def _encode(
        self, image: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        # image: (B, H, W, C) uint8 → (B, C, H, W) float, normalised
        x = image.float() / 10.0   # MiniGrid encodes object/color/state as ints ≤ 10
        x = x.permute(0, 3, 1, 2)
        return self.trunk(torch.cat([self.cnn(x), self.dir_embed(direction)], dim=-1))

    def get_action_and_value(
        self,
        image: torch.Tensor,
        direction: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat   = self._encode(image, direction)
        dist   = Categorical(logits=self.actor(feat))
        value  = self.critic(feat).squeeze(-1)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


##############################################################################
# Rollout buffer
##############################################################################

class RolloutBuffer:
    """
    Fixed-length rollout storage with GAE advantage computation.
    Stores everything on `device` as tensors to avoid repeated host to device copies.
    """

    def __init__(self, n_steps: int, device: torch.device):
        self.n_steps = n_steps
        self.device  = device
        self._images:     list[torch.Tensor] = []
        self._directions: list[torch.Tensor] = []
        self._actions:    list[torch.Tensor] = []
        self._log_probs:  list[torch.Tensor] = []
        self._rewards:    list[float]        = []
        self._dones:      list[bool]         = []
        self._values:     list[torch.Tensor] = []
        # Set by vectorised collect_rollout; avoids re-computing GAE in update()
        self._precomputed_returns:    torch.Tensor | None = None
        self._precomputed_advantages: torch.Tensor | None = None
        # Completed-episode returns tracked during collection (for success-rate)
        self._episode_returns: list[float] = []

    def add(
        self,
        image:     torch.Tensor,
        direction: torch.Tensor,
        action:    torch.Tensor,
        log_prob:  torch.Tensor,
        reward:    float,
        done:      bool,
        value:     torch.Tensor,
    ) -> None:
        self._images.append(image)
        self._directions.append(direction)
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._rewards.append(reward)
        self._dones.append(done)
        self._values.append(value)

    def __len__(self) -> int:
        return len(self._rewards)

    def compute_gae(
        self,
        last_value: torch.Tensor,
        last_done:  bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (returns, advantages) tensors of shape (n_steps,)."""
        n          = len(self._rewards)
        values_t   = torch.stack(self._values)          # (n,)
        advantages = torch.zeros(n, device=self.device)
        last_gae   = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                nont  = 1.0 - float(last_done)
                nv    = last_value
            else:
                nont  = 1.0 - float(self._dones[t + 1])
                nv    = values_t[t + 1]

            delta    = self._rewards[t] + config.GAMMA * nv * nont - values_t[t]
            last_gae = delta + config.GAMMA * config.GAE_LAMBDA * nont * last_gae
            advantages[t] = last_gae

        returns = advantages + values_t
        return returns, advantages

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "images":     torch.stack(self._images),
            "directions": torch.stack(self._directions),
            "actions":    torch.stack(self._actions),
            "log_probs":  torch.stack(self._log_probs),
            "values":     torch.stack(self._values),
        }


##############################################################################
# PPO agent
##############################################################################

class PPOAgent:
    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = ActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.AGENT_LR, eps=1e-5
        )

    ##########################################################################
    # Observation helpers
    ##########################################################################

    def _obs_to_tensors(
        self, obs: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a single MiniGrid dict obs to (image, direction) tensors."""
        image = torch.as_tensor(
            obs["image"], dtype=torch.uint8, device=self.device
        ).unsqueeze(0)                              # (1, 7, 7, 3)
        direction = torch.tensor(
            [obs["direction"]], dtype=torch.long, device=self.device
        )                                           # (1,)
        return image, direction

    ##########################################################################
    # Inner-loop rollout collection
    ##########################################################################

    def collect_rollout(
        self, env_fn: Callable, n_steps: int
    ) -> RolloutBuffer:
        """
        Run N_ENVS environments in parallel for n_steps total transitions.

        env_fn  — zero-argument callable that returns a fresh env instance.
        n_steps — total transitions to collect (split evenly across envs).
        """
        n_envs        = config.N_ENVS
        steps_per_env = max(n_steps // n_envs, 1)
        total_steps   = steps_per_env * n_envs

        buf = RolloutBuffer(total_steps, self.device)

        envs = gymnasium.vector.SyncVectorEnv([env_fn] * n_envs)
        try:
            obs_dict, _ = envs.reset()
            ep_rets    = np.zeros(n_envs)
            last_dones = np.zeros(n_envs, dtype=bool)

            # Per-env trajectory lists
            ev = [
                {"images": [], "dirs": [], "actions": [], "log_probs": [],
                 "rewards": [], "dones": [], "values": []}
                for _ in range(n_envs)
            ]

            self.model.eval()
            with torch.no_grad():
                for _ in range(steps_per_env):
                    images = torch.as_tensor(
                        obs_dict["image"], dtype=torch.uint8, device=self.device
                    )
                    dirs = torch.as_tensor(
                        obs_dict["direction"], dtype=torch.long, device=self.device
                    )

                    actions, log_probs, _, values = self.model.get_action_and_value(
                        images, dirs
                    )

                    obs_dict, rewards, terminated, truncated, _ = envs.step(
                        actions.cpu().numpy().tolist()
                    )
                    last_dones = terminated | truncated

                    for i in range(n_envs):
                        ev[i]["images"].append(images[i])
                        ev[i]["dirs"].append(dirs[i])
                        ev[i]["actions"].append(actions[i])
                        ev[i]["log_probs"].append(log_probs[i])
                        ev[i]["rewards"].append(float(rewards[i]))
                        ev[i]["dones"].append(bool(last_dones[i]))
                        ev[i]["values"].append(values[i])
                        ep_rets[i] += float(rewards[i])
                        if last_dones[i]:
                            buf._episode_returns.append(float(ep_rets[i]))
                            ep_rets[i] = 0.0

                # Bootstrap values from the final observation of each env
                images = torch.as_tensor(
                    obs_dict["image"], dtype=torch.uint8, device=self.device
                )
                dirs = torch.as_tensor(
                    obs_dict["direction"], dtype=torch.long, device=self.device
                )
                _, _, _, bootstrap_values = self.model.get_action_and_value(images, dirs)

        finally:
            envs.close()

        # Compute GAE per env, then concatenate into the buffer
        all_returns, all_advantages = [], []
        for i in range(n_envs):
            d        = ev[i]
            n        = steps_per_env
            vals_t   = torch.stack(d["values"])
            adv      = torch.zeros(n, device=self.device)
            last_gae = 0.0

            for t in reversed(range(n)):
                if t == n - 1:
                    nont = 1.0 - float(last_dones[i])
                    nv   = bootstrap_values[i]
                else:
                    nont = 1.0 - float(d["dones"][t + 1])
                    nv   = vals_t[t + 1]
                delta    = d["rewards"][t] + config.GAMMA * nv * nont - vals_t[t]
                last_gae = delta + config.GAMMA * config.GAE_LAMBDA * nont * last_gae
                adv[t]   = last_gae

            all_advantages.append(adv)
            all_returns.append(adv + vals_t)

            buf._images.extend(d["images"])
            buf._directions.extend(d["dirs"])
            buf._actions.extend(d["actions"])
            buf._log_probs.extend(d["log_probs"])
            buf._rewards.extend(d["rewards"])
            buf._dones.extend(d["dones"])
            buf._values.extend(d["values"])

        buf._precomputed_returns    = torch.cat(all_returns)
        buf._precomputed_advantages = torch.cat(all_advantages)

        return buf

    ########################################################################
    # PPO update
    ########################################################################
    def update(self, buf: RolloutBuffer) -> dict:
        """
        Run PPO_EPOCHS passes over the rollout buffer with minibatch SGD.
        Returns a dict of training diagnostics.
        """
        if buf._precomputed_returns is not None:
            returns    = buf._precomputed_returns
            advantages = buf._precomputed_advantages
        else:
            returns, advantages = buf.compute_gae(buf._last_value, buf._last_done)

        # Normalise advantages over the whole rollout
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        data = buf.tensors()
        images     = data["images"]
        directions = data["directions"]
        actions    = data["actions"]
        old_lps    = data["log_probs"]

        n     = len(buf)
        idxs  = np.arange(n)

        total_pg_loss  = 0.0
        total_vf_loss  = 0.0
        total_ent      = 0.0
        total_clip_frac = 0.0
        n_updates = 0

        self.model.train()
        for _ in range(config.PPO_EPOCHS):
            np.random.shuffle(idxs)
            for start in range(0, n, config.MINIBATCH_SIZE):
                mb = idxs[start : start + config.MINIBATCH_SIZE]
                mb = torch.as_tensor(mb, device=self.device)

                _, new_lp, entropy, new_val = self.model.get_action_and_value(
                    images[mb], directions[mb], actions[mb]
                )

                ratio = (new_lp - old_lps[mb]).exp()
                adv   = advantages[mb]

                # Clipped surrogate objective
                pg_loss = torch.max(
                    -adv * ratio,
                    -adv * ratio.clamp(1 - config.CLIP_EPS, 1 + config.CLIP_EPS),
                ).mean()

                # Value loss (unclipped)
                vf_loss = 0.5 * (new_val - returns[mb]).pow(2).mean()

                loss = pg_loss + config.VF_COEF * vf_loss - config.ENTROPY_COEF * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > config.CLIP_EPS).float().mean()

                total_pg_loss   += pg_loss.item()
                total_vf_loss   += vf_loss.item()
                total_ent       += entropy.mean().item()
                total_clip_frac += clip_frac.item()
                n_updates       += 1

        denom = max(n_updates, 1)
        return {
            "pg_loss":   total_pg_loss  / denom,
            "vf_loss":   total_vf_loss  / denom,
            "entropy":   total_ent      / denom,
            "clip_frac": total_clip_frac / denom,
        }

    #########################################################################
    # Evaluation
    #########################################################################

    def evaluate(
        self,
        task_params_list: list[dict],
        n_episodes: int,
    ) -> dict:
        """
        Evaluate agent on a list of task parameter dicts.

        Runs n_tasks * n_episodes environments simultaneously with one env per
        episode per task, so each timestep is a single batched forward pass
        instead of n_tasks * n_episodes sequential passes.

        Returns:
            mean_success  — fraction of episodes where agent reached goal
            mean_return   — mean undiscounted episodic return
            per_task      — list of per-task (success_rate, mean_return)
        """
        n_tasks      = len(task_params_list)
        n_envs_total = n_tasks * n_episodes

        # One env per (task, episode) pair; env index i belongs to task i // n_episodes
        env_fns  = [
            (lambda p=params: TemplateEnv(p))
            for params in task_params_list
            for _ in range(n_episodes)
        ]
        task_of  = [i // n_episodes for i in range(n_envs_total)]

        # Hard step limit: every episode must finish within max_steps steps.
        # This guards against any edge case where done signals are never True.
        max_ep_steps = max(4 * p["grid_size"] ** 2 for p in task_params_list)
        step_limit   = max_ep_steps * 2

        envs     = gymnasium.vector.SyncVectorEnv(env_fns)
        try:
            obs, _ = envs.reset()

            ep_rets   = np.zeros(n_envs_total)
            completed = np.zeros(n_envs_total, dtype=bool)
            task_successes = [[] for _ in range(n_tasks)]
            task_returns   = [[] for _ in range(n_tasks)]

            self.model.eval()
            with torch.no_grad():
                for _ in range(step_limit):
                    if completed.all():
                        break

                    images = torch.as_tensor(
                        obs["image"], dtype=torch.uint8, device=self.device
                    )
                    dirs = torch.as_tensor(
                        obs["direction"], dtype=torch.long, device=self.device
                    )
                    actions, _, _, _ = self.model.get_action_and_value(images, dirs)

                    # Ensure actions are plain Python ints so gymnasium 1.x
                    # passes scalars (not arrays) to individual env.step() calls.
                    action_list = actions.cpu().numpy().tolist()

                    obs, rewards, terminated, truncated, _ = envs.step(action_list)
                    dones = terminated | truncated

                    for i in range(n_envs_total):
                        if completed[i]:
                            continue
                        ep_rets[i] += float(rewards[i])
                        if dones[i]:
                            tid = task_of[i]
                            task_successes[tid].append(float(ep_rets[i] > 0))
                            task_returns[tid].append(float(ep_rets[i]))
                            ep_rets[i]   = 0.0
                            completed[i] = True

                # Force-complete any envs that hit the step limit as failures
                for i in range(n_envs_total):
                    if not completed[i]:
                        tid = task_of[i]
                        task_successes[tid].append(0.0)
                        task_returns[tid].append(float(ep_rets[i]))
        finally:
            envs.close()

        per_task = [
            {
                "success_rate": float(np.mean(task_successes[t])),
                "mean_return":  float(np.mean(task_returns[t])),
            }
            for t in range(n_tasks)
        ]
        return {
            "mean_success": float(np.mean([t["success_rate"] for t in per_task])),
            "mean_return":  float(np.mean([t["mean_return"]  for t in per_task])),
            "per_task":     per_task,
        }
