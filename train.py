"""
Outer training loop: curriculum generator + PPO agent.

Three runnable conditions (--mode flag):
  curriculum  : generator learns which tasks to assign  (our method)
  random      : tasks sampled uniformly from template space  (baseline)
  none        : agent trains only on original tasks  (lower bound)

Results are written to results/<mode>_<seed>.json for plotting.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config
from agent.ppo import PPOAgent
from curriculum.generator import CurriculumGenerator
from curriculum.subtask_selector import SubtaskSelector
from envs.subtask_wrapper import SubtaskWrapper
from envs.template_env import TEMPLATE_SPACE, TemplateEnv, is_valid_task


#############################################################################
# Helpers
#############################################################################

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rollout_success_rate(buf) -> float:
    """
    Estimate agent success rate on the generated task from the rollout buffer.
    A completed episode is successful if its total reward > 0.
    Uses per-episode returns tracked during vectorised collection when available.
    """
    eps = buf._episode_returns
    if eps:
        return sum(r > 0 for r in eps) / len(eps)
    # Fallback for single-env buffers
    ep_return  = 0.0
    successes  = 0
    n_episodes = 0
    for r, done in zip(buf._rewards, buf._dones):
        ep_return += r
        if done:
            if ep_return > 0:
                successes += 1
            n_episodes += 1
            ep_return   = 0.0
    return 0.0 if n_episodes == 0 else successes / n_episodes


def difficulty_penalty(success_rate: float) -> float:
    """
    Return 1.0 if the task is too easy or too hard, 0.0 otherwise.
    The Goldilocks zone is [TOO_HARD_THRESH, TOO_EASY_THRESH].
    """
    if success_rate > config.TOO_EASY_THRESH or success_rate < config.TOO_HARD_THRESH:
        return 1.0
    return 0.0


def random_task() -> dict:
    """Sample a task uniformly from the template space (random baseline)."""
    for _ in range(256):
        params = {key: random.choice(values) for key, values in TEMPLATE_SPACE.items()}
        if is_valid_task(params):
            return params
    raise RuntimeError("Failed to sample a valid random task")


def log_outer(
    it: int,
    mode: str,
    eval_result: dict,
    reward: float | None,
    gen_stats: dict | None,
    gen: CurriculumGenerator | None,
    elapsed: float,
    selector: SubtaskSelector | None = None,
    logf=None,
    console: bool = True,
) -> None:
    def w(msg: str) -> None:
        """Write to log file and optionally to stdout."""
        if console:
            print(msg, flush=True)
        if logf is not None:
            logf.write(msg + "\n")
            logf.flush()
            os.fsync(logf.fileno())

    w(
        f"[{mode}] iter {it:4d} | "
        f"success {eval_result['mean_success']:.3f} | "
        f"return {eval_result['mean_return']:.3f} | "
        + (f"gen_reward {reward:+.4f} | " if reward is not None else "")
        + (f"loss {gen_stats['loss']:+.4f} | " if gen_stats is not None else "")
        + f"t {elapsed:.1f}s"
    )
    if console and it % (config.LOG_INTERVAL * 5) == 0:
        if gen is not None:
            w("  generator distributions:")
            for k, dist in gen.get_distributions().items():
                w(f"    {k}: {dist}")
        if selector is not None:
            w(f"  subtask distribution: {selector.get_distribution()}")


#############################################################################
# Main training function
#############################################################################

def train(mode: str, seed: int, logf=None) -> list[dict]:
    """
    Run one condition to completion and return the history list.

    history is a list of dicts, one per outer iteration, containing:
      iteration, mean_success, mean_return, per_task,
      generator_reward (curriculum only), generator_loss (curriculum only)

    logf: optional open file object for explicit log writes (in addition to stdout).
    """
    subtask_shaping_enabled = getattr(train, "_subtask_shaping_enabled", True)
    difficulty_penalty_enabled = getattr(train, "_difficulty_penalty_enabled", True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def wlog(msg: str) -> None:
        print(msg, flush=True)
        if logf is not None:
            logf.write(msg + "\n")
            logf.flush()
            os.fsync(logf.fileno())

    wlog(
        f"\n=== mode={mode}  seed={seed}  device={device}  "
        f"subtask_shaping={subtask_shaping_enabled}  "
        f"difficulty_penalty={difficulty_penalty_enabled} ==="
    )

    bootstrap_agent = PPOAgent(device=device)
    init_model_state = deepcopy(bootstrap_agent.model.state_dict())

    generator    : CurriculumGenerator | None = None
    gen_optimizer: torch.optim.Optimizer | None = None
    selector     : SubtaskSelector | None = None
    sel_optimizer: torch.optim.Optimizer | None = None

    if mode == "curriculum":
        generator     = CurriculumGenerator()
        gen_optimizer = torch.optim.Adam(
            generator.parameters(), lr=config.GENERATOR_LR
        )
        selector      = SubtaskSelector()
        sel_optimizer = torch.optim.Adam(
            selector.parameters(), lr=config.SUBTASK_SELECTOR_LR
        )

    history: list[dict] = []

    #########################################################################
    # Outer loop
    #########################################################################
    t0 = time.time()
    last_console_t = t0     # throttle end-of-iter console log to ~60s
    reward_ema: float | None = None

    def fresh_agent() -> PPOAgent:
        """Return a newly initialized student agent with shared initial weights."""
        agent = PPOAgent(device=device)
        agent.model.load_state_dict(init_model_state)
        return agent

    def aggregate_eval(results: list[dict]) -> dict:
        """Average evaluation metrics across freshly initialized student runs."""
        if not results:
            return {"mean_success": 0.0, "mean_return": 0.0, "per_task": []}
        n_tasks = len(results[0]["per_task"])
        return {
            "mean_success": float(np.mean([r["mean_success"] for r in results])),
            "mean_return": float(np.mean([r["mean_return"] for r in results])),
            "per_task": [
                {
                    "success_rate": float(np.mean([r["per_task"][i]["success_rate"] for r in results])),
                    "mean_return": float(np.mean([r["per_task"][i]["mean_return"] for r in results])),
                }
                for i in range(n_tasks)
            ],
        }

    def evaluate_sample(params: dict, subtask: str, it: int, bi: int) -> tuple[list[dict], list[float], list[float]]:
        """Train multiple fresh agents on one sampled task and average their outcomes."""
        sample_evals: list[dict] = []
        task_success_rates: list[float] = []
        raw_rewards: list[float] = []
        env_fn = lambda p=params, s=subtask: SubtaskWrapper(TemplateEnv(p), s)

        for agent_idx in range(config.AGENTS_PER_SAMPLE):
            progress(it, bi, f"rollout a{agent_idx + 1}/{config.AGENTS_PER_SAMPLE}")
            agent = fresh_agent()
            rollout = agent.collect_rollout(env_fn, config.INNER_STEPS)
            agent.update(rollout)

            progress(it, bi, f"post-eval a{agent_idx + 1}/{config.AGENTS_PER_SAMPLE}")
            post_eval = agent.evaluate(config.ORIGINAL_TASKS, config.EVAL_EPISODES)
            sample_evals.append(post_eval)
            task_success_rates.append(rollout_success_rate(rollout))

            if mode in ("curriculum", "random"):
                penalty = difficulty_penalty(task_success_rates[-1])
                delta = post_eval["mean_success"] - base_eval["mean_success"]
                penalty_term = config.DIFFICULTY_LAMBDA * penalty if difficulty_penalty_enabled else 0.0
                raw_rewards.append(delta - penalty_term)

        return sample_evals, task_success_rates, raw_rewards

    def progress(it: int, bi: int, phase: str) -> None:
        """Always print a brief progress line to console and log file."""
        wlog(
            f"  [{mode}] iter {it:4d}/{config.OUTER_ITERS} "
            f"batch {bi}/{config.BATCH_SIZE} {phase} | "
            f"t {time.time() - t0:.0f}s"
        )

    for it in range(config.OUTER_ITERS):
        progress(it, 0, "pre-eval")
        base_eval = aggregate_eval([
            fresh_agent().evaluate(config.ORIGINAL_TASKS, config.EVAL_EPISODES)
            for _ in range(config.AGENTS_PER_SAMPLE)
        ])

        record: dict = {
            "iteration":                 it,
            "baseline_success":          base_eval["mean_success"],
            "baseline_return":           base_eval["mean_return"],
            "subtask_shaping_enabled":   subtask_shaping_enabled,
            "difficulty_penalty_enabled": difficulty_penalty_enabled,
        }

        # Generate and train on tasks
        batch_params:       list[dict]          = []
        batch_rewards:      list[float]         = []
        batch_log_probs:    list[torch.Tensor]  = []
        batch_sel_log_probs: list[torch.Tensor] = []
        sampled_subtasks:   list[str]           = []
        post_eval_results:  list[dict]          = []

        for bi in range(config.BATCH_SIZE):
            # Select task
            if mode == "curriculum":
                params, log_prob   = generator.sample_valid()
                if subtask_shaping_enabled:
                    subtask, sel_lp = selector.sample()
                else:
                    subtask = "none"
                    sel_lp = torch.tensor(0.0)
                batch_params.append(deepcopy(params))
                batch_log_probs.append(log_prob)
                sampled_subtasks.append(subtask)
                if subtask_shaping_enabled:
                    batch_sel_log_probs.append(sel_lp)
            elif mode == "random":
                params  = random_task()
                subtask = "none"
            else:  # none — train on a random original task
                params  = random.choice(config.ORIGINAL_TASKS)
                subtask = "none"

            sample_evals, task_success_rates, raw_rewards = evaluate_sample(params, subtask, it, bi)
            post_eval_results.extend(sample_evals)

            # Compute generator reward from rollout + next eval
            if mode in ("curriculum", "random"):
                raw_reward = float(np.mean(raw_rewards))
                if reward_ema is None:
                    reward_ema = raw_reward
                else:
                    alpha = config.CURRICULUM_REWARD_EMA_ALPHA
                    reward_ema = (1 - alpha) * reward_ema + alpha * raw_reward
                reward = reward_ema

                batch_rewards.append(reward)

        eval_result = aggregate_eval(post_eval_results)
        record["mean_success"] = eval_result["mean_success"]
        record["mean_return"] = eval_result["mean_return"]
        record["per_task"] = eval_result["per_task"]

        # Update generator + selector (curriculum mode only)
        gen_stats: dict | None = None
        if mode == "curriculum" and batch_log_probs:
            gen_stats = generator.update(
                batch_params, batch_log_probs, batch_rewards, gen_optimizer
            )
            record["generator_reward"]    = float(np.mean(batch_rewards))
            record["generator_loss"]      = gen_stats["loss"]
            record["generator_entropy"]   = gen_stats["entropy"]
            record["most_likely_task"]    = generator.most_likely_task()
            record["generator_distributions"] = generator.get_distributions()
            record["sampled_tasks"] = [deepcopy(params) for params in batch_params]
            record["sampled_task_counts"] = {
                json.dumps(task_key): count
                for task_key, count in Counter(
                    tuple(sorted(params.items())) for params in batch_params
                ).items()
            }
            if subtask_shaping_enabled and batch_sel_log_probs:
                sel_stats = selector.update(batch_sel_log_probs, batch_rewards, sel_optimizer)
                record["selector_loss"]       = sel_stats["loss"]
                record["selector_entropy"]    = sel_stats["entropy"]
                record["most_likely_subtask"] = selector.most_likely()
                record["selector_distribution"] = selector.get_distribution()
                record["sampled_subtasks"] = list(sampled_subtasks)
                record["sampled_subtask_counts"] = dict(Counter(sampled_subtasks))
            else:
                record["most_likely_subtask"] = "none"
                record["selector_distribution"] = {"none": 1.0}
                record["sampled_subtasks"] = list(sampled_subtasks)
                record["sampled_subtask_counts"] = dict(Counter(sampled_subtasks))

        history.append(record)

        # Log every iteration to file; print to console on interval
        now = time.time()
        to_console = (now - last_console_t >= 60) or (it == 0)
        if to_console:
            last_console_t = now
        log_outer(
            it, mode, eval_result,
            record.get("generator_reward"), gen_stats,
            generator, now - t0, selector,
            logf=logf, console=to_console,
        )

    #########################################################################
    # Final evaluation + summary
    #########################################################################
    final = eval_result
    wlog(
        f"\n[{mode}] FINAL  success={final['mean_success']:.3f}  "
        f"return={final['mean_return']:.3f}  "
        f"total_time={time.time()-t0:.1f}s"
    )
    if generator is not None:
        wlog("Final generator distributions:")
        for k, dist in generator.get_distributions().items():
            wlog(f"  {k}: {dist}")
    if selector is not None:
        wlog(f"Final subtask distribution: {selector.get_distribution()}")

    history.append({"iteration": config.OUTER_ITERS, **final})
    return history


#############################################################################
# Entry point
#############################################################################

def main() -> None:
    def choose_seed(out_dir: Path, explicit_seed: int | None, modes: list[str]) -> int:
        """Pick the next unused seed unless the user explicitly set one."""
        if explicit_seed is not None:
            return explicit_seed

        used = set()
        for mode in modes:
            for path in out_dir.glob(f"{mode}_*.json"):
                suffix = path.stem.rsplit("_", 1)[-1]
                if suffix.isdigit():
                    used.add(int(suffix))
        for path in out_dir.glob("train_*.log"):
            suffix = path.stem.rsplit("_", 1)[-1]
            if suffix.isdigit():
                used.add(int(suffix))

        seed = config.SEED
        while seed in used:
            seed += 1
        return seed

    parser = argparse.ArgumentParser(description="Curriculum learning experiment")
    parser.add_argument(
        "--mode",
        choices=["curriculum", "random", "none", "all"],
        default="curriculum",
        help="Training condition. 'all' runs all three sequentially.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to use. If omitted, the next unused seed in the output directory is chosen automatically.",
    )
    parser.add_argument(
        "--out", default="results", help="Directory to write result JSON files"
    )
    parser.add_argument(
        "--no-subtask-shaping",
        action="store_true",
        help="Disable subtask reward shaping and force curriculum mode to use subtask='none'.",
    )
    parser.add_argument(
        "--no-difficulty-penalty",
        action="store_true",
        help="Disable the curriculum reward difficulty penalty term.",
    )
    args = parser.parse_args()
    train._subtask_shaping_enabled = not args.no_subtask_shaping
    train._difficulty_penalty_enabled = not args.no_difficulty_penalty

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    modes = ["curriculum", "random", "none"] if args.mode == "all" else [args.mode]
    seed = choose_seed(out_dir, args.seed, modes)
    log_path = out_dir / f"train_{seed}.log"

    with open(log_path, "w", buffering=1) as logf:
        for mode in modes:
            history  = train(mode=mode, seed=seed, logf=logf)
            out_path = out_dir / f"{mode}_{seed}.json"
            with open(out_path, "w") as f:
                json.dump(history, f, indent=2)
            msg = f"Results written to {out_path}"
            print(msg, flush=True)
            logf.write(msg + "\n")
            logf.flush()
            os.fsync(logf.fileno())


if __name__ == "__main__":
    main()
