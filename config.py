"""Central hyperparameter config."""

# Environment
ORIGINAL_TASKS = [
    # Single room with lava, requires deliberate navigation, not solvable by random walk
    {"grid_size": 7, "num_rooms": 1, "num_keys": 0, "num_doors": 0, "num_lava": 3, "goal_type": "reach_goal"},
    # Two rooms with locked door, requires finding key before door
    {"grid_size": 8, "num_rooms": 2, "num_keys": 1, "num_doors": 1, "num_lava": 0, "goal_type": "reach_goal"},
    # Four rooms with doors and lava, hardest, multi-room navigation under hazard
    {"grid_size": 9, "num_rooms": 4, "num_keys": 0, "num_doors": 2, "num_lava": 3, "goal_type": "reach_goal"},
]

# Parallelism
N_ENVS            = 8         # parallel environments for vectorised rollout collection

# Inner loop (agent training per generated task)
INNER_STEPS       = 4096      # env steps to train agent on each generated task
EVAL_EPISODES     = 20        # more stable reward signal for curriculum updates
AGENTS_PER_SAMPLE = 3         # number of fresh agents averaged per sampled curriculum
AGENT_LR          = 2.5e-4
GAMMA             = 0.99
GAE_LAMBDA        = 0.95
CLIP_EPS          = 0.2
ENTROPY_COEF      = 0.01
VF_COEF           = 0.5
PPO_EPOCHS        = 4
MINIBATCH_SIZE    = 256

# Outer loop (curriculum generator update) 
OUTER_ITERS       = 100       # number of generator update steps
BATCH_SIZE        = 32        # more curriculum samples per generator update
GENERATOR_LR      = 8e-3
GENERATOR_ENTROPY_COEF = 1e-3
CURRICULUM_REWARD_EMA_ALPHA = 0.25
TASK_SCORE_ALPHA  = 0.15
NEGATIVE_TASK_THRESHOLD = -0.05
NEGATIVE_TASK_REJECT_PROB = 0.8
CURRICULUM_GOAL_TYPES = ["reach_goal"]

# Reward shaping
DIFFICULTY_LAMBDA    = 0.3    # penalty weight for tasks too easy or too hard
TOO_EASY_THRESH      = 0.90   # agent success rate above this → penalize
TOO_HARD_THRESH      = 0.05   # agent success rate below this → penalize
SUBTASK_SHAPING_COEF = 0.2    # magnitude of subtask shaped reward per trigger
SUBTASK_SELECTOR_LR  = 8e-3   # learning rate for the subtask selector
SUBTASK_ENTROPY_COEF = 1e-3

# Logging
LOG_INTERVAL      = 10        # outer iterations between logs
SEED              = 43
