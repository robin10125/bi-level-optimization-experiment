# Run Requirements

This file describes what another machine needs in order to run my experiment.

## Supported setup

- OS: Linux
- Python: `3.12` 
- GPU: not required; my experiments is designed to run on CPU

## Python packages

These are the package versions currently used in my `.venv`:

```text
torch==2.11.0
gymnasium==1.2.3
minigrid==3.0.0
numpy==2.4.4
matplotlib==3.10.8
pygame==2.6.1
networkx==3.6.1
```

## Minimal setup on a new machine

From the project root:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install torch==2.11.0 gymnasium==1.2.3 minigrid==3.0.0 numpy==2.4.4 matplotlib==3.10.8 pygame==2.6.1 networkx==3.6.1
```

## Files that need to be copied

If the goal is only to run the main context-free curriculum experiment, copy:

- `train.py`
- `config.py`
- `train`
- `curriculum.sh`
- `agent/`
- `curriculum/`
- `envs/`
- `path/`


## Expected project layout

The shell wrappers assume this layout:

```text
project-root/
  .venv/
  train
  curriculum.sh
  train.py
  config.py
  agent/
  curriculum/
  envs/
  path/
  reward_generator_experiment/
```

The scripts use `project-root/.venv/bin/python` directly. If the virtual
environment lives somewhere else, either:

- recreate it at `.venv`, or
- edit `train`, `curriculum.sh`

## Commands to run

Main full experiment:

```bash
./train
```

Curriculum-only run:

```bash
./curriculum.sh
```

Reward-generator experiment:

```bash
./run_reward_generator
```

## Useful command-line options

Manual seed:

```bash
./train --seed 44
./run_reward_generator --seed 11
```

Curriculum ablations:

```bash
./train --no-subtask-shaping
./train --no-difficulty-penalty
./train --no-subtask-shaping --no-difficulty-penalty
```

## Outputs

Main experiment outputs are written to:

- `results/curriculum_<seed>.json`
- `results/random_<seed>.json`
- `results/none_<seed>.json`
- `results/train_<seed>.log`

