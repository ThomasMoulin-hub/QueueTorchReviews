# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QueueTorch is a PyTorch-based package for **differentiable queuing network control**. It provides discrete-event simulation of queuing networks where RL policies can be learned via direct backpropagation (pathwise/straight-through estimator) or PPO. The core idea is that the simulation is differentiable end-to-end, enabling gradient-based optimization of scheduling policies.

## Build & Install

```bash
python3 -m build
python3 -m pip install .
```

Requires Python >=3.9, <3.13. Key pinned dependencies: `torch==2.2.0`, `numpy==1.26.4`, `scipy==1.11.4`, `cvxpy==1.4.2`.

## Running Tests

```bash
python3 tests/mms.py
```

Tests validate the simulator against known closed-form results (M/M/S queues, priority queues) by checking that simulated average costs match analytical solutions within tolerance.

## Training a Policy

Training must be run from the `train/` directory (relative paths to configs):

```bash
cd train
python3 ./train_policy.py -e=ENV.yaml -m=MODEL.yaml [--algo ste|ppo]
```

- `-e`: environment config from `configs/env/`
- `-m`: model config from `configs/model/`
- `-d`: optional device override
- `--algo`: `ste` (default, straight-through estimator / pathwise) or `ppo`

Checkpoints save to `train/models/`, loss logs to `train/loss/`, plots to `train/plot/`.

## PPO Training (Stable Baselines variant)

A separate PPO implementation using Stable Baselines 3 lives in `PPO/`:

```bash
cd PPO
python3 train.py <config_name> <env_config_name>
```

This requires `stable_baselines3` and `gym` (not listed in pyproject.toml dependencies).

## Architecture

### Core Library (`queuetorch/`)

- **`env.py`** — The central module. `QueuingNetwork` is the differentiable discrete-event simulator. State is `EnvState(queues, time, service_times, arrival_times)`, observations are `Obs(queues, time)`. The `step()` method uses a straight-through estimator (hard argmin forward, softmax backward via `temp` parameter) to make event selection differentiable. `load_env()` constructs a `QueuingNetwork` from a YAML config dict.
- **`routing.py`** — Sinkhorn-based differentiable optimal transport for server-queue assignment. Contains `Sinkhorn` (custom autograd function with analytical backward pass), `linear_assignment_batch` (scipy LP solver), and `pad` (prepares assignment matrices).
- **`policies.py`** — Simple policy architectures: `SoftPriorityPolicy` (state-independent softmax), `SoftMaxWeightPolicy` (linear in queue lengths), `SoftMaxPressurePolicy`.
- **`ppo.py`** — PPO buffer and loss computation (GAE, clipped surrogate).

### Training (`train/train_policy.py`)

`PriorityNet` is the main neural policy (MLP mapping queue state to server-queue softmax priorities). `CriticNet` is the value function for PPO mode. Both support optional time input (`f_time`). The STE training loop runs the full simulation forward, accumulates cost, then backpropagates through the entire trajectory.

### Configs

- **`configs/env/`** — YAML files defining queuing network topology (`network`, `mu`, `h`), arrival process (`lam_type`: constant/step/sawtooth/hyper), service distribution, and simulation horizons. See `template_env.yaml` for documentation.
- **`configs/model/`** — YAML files for training hyperparameters (optimizer, architecture, policy type). See `ppg_softmax.yaml` for an example.
- **`env_data/`** — NumPy arrays (`.npy`) for network/mu/lambda of predefined environments (reentrant lines). Referenced by config when `network`/`mu` are `null`.

### Experiments (`experiments/`)

Standalone scripts for specific studies (CMU rule comparisons via pathwise vs REINFORCE, admission control, gradient analysis). These use `pathos.multiprocessing` for parallel trials and write results as JSON to `cmu/`.

### Key Concepts

- **Network matrix** (`s x q`): binary compatibility between servers and queues
- **Actions**: softmax distributions over queues per server, representing scheduling priorities
- **Temperature (`temp`)**: controls the sharpness of the straight-through softmax in event selection (lower = closer to hard argmin)
- **Work-conserving policy**: actions are clipped so servers only serve non-empty queues and respect the network structure
- **Batch simulation**: all environments vectorize across a batch dimension for parallel sample paths
