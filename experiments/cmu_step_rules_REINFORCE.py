import os
# Set thread limits BEFORE importing torch/numpy to prevent oversubscription
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

import numpy as np
import yaml
import json
import time
from torch import nn
from collections import defaultdict
import sys
import copy
import pathos.multiprocessing as mp
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from queuetorch.env import load_env
from step_rules import make_step_rule, STEP_RULES, STEP_RULE_ALPHAS
import torch.distributions.one_hot_categorical as one_hot_sample

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CMU_DIR = os.path.join(PROJECT_ROOT, 'cmu')
SEEDS_FILE = os.path.join(CMU_DIR, 'seeds_cmu_10class.json')

# Fixed experiment parameters
GAPS = [1, 0.5, 0.05, 0.01]
NUM_CORES = min(80, os.cpu_count() or 40)
NUM_TRIALS = 100
TRIAL_OFFSET = 0
EVAL_T = 20000
NUM_ITER = 100
RHO = 0.95
QUEUE_CLASS = 10
T = 1000
GAMMA = 0.99
REINFORCE_BATCH = 100

# Only run these rules (subset of STEP_RULES)
RUN_RULES = {
    'normalized_fixed': STEP_RULES['normalized_fixed'],
    'adam':             STEP_RULES['adam'],
    'rmsprop':         STEP_RULES['rmsprop'],
}
RUN_ALPHAS = {
    'normalized_fixed': STEP_RULE_ALPHAS['normalized_fixed'],
    'adam':             STEP_RULE_ALPHAS['adam'],
    'rmsprop':         STEP_RULE_ALPHAS['rmsprop'],
}

# Output filename tag to avoid overwriting K=20 results
FILE_TAG = f'K{NUM_ITER}'


class ValueNet(nn.Module):
    def __init__(self, q, layers, hidden_dim, x_stats=None, y_stats=None):
        super().__init__()
        self.q = q
        self.x_stats = x_stats
        self.y_stats = y_stats
        self.layers = layers
        self.hidden_dim = hidden_dim

        self.input_fc = nn.Linear(self.q, hidden_dim)

        self.layers_fc = nn.ModuleList()
        for _ in range(layers):
            self.layers_fc.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if self.x_stats is not None:
            x = (x - self.x_stats[0]) / self.x_stats[1]

        x = F.relu(self.input_fc(x))

        for l in range(self.layers):
            x = F.relu(self.layers_fc[l](x))

        x = self.output_fc(x)
        return x


def evaluate_iterate_fast(priority, env_config, batch=100, eval_T=10000):
    """Evaluation with no_grad for speed - no gradient tracking needed."""
    torch.set_num_threads(1)
    dq = load_env(env_config, temp=0.5, batch=batch, seed=42, device='cpu')
    torch.manual_seed(42)
    obs, state = dq.reset(seed=42)

    total_cost = torch.tensor([[0.]] * batch)

    with torch.no_grad():
        for _ in range(eval_T):
            queues, time = obs

            pr = F.softmax(priority.repeat(dq.batch, dq.s, 1), -1)
            pr = F.one_hot(torch.argmax(pr * 1. * (queues > 0.).unsqueeze(1), dim=2), num_classes=dq.q)
            pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
            pr += 1 * torch.all(pr == 0., dim=2).reshape(dq.batch, dq.s, 1).repeat(1, 1, dq.q) * dq.network
            pr /= torch.sum(pr, dim=-1).reshape(dq.batch, dq.s, 1)

            action = one_hot_sample.OneHotCategorical(probs=pr).sample()
            obs, state, cost, event_time = dq.step(state, action)
            total_cost += cost

    return float(torch.mean(total_cost / state.time))


def calculate_returns(rewards, discount_factor, normalize=True):
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)

    cat_returns = torch.cat(returns)

    if normalize:
        for i in range(len(returns)):
            returns[i] = (returns[i] - cat_returns.mean()) / cat_returns.std()

    return returns


def reinforce_value_cmu_step_rule(env_config, seed, num_iter, step_rule_name, alpha,
                                  T=1000, gamma=0.99, batch=1000, eval_T=10000):
    """reinforce_value_cmu with pluggable step rule instead of fixed normalized SGD."""
    torch.set_num_threads(1)

    step_rule = make_step_rule(step_rule_name, alpha)

    dq = load_env(env_config, temp=1.0, batch=1, seed=seed, device='cpu')
    priority = torch.zeros((1, dq.q)).float()
    priority.requires_grad = True

    sum_priority = priority.detach().clone()
    reinforce_avg_iterate = [sum_priority.clone()]
    num = 1

    for i in range(num_iter):
        dq = load_env(env_config, temp=1.0, batch=batch, seed=seed, device='cpu')

        if i > 0:
            obs, state = dq.reset(seed=seed, init_queues=init_queues)
        else:
            obs, state = dq.reset(seed=seed)

        torch.manual_seed(8838383 + i)

        log_prob_buffer = []
        costs = []
        state_buffer = []

        for t_step in range(T):
            queues, time = obs

            pr = F.softmax(priority.repeat(dq.batch, dq.s, 1), -1) * dq.network
            pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
            pr += 1 * torch.all(pr == 0., dim=2).reshape(dq.batch, dq.s, 1) * dq.network
            pr /= torch.sum(pr, dim=-1).reshape(dq.batch, dq.s, 1)

            pr_sample = one_hot_sample.OneHotCategorical(probs=pr)
            action = pr_sample.sample()

            log_prob = torch.sum(torch.log(torch.sum(action.detach() * pr, dim=2)), dim=1, keepdims=True)
            log_prob_buffer.append(log_prob)

            obs, state, cost, event_time = dq.step(state, action)
            costs.append(cost.detach().tolist())
            state_buffer.append(torch.hstack((queues, t_step * torch.ones(dq.batch).unsqueeze(1))))

        init_queues = queues.detach()

        cost_buffer = torch.tensor(costs)
        returns = calculate_returns(cost_buffer, gamma)

        all_states = torch.cat(state_buffer, axis=0)
        all_returns = torch.cat(returns)

        if i == 0:
            state_mean = all_states.mean(0)
            state_std = all_states.std(0)

            value_net = ValueNet(dq.q + 1, 2, 64)
            value_net.x_stats = [state_mean, state_std]

            adam = torch.optim.Adam(value_net.parameters(), lr=0.001)

        return_dataset = torch.utils.data.TensorDataset(all_states, all_returns)
        return_dataloader = torch.utils.data.DataLoader(return_dataset, batch_size=1024, shuffle=True)

        for epoch in range(3):
            for count, (state_batch, return_batch) in enumerate(return_dataloader):
                adam.zero_grad()
                out = value_net(state_batch)
                value_loss = F.mse_loss(out, return_batch, reduction='sum')
                value_loss.backward()
                adam.step()

        policy_loss = torch.tensor(0.)
        for t_step in range(len(returns)):
            s = state_buffer[t_step]
            policy_loss = policy_loss + (returns[t_step] - value_net(s).detach()) * log_prob_buffer[t_step]

        torch.mean(policy_loss).backward()

        # Apply step rule instead of fixed normalized update
        update = step_rule.step(priority.grad, i)
        priority = priority.detach() - update

        sum_priority += priority.detach()
        num += 1
        reinforce_avg_iterate.append(sum_priority.clone() / num)

        priority.requires_grad = True

    avg_cost = evaluate_iterate_fast(reinforce_avg_iterate[-1], env_config, eval_T=eval_T)

    return {'last_iterate': reinforce_avg_iterate[-1].detach().tolist(),
            'avg_cost': avg_cost}


def build_env_config(queue_class, rho, gap):
    with open(os.path.join(PROJECT_ROOT, 'configs/env/multiclass.yaml'), 'r') as f:
        env_config = yaml.safe_load(f)

    env_config['init_queues'] = [0] * queue_class
    env_config['network'] = [[1] * queue_class]
    env_config['queue_event_options'] = np.vstack(
        (np.eye(queue_class), -np.eye(queue_class))
    ).tolist()
    env_config['h'] = [1] * queue_class

    mu = np.array([[1 + gap * i for i in range(1, queue_class + 1)]])
    env_config['mu'] = mu
    env_config['lam_params']['val'] = np.repeat(
        rho / np.sum(1 / mu), queue_class
    ).tolist()

    return env_config


def _run_job(kwargs):
    torch.set_num_threads(1)
    return reinforce_value_cmu_step_rule(**kwargs)


def run_all_step_rules(seeds):
    """Run step rules with per-combo progress reporting and incremental file saves."""

    # Build list of (rule_name, alpha, gap) combos
    combos = []
    for rule_name in RUN_RULES:
        alphas = RUN_ALPHAS[rule_name]
        for alpha in alphas:
            for gap in GAPS:
                combos.append((rule_name, alpha, gap))

    total_combos = len(combos)
    total_jobs = total_combos * NUM_TRIALS

    print(f"{'='*60}", flush=True)
    print(f"Step Rules Experiment (REINFORCE, {FILE_TAG})", flush=True)
    print(f"  Rules: {list(RUN_RULES.keys())}", flush=True)
    print(f"  {total_combos} combos x {NUM_TRIALS} trials = {total_jobs} total jobs", flush=True)
    print(f"  Using {NUM_CORES} cores", flush=True)
    print(f"{'='*60}", flush=True)

    per_rule = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    start_time = time.time()
    completed_combos = 0

    with mp.ProcessingPool(NUM_CORES) as pool:
        for rule_name, alpha, gap in combos:
            env_config = build_env_config(QUEUE_CLASS, RHO, gap)
            alpha_str, gap_str = str(alpha), str(gap)

            jobs = []
            for i in range(TRIAL_OFFSET, TRIAL_OFFSET + NUM_TRIALS):
                jobs.append({
                    'env_config': copy.deepcopy(env_config),
                    'seed': seeds[i],
                    'num_iter': NUM_ITER,
                    'step_rule_name': rule_name,
                    'alpha': alpha,
                    'T': T,
                    'gamma': GAMMA,
                    'batch': REINFORCE_BATCH,
                    'eval_T': EVAL_T,
                })

            combo_start = time.time()
            results = pool.map(_run_job, jobs)
            combo_elapsed = time.time() - combo_start

            per_rule[rule_name][alpha_str][gap_str] = results
            completed_combos += 1

            # Save incremental results for this rule
            out_path = os.path.join(CMU_DIR, f'reinforce_step_rule_{rule_name}_{FILE_TAG}.json')
            with open(out_path, 'w') as f:
                json.dump(per_rule[rule_name], f)

            # Progress report
            total_elapsed = time.time() - start_time
            avg_per_combo = total_elapsed / completed_combos
            remaining_combos = total_combos - completed_combos
            eta = avg_per_combo * remaining_combos

            print(f"[{completed_combos}/{total_combos}] "
                  f"rule={rule_name} alpha={alpha} gap={gap} "
                  f"| combo: {combo_elapsed:.1f}s "
                  f"| total: {total_elapsed/60:.1f}min "
                  f"| ETA: {eta/60:.1f}min "
                  f"| saved -> {out_path}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"All done in {(time.time() - start_time)/60:.1f} minutes", flush=True)
    print(f"{'='*60}", flush=True)


def load_seeds():
    if os.path.exists(SEEDS_FILE):
        with open(SEEDS_FILE, 'r') as f:
            seeds = json.load(f)
    else:
        seeds = [int.from_bytes(os.urandom(4), 'big') for _ in range(10000)]
        with open(SEEDS_FILE, 'w') as f:
            json.dump(seeds, f)
    return seeds


if __name__ == '__main__':

    seeds = load_seeds()
    run_all_step_rules(seeds)
