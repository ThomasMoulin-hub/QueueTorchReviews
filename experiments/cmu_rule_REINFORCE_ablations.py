import torch
import numpy as np
import yaml
import os
import json
from collections import defaultdict
import sys
import copy
import pathos.multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cmu_rule_REINFORCE import reinforce_value_cmu

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CMU_DIR = os.path.join(PROJECT_ROOT, 'cmu')
SEEDS_FILE = os.path.join(CMU_DIR, 'seeds_cmu_10class.json')

# ── Baseline parameters ──
BASELINE = {
    'num_iter': 20,      # gradient steps
    'rho': 0.95,         # traffic intensity
    'T': 1000,           # horizon (events per gradient step)
    'queue_class': 10,   # number of job classes
}

# ── REINFORCE-specific defaults (matching original script) ──
GAMMA = 0.99
REINFORCE_BATCH = 100

# ── Ablation axes (one varies, rest at baseline) ──
ABLATIONS = {
    'num_iter':    [10, 20, 50, 100],
    'rho':         [0.9, 0.95, 0.99],
    'T':           [500, 1000, 2000, 5000],
    'queue_class': [5, 10, 15, 20],
}

# ── Sweep grid (same as original experiments) ──
ALPHAS = [0.01, 0.1, 0.5, 1.0]
GAPS = [1, 0.5, 0.05, 0.01]

# ── Scale ──
NUM_CORES = 40
NUM_TRIALS = 100
EVAL_T = 20000


def build_env_config(queue_class, rho, gap):
    """Build a multiclass env_config dict for the given parameters."""
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


def _run_reinforce(kwargs):
    return reinforce_value_cmu(**kwargs)


def run_sweep(seeds, num_iter, rho, T, queue_class, tag):
    """Run reinforce_value_cmu over all (alpha, gap) pairs in a single pool submission."""

    # Flatten all (alpha, gap, trial) combinations into one job list
    all_jobs = []
    job_keys = []  # track (alpha_str, gap_str) for each job

    for alpha in ALPHAS:
        for gap in GAPS:
            env_config = build_env_config(queue_class, rho, gap)
            alpha_str, gap_str = str(alpha), str(gap)

            for i in range(NUM_TRIALS):
                all_jobs.append({
                    'env_config': copy.deepcopy(env_config),
                    'seed': seeds[i],
                    'num_iter': num_iter,
                    'alpha': alpha,
                    'T': T,
                    'gamma': GAMMA,
                    'batch': REINFORCE_BATCH,
                    'eval_T': EVAL_T,
                })
                job_keys.append((alpha_str, gap_str))

    total = len(all_jobs)
    print(f'  [{tag}] submitting {total} jobs '
          f'(num_iter={num_iter}, rho={rho}, T={T}, classes={queue_class})')

    with mp.ProcessingPool(NUM_CORES) as pool:
        all_outputs = pool.map(_run_reinforce, all_jobs)

    # Partition results by (alpha, gap)
    results = defaultdict(lambda: defaultdict(list))
    for (alpha_str, gap_str), result in zip(job_keys, all_outputs):
        results[alpha_str][gap_str].append(result)

    out_path = os.path.join(CMU_DIR, f'reinforce_ablation_{tag}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f)

    return dict(results)


def run_ablation_axis(seeds, axis_name, axis_values, cache):
    """Run all values for one ablation axis, skipping already-cached baseline runs."""

    print(f'=== Ablation: {axis_name} ===')
    for val in axis_values:
        params = dict(BASELINE)
        params[axis_name] = val
        tag = f'{axis_name}_{val}'

        # Skip if this exact setting was already run in a previous axis
        cache_key = (params['num_iter'], params['rho'],
                     params['T'], params['queue_class'])
        if cache_key in cache:
            prev_tag = cache[cache_key]
            prev_path = os.path.join(CMU_DIR, f'reinforce_ablation_{prev_tag}.json')
            cur_path = os.path.join(CMU_DIR, f'reinforce_ablation_{tag}.json')
            with open(prev_path, 'r') as f:
                data = json.load(f)
            with open(cur_path, 'w') as f:
                json.dump(data, f)
            print(f'  [{tag}] reusing results from [{prev_tag}]')
            continue

        run_sweep(seeds, tag=tag, **params)
        cache[cache_key] = tag


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

    # Cache tracks (num_iter, rho, T, queue_class) -> tag to avoid re-running
    # identical parameter combinations across ablation axes.
    cache = {}

    for axis_name in ['num_iter', 'rho', 'T', 'queue_class']:
        run_ablation_axis(seeds, axis_name, ABLATIONS[axis_name], cache)
