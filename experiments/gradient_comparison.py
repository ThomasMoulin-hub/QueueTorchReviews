#!/user/tmm2219/.conda/envs/qt_env/bin/python
"""
Gradient Comparison Experiment.

This script compares two gradient estimators for policy gradient methods in queuing networks:
1. **Pathwise (STE) Estimator**: Uses the Straight-Through Estimator with batch size B=1.
2. **REINFORCE Estimator**: Uses Monte Carlo sampling with batch size B=1000.

Both estimators are compared against a ground truth gradient computed using REINFORCE
with a very large batch size (default: 1,000,000 samples).

The comparison metric is cosine similarity between the estimated gradient and ground truth.

Usage:
    python gradient_comparison.py --env criss_cross_bh.yaml --horizon 1000 --gt_batch 1000000
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from copy import deepcopy
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions.one_hot_categorical as one_hot_sample
import yaml
from tqdm import tqdm

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from queuetorch.env import load_env
from queuetorch.policies import SoftPriorityPolicy, SoftMaxWeightPolicy, SoftMaxPressurePolicy

# =============================================================================
# Constants
# =============================================================================

POLICY_TYPES = ['sPR', 'sMW', 'sMP']
DEFAULT_GAMMA = 0.999
DEFAULT_REINFORCE_BATCH = 1000

# =============================================================================
# Policy Factory
# =============================================================================


def get_policy(policy_type: str, s: int, q: int) -> torch.nn.Module:
    """
    Factory function to create a policy network.

    Args:
        policy_type: Type of policy ('sPR', 'sMW', or 'sMP').
        s: Number of servers.
        q: Number of queues.

    Returns:
        Instantiated policy network.

    Raises:
        ValueError: If policy_type is not recognized.
    """
    policy_map = {
        'sPR': SoftPriorityPolicy,
        'sMW': SoftMaxWeightPolicy,
        'sMP': SoftMaxPressurePolicy,
    }

    if policy_type not in policy_map:
        raise ValueError(f"Unknown policy type: {policy_type}. Valid types: {list(policy_map.keys())}")

    return policy_map[policy_type](s, q)


# =============================================================================
# Gradient Estimators
# =============================================================================


def _compute_reinforce_grad_core(
    net: torch.nn.Module,
    env_config: Dict[str, Any],
    batch_size: int,
    T: int,
    gamma: float,
    device: str
) -> torch.Tensor:
    """
    Compute REINFORCE gradient estimator for a single batch.

    This function implements the standard REINFORCE algorithm (Williams, 1992)
    for computing policy gradients using Monte Carlo returns.

    Args:
        net: Policy network with parameters theta.
        env_config: Environment configuration dictionary.
        batch_size: Number of parallel trajectories to sample.
        T: Trajectory horizon (number of steps).
        gamma: Discount factor for computing returns.
        device: Computation device ('cpu' or 'cuda').

    Returns:
        Flattened gradient tensor over all policy parameters.
    """
    net.zero_grad()

    # Initialize environment
    dq = load_env(env_config, temp=1.0, batch=batch_size, seed=None, device=device)
    obs, state = dq.reset()

    log_probs: List[torch.Tensor] = []
    rewards: List[torch.Tensor] = []

    # Collect trajectories
    for _ in range(T):
        queues, time = obs
        probs = net(queues)
        
        # Apply network structure constraints
        probs = probs * dq.network
        probs = torch.minimum(probs, queues.unsqueeze(1).repeat(1, dq.s, 1))

        # Fallback for zero probability rows
        all_zero_mask = torch.all(probs == 0., dim=2).reshape(batch_size, dq.s, 1)
        probs = probs + all_zero_mask.repeat(1, 1, dq.q) * dq.network

        # Normalize probabilities
        sum_probs = torch.sum(probs, dim=-1, keepdim=True)
        sum_probs = torch.where(sum_probs == 0, torch.ones_like(sum_probs), sum_probs)
        probs = probs / sum_probs

        # Sample action and compute log probability
        dist = one_hot_sample.OneHotCategorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1)  # Sum over servers

        log_probs.append(log_prob)
        
        obs, state, cost, event_time = dq.step(state, action)
        rewards.append(-cost.squeeze(1))  # Reward = -Cost

    # Compute discounted returns and policy loss
    policy_loss = torch.zeros(1, device=device)
    returns = torch.zeros(batch_size, device=device)

    for t in reversed(range(T)):
        returns = rewards[t] + gamma * returns
        policy_loss = policy_loss - (log_probs[t] * returns).mean()

    # Backpropagate
    policy_loss.backward()

    # Collect gradients
    grads = []
    for param in net.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1).detach().cpu())
        else:
            grads.append(torch.zeros_like(param.view(-1)).cpu())

    return torch.cat(grads)


def compute_pathwise_grad(
    net: torch.nn.Module,
    env_config: Dict[str, Any],
    batch_size: int,
    T: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Compute the Pathwise (Straight-Through Estimator) gradient.

    This estimator uses the reparameterization trick by directly passing
    probabilities as soft actions, allowing gradients to flow through
    the environment dynamics.

    Args:
        net: Policy network with parameters theta.
        env_config: Environment configuration dictionary.
        batch_size: Number of parallel trajectories.
        T: Trajectory horizon (number of steps).
        device: Computation device ('cpu' or 'cuda').

    Returns:
        Flattened gradient tensor over all policy parameters.
    """
    net.zero_grad()

    dq = load_env(env_config, temp=1.0, batch=batch_size, seed=None, device=device)
    obs, state = dq.reset()

    total_cost = torch.zeros(1, device=device)

    for _ in range(T):
        queues, time = obs
        probs = net(queues)
        
        # Apply network structure constraints
        probs = probs * dq.network
        probs = torch.minimum(probs, queues.unsqueeze(1).repeat(1, dq.s, 1))

        # Fallback for zero probability rows
        all_zero_mask = torch.all(probs == 0., dim=2).reshape(batch_size, dq.s, 1)
        probs = probs + all_zero_mask.repeat(1, 1, dq.q) * dq.network

        # Normalize probabilities
        sum_probs = torch.sum(probs, dim=-1, keepdim=True)
        sum_probs = torch.where(sum_probs == 0, torch.ones_like(sum_probs), sum_probs)
        probs = probs / sum_probs

        # Use soft action (STE)
        action = probs

        obs, state, cost, event_time = dq.step(state, action)
        total_cost = total_cost + cost.mean()

    # Compute average cost per step
    loss = total_cost / T
    loss.backward()
    
    # Collect gradients
    grads = []
    for param in net.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1).detach().cpu())
        else:
            grads.append(torch.zeros_like(param.view(-1)).cpu())

    return torch.cat(grads)


def cosine_similarity(g1: torch.Tensor, g2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two gradient vectors.

    Args:
        g1: First gradient vector.
        g2: Second gradient vector.

    Returns:
        Cosine similarity value in [-1, 1], or 0.0 if either vector has zero norm.
    """
    norm1 = torch.norm(g1)
    norm2 = torch.norm(g2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()


# =============================================================================
# Multiprocessing Worker Functions
# =============================================================================


def gt_worker(args: Tuple) -> torch.Tensor:
    """
    Worker function for computing ground truth gradient in parallel.

    Args:
        args: Tuple containing (policy_type, s, q, state_dict, env_config,
              batch_size, T, gamma).

    Returns:
        Computed gradient tensor for this batch.
    """
    policy_type, s, q, state_dict, env_config, batch_size, T, gamma = args
    
    net = get_policy(policy_type, s, q)
    net.load_state_dict(state_dict)
    net.to('cpu')

    grad = _compute_reinforce_grad_core(net, env_config, batch_size, T, gamma, device='cpu')
    return grad


def estimator_worker(args: Tuple) -> Tuple[float, float]:
    """
    Worker function for computing both Pathwise and REINFORCE estimators.

    Args:
        args: Tuple containing (policy_type, s, q, state_dict, env_config, T, gt_grad).

    Returns:
        Tuple of (pathwise_similarity, reinforce_similarity).
    """
    policy_type, s, q, state_dict, env_config, T, gt_grad = args
    
    # Reconstruct policy
    net = get_policy(policy_type, s, q)
    net.load_state_dict(state_dict)
    net.to('cpu')
    
    # Pathwise Estimator (B=1)
    pw_grad = compute_pathwise_grad(net, env_config, batch_size=1, T=T, device='cpu')

    # REINFORCE Estimator (B=1000)
    rf_grad = _compute_reinforce_grad_core(
        net, env_config, batch_size=DEFAULT_REINFORCE_BATCH, T=T, gamma=DEFAULT_GAMMA, device='cpu'
    )

    # Calculate similarities
    sim_pw = cosine_similarity(pw_grad, gt_grad)
    sim_rf = cosine_similarity(rf_grad, gt_grad)

    return sim_pw, sim_rf


# =============================================================================
# Main Computation Functions
# =============================================================================


def compute_gt_multiprocessing(
    net: torch.nn.Module,
    env_config: Dict[str, Any],
    total_batch: int,
    T: int,
    gamma: float = DEFAULT_GAMMA,
    policy_type: str = 'sPR',
    num_cores: int = None
) -> torch.Tensor:
    """
    Compute ground truth gradient using multiprocessing.

    Distributes the total batch across multiple CPU cores and aggregates
    the gradient estimates.

    Args:
        net: Policy network.
        env_config: Environment configuration dictionary.
        total_batch: Total number of trajectories for ground truth estimation.
        T: Trajectory horizon.
        gamma: Discount factor.
        policy_type: Type of policy being used.
        num_cores: Number of CPU cores to use. Defaults to min(cpu_count, 100).

    Returns:
        Weighted average gradient tensor.
    """
    if num_cores is None:
        num_cores = min(mp.cpu_count(), 100)

    chunk_size = total_batch // num_cores
    remainder = total_batch % num_cores
    
    state_dict = deepcopy(net.state_dict())
    s, q = net.s, net.q
    
    # Create tasks for each worker
    tasks = []
    for i in range(num_cores):
        current_batch = chunk_size + (1 if i < remainder else 0)
        if current_batch > 0:
            tasks.append((policy_type, s, q, state_dict, env_config, current_batch, T, gamma))

    # Run parallel computation
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(gt_worker, tasks)

    # Aggregate results (weighted average)
    total_grad = torch.zeros_like(results[0])
    total_samples = 0

    for i, grad in enumerate(results):
        batch_n = tasks[i][5]
        total_grad += grad * batch_n
        total_samples += batch_n

    return total_grad / total_samples


def compute_estimators_multiprocessing(
    net: torch.nn.Module,
    env_config: Dict[str, Any],
    T: int,
    gt_grad: torch.Tensor,
    policy_type: str,
    num_estimators: int,
    num_cores: int = None
) -> Tuple[List[float], List[float]]:
    """
    Compute multiple gradient estimators in parallel and return similarities to ground truth.

    Args:
        net: Policy network.
        env_config: Environment configuration dictionary.
        T: Trajectory horizon.
        gt_grad: Ground truth gradient for comparison.
        policy_type: Type of policy being used.
        num_estimators: Number of estimator samples to compute.
        num_cores: Number of CPU cores to use.

    Returns:
        Tuple of (pathwise_similarities, reinforce_similarities) lists.
    """
    if num_cores is None:
        num_cores = min(mp.cpu_count(), 100)

    state_dict = deepcopy(net.state_dict())
    s, q = net.s, net.q

    # Create tasks
    tasks = [
        (policy_type, s, q, state_dict, env_config, T, gt_grad)
        for _ in range(num_estimators)
    ]

    # Run parallel computation
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(estimator_worker, tasks)

    # Unpack results
    sim_pw_list = [r[0] for r in results]
    sim_rf_list = [r[1] for r in results]

    return sim_pw_list, sim_rf_list


# =============================================================================
# Experiment Runner
# =============================================================================


def run_experiment(args: argparse.Namespace) -> None:
    """
    Run the gradient comparison experiment.

    For each traffic intensity (rho) and policy type, this function:
    1. Samples random policy parameters
    2. Computes ground truth gradient using large-batch REINFORCE
    3. Computes Pathwise and REINFORCE estimators with small batches
    4. Measures cosine similarity between estimators and ground truth

    Args:
        args: Command-line arguments containing experiment configuration.
    """
    device = 'cpu'
    print(f"Running on {device} with {mp.cpu_count()} cores available.")

    # Traffic intensities to test
    intensities = [0.8, 0.9, 0.95, 0.99]

    # Resolve config path
    base_config_path = f'./configs/env/{args.env}'
    if not os.path.exists(base_config_path):
        base_config_path = f'../configs/env/{args.env}'
    
    for rho in intensities:
        print(f"\n{'=' * 50}")
        print(f"Starting Experiment for Intensity rho = {rho}")
        print(f"{'=' * 50}")

        # Load fresh config for each intensity
        with open(base_config_path, 'r') as f:
            env_config = yaml.safe_load(f)

        # Scale arrival rates by traffic intensity
        if env_config['lam_type'] == 'constant':
            if 'val' in env_config['lam_params'] and env_config['lam_params']['val'] is not None:
                original_vals = np.array(env_config['lam_params']['val'])
                scaled_vals = original_vals * rho
                env_config['lam_params']['val'] = scaled_vals.tolist()
        else:
            print(f"Warning: lam_type is {env_config['lam_type']}, scaling might not be applied correctly.")

        results = []

        for policy_name in POLICY_TYPES:
            print(f"\nTesting Policy: {policy_name} (rho={rho})")

            for sample_idx in tqdm(range(args.num_samples), desc=f"Samples {policy_name}"):
                # Initialize policy with random weights
                temp_dq = load_env(env_config, temp=1, batch=1, seed=None, device='cpu')
                s, q = temp_dq.s, temp_dq.q
                net = get_policy(policy_name, s, q).to(device)

                # Compute ground truth gradient
                print(f"  [Sample {sample_idx}] Computing Ground Truth (Batch={args.gt_batch})...")
                gt_grad = compute_gt_multiprocessing(
                    net, env_config,
                    total_batch=args.gt_batch,
                    T=args.horizon,
                    policy_type=policy_name,
                    num_cores=args.num_cores
                )
                gt_norm = torch.norm(gt_grad).item()
                print(f"  [Sample {sample_idx}] GT Computed. Norm: {gt_norm:.4f}")

                # Compute estimator similarities
                print(f"  [Sample {sample_idx}] Estimating Similarity ({args.estimators_per_sample} estimators)...")
                sim_pw_list, sim_rf_list = compute_estimators_multiprocessing(
                    net, env_config,
                    T=args.horizon,
                    gt_grad=gt_grad,
                    policy_type=policy_name,
                    num_estimators=args.estimators_per_sample,
                    num_cores=args.num_cores
                )
                
                avg_sim_pw = np.mean(sim_pw_list)
                avg_sim_rf = np.mean(sim_rf_list)
                print(f"  [Sample {sample_idx}] Done. Avg PW Sim: {avg_sim_pw:.4f}, Avg RF Sim: {avg_sim_rf:.4f}")

                results.append({
                    'policy': policy_name,
                    'sample_idx': sample_idx,
                    'rho': rho,
                    'avg_sim_pathwise': avg_sim_pw,
                    'avg_sim_reinforce': avg_sim_rf,
                    'gt_norm': gt_norm
                })

        # Save results for this intensity
        output_dir = './results'
        os.makedirs(output_dir, exist_ok=True)

        filename = f'{output_dir}/gradient_comparison_{args.env.replace(".yaml", "")}_rho{rho}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results for rho={rho} saved to {filename}")


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Pathwise and REINFORCE gradient estimators for queuing networks."
    )
    parser.add_argument(
        '--env', type=str, default='criss_cross_bh.yaml',
        help="Environment configuration file name"
    )
    parser.add_argument(
        '--horizon', type=int, default=1000,
        help="Trajectory horizon (number of steps)"
    )
    parser.add_argument(
        '--gt_batch', type=int, default=1000000,
        help="Batch size for ground truth gradient estimation"
    )
    parser.add_argument(
        '--num_samples', type=int, default=100,
        help="Number of random policy parameter samples"
    )
    parser.add_argument(
        '--estimators_per_sample', type=int, default=100,
        help="Number of estimator samples for averaging similarity"
    )
    parser.add_argument(
        '--num_cores', type=int, default=100,
        help="Number of CPU cores for parallel computation"
    )
    return parser.parse_args()


if __name__ == '__main__':

    mp.set_start_method('spawn', force=True)

    args = parse_args()
    run_experiment(args)
