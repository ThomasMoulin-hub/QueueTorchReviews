#!<path to your python bin>
"""
Admission Control Experiment.

This script compares two gradient estimators for optimizing buffer sizes in queuing networks:
1. **Pathwise Estimator**: Uses differentiable simulation with batch size B=1.
2. **SPSA Estimator**: Uses Simultaneous Perturbation Stochastic Approximation with various batch sizes.

The experiment optimizes buffer limits (L) for different queuing network configurations
and compares the convergence and final costs of both methods.

Usage:
    python admission_control.py --device cpu --num_trials 5 --policy MaxWeight
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions.one_hot_categorical as one_hot_sample
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from queuetorch.env import load_env, QueuingNetwork, Obs, EnvState

# =============================================================================
# Constants
# =============================================================================

DEFAULT_BUFFER_COST = 1000.0
DEFAULT_LEARNING_RATE = 1.0
SIMULATION_HORIZON = 1000
EVALUATION_HORIZON = 50000
NUM_ITERATIONS = 100


# =============================================================================
# Helper Functions
# =============================================================================


def get_physical_mapping(env_type: str, num_queues: int) -> List[List[int]]:
    """
    Get physical server-to-queue mapping based on environment type.

    For reentrant networks, queues are divided into two groups:
    - Server 0: handles columns 0, 2 (queues where index % 3 != 1)
    - Server 1: handles column 1 (queues where index % 3 == 1)

    Args:
        env_type: Environment family ('reentrant_1' or 'reentrant_2').
        num_queues: Total number of queues in the network.

    Returns:
        List of queue indices for each server.
    """
    server_0_queues = [k for k in range(num_queues) if k % 3 != 1]
    server_1_queues = [k for k in range(num_queues) if k % 3 == 1]
    return [server_0_queues, server_1_queues]


def _load_env_config(env_name: str) -> Dict[str, Any]:
    """
    Load environment configuration from YAML file.

    Args:
        env_name: Name of the environment configuration file.

    Returns:
        Environment configuration dictionary.
    """
    config_path = f'./configs/env/{env_name}'
    if not os.path.exists(config_path):
        config_path = f'../configs/env/{env_name}'

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# =============================================================================
# Simulation Core
# =============================================================================


def simulate_trajectory_batch(
    env_name: str,
    buffer_params: torch.Tensor,
    T: int,
    total_batch_size: int,
    device: str,
    seed: Optional[int] = None,
    policy_type: str = 'MaxWeight',
    buffer_cost: float = DEFAULT_BUFFER_COST,
    init_obs: Optional[Obs] = None,
    init_state: Optional[EnvState] = None,
    temp: float = 1.0
) -> Tuple[torch.Tensor, Obs, EnvState]:
    """
    Simulate a batch of trajectories with given buffer parameters.

    This function runs parallel simulations of the queuing network, applying
    either MaxWeight or LBFS scheduling policies.

    Args:
        env_name: Name of the environment configuration file.
        buffer_params: Buffer limits tensor (total_batch_size, q).
        T: Number of simulation steps (horizon).
        total_batch_size: Number of parallel simulations.
        device: Computation device ('cpu' or 'cuda').
        seed: Random seed for reproducibility.
        policy_type: Scheduling policy ('MaxWeight' or 'LBFS').
        buffer_cost: Cost penalty for buffer overflow.
        init_obs: Optional initial observation for warm-start.
        init_state: Optional initial state for warm-start.
        temp: Temperature parameter for environment.

    Returns:
        Tuple of (average_costs, last_obs, last_state).
    """
    env_config = _load_env_config(env_name)

    # Initialize environment
    dq = load_env(env_config, temp=temp, batch=total_batch_size, seed=seed, device=device)
    dq.buffer_control = True
    dq.b = torch.ones(dq.q, device=device).float() * buffer_cost

    # Initialize or restore state
    if init_obs is not None and init_state is not None:
        obs, state = init_obs, init_state
        # Clip queues to buffer limits
        queues, time = obs
        queues = torch.min(torch.stack((queues, buffer_params), dim=2), dim=2).values
        obs = Obs(queues, time)
        state = EnvState(queues, *state[1:])
    else:
        obs, state = dq.reset(buffer=buffer_params)

    # Tracking variables
    total_cumulative_cost = torch.zeros(total_batch_size, device=device)
    total_time = torch.zeros(total_batch_size, device=device)

    # Determine physical mapping for LBFS policy
    env_family = 'reentrant_2' if 're-reentrant' in env_name else 'reentrant_1'
    physical_mapping = get_physical_mapping(env_family, dq.q)

    # Simulation loop
    for _ in range(T):
        queues, time = obs

        # Check for numerical issues
        if torch.isnan(queues).any():
            return torch.full((total_batch_size,), float('nan'), device=device), obs, state

        # Apply scheduling policy
        if policy_type == 'LBFS':
            action = _apply_lbfs_policy(queues, physical_mapping, dq, total_batch_size, device)
        else:  # MaxWeight
            action = _apply_maxweight_policy(queues, dq, total_batch_size, device)

        # Execute step
        next_obs, next_state, holding_cost, overflow_cost, event_time = dq.step(
            state, action, buffer=buffer_params
        )
        step_cost = holding_cost + overflow_cost

        total_cumulative_cost += step_cost.squeeze(1)
        total_time += event_time.squeeze(1)

        obs, state = next_obs, next_state

    # Compute average cost rate
    avg_cost = total_cumulative_cost / (total_time + 1e-8)
    return avg_cost, obs, state


def _apply_lbfs_policy(
    queues: torch.Tensor,
    physical_mapping: List[List[int]],
    dq: QueuingNetwork,
    batch_size: int,
    device: str
) -> torch.Tensor:
    """
    Apply Last Buffer First Serve (LBFS) scheduling policy.

    Args:
        queues: Current queue lengths (batch_size, q).
        physical_mapping: Server-to-queue mapping.
        dq: Queuing network environment.
        batch_size: Number of parallel simulations.
        device: Computation device.

    Returns:
        Action tensor (batch_size, s, q).
    """
    action = torch.zeros((batch_size, dq.s, dq.q), device=device)

    for s_idx, group in enumerate(physical_mapping):
        group_indices = torch.tensor(group, device=device)
        group_queues = queues[:, group_indices]

        # Identify non-empty queues
        valid = (group_queues > 0.001).float()

        # Priority: higher index = higher priority (LBFS)
        priorities = (torch.arange(len(group), device=device).float() + 1) * valid
        best_idx_in_group = torch.argmax(priorities, dim=1)
        any_valid = torch.max(valid, dim=1).values > 0

        # Create one-hot action
        local_action = F.one_hot(best_idx_in_group, num_classes=len(group)).float()
        local_action = local_action * any_valid.unsqueeze(1)

        # Map to global action tensor
        for i, k in enumerate(group):
            if dq.s == dq.q:
                action[:, k, k] = local_action[:, i]
            elif dq.s == 2:
                action[:, s_idx, k] = local_action[:, i]
            elif s_idx < dq.s:
                action[:, s_idx, k] = local_action[:, i]

    return action


def _apply_maxweight_policy(
    queues: torch.Tensor,
    dq: QueuingNetwork,
    batch_size: int,
    device: str
) -> torch.Tensor:
    """
    Apply MaxWeight scheduling policy.

    MaxWeight selects the queue with highest (service_rate * holding_cost * occupancy).

    Args:
        queues: Current queue lengths (batch_size, q).
        dq: Queuing network environment.
        batch_size: Number of parallel simulations.
        device: Computation device.

    Returns:
        Sampled action tensor (batch_size, s, q).
    """
    # Compute priorities: mu * h * indicator(queue > 0)
    best_q = torch.argmax(dq.mu * dq.h * (queues > 0.).unsqueeze(1), dim=2)
    pr = F.one_hot(best_q, num_classes=dq.q).float()

    # Apply network constraints
    pr = torch.minimum(pr * dq.network, queues.unsqueeze(1).expand(-1, dq.s, -1))

    # Fallback for zero probability rows
    is_all_zero = torch.all(pr == 0., dim=2).reshape(batch_size, dq.s, 1)
    pr = pr + is_all_zero * dq.network

    # Normalize
    pr = F.relu(pr)
    pr = pr / (torch.sum(pr, dim=-1, keepdim=True) + 1e-8)

    # Sample action
    action = one_hot_sample.OneHotCategorical(probs=pr).sample()
    return action


# =============================================================================
# Gradient Estimators
# =============================================================================


def compute_pathwise_grad_L_batch(
    env_name: str,
    L_params: torch.Tensor,
    T: int,
    device: str,
    policy_type: str,
    buffer_cost: float,
    init_obs: Optional[Obs] = None,
    init_state: Optional[EnvState] = None,
    temp: float = 0.1
) -> Tuple[torch.Tensor, Obs, EnvState]:
    """
    Compute pathwise (differentiable) gradient for buffer parameters.

    Uses automatic differentiation through the simulation to estimate
    gradients with respect to buffer limits.

    Args:
        env_name: Environment configuration file name.
        L_params: Buffer limit parameters (num_trials, q) with requires_grad=True.
        T: Simulation horizon.
        device: Computation device.
        policy_type: Scheduling policy type.
        buffer_cost: Cost penalty for overflow.
        init_obs: Optional warm-start observation.
        init_state: Optional warm-start state.
        temp: Temperature parameter.

    Returns:
        Tuple of (gradients, last_obs, last_state).
    """
    num_trials = L_params.shape[0]

    if L_params.grad is not None:
        L_params.grad.zero_()

    costs, last_obs, last_state = simulate_trajectory_batch(
        env_name, L_params, T,
        total_batch_size=num_trials,
        device=device,
        policy_type=policy_type,
        buffer_cost=buffer_cost,
        init_obs=init_obs,
        init_state=init_state,
        temp=temp
    )

    # Handle NaN costs
    mask = ~torch.isnan(costs)
    if not mask.any():
        return torch.zeros_like(L_params), last_obs, last_state

    loss = costs[mask].sum()
    loss.backward()
    
    grad = L_params.grad.clone()
    grad[~mask] = 0.0

    # Detach state to avoid backprop through simulation history
    last_obs = Obs(*[x.detach() for x in last_obs])
    last_state = EnvState(*[x.detach() for x in last_state])

    return grad, last_obs, last_state


def compute_spsa_grad_L_batch(
    env_name: str,
    L_params: torch.Tensor,
    T: int,
    spsa_batch_size: int,
    device: str,
    policy_type: str,
    buffer_cost: float,
    perturbation_scale: float = 1.0,
    init_obs: Optional[Obs] = None,
    init_state: Optional[EnvState] = None,
    temp: float = 1.0
) -> Tuple[torch.Tensor, Obs, EnvState]:
    """
    Compute SPSA (Simultaneous Perturbation Stochastic Approximation) gradient.

    Uses finite differences with random perturbations to estimate gradients
    without requiring differentiability.

    Args:
        env_name: Environment configuration file name.
        L_params: Buffer limit parameters (num_trials, q).
        T: Simulation horizon.
        spsa_batch_size: Number of perturbation samples per trial.
        device: Computation device.
        policy_type: Scheduling policy type.
        buffer_cost: Cost penalty for overflow.
        perturbation_scale: Scale of random perturbations.
        init_obs: Optional warm-start observation.
        init_state: Optional warm-start state.
        temp: Temperature parameter.

    Returns:
        Tuple of (gradients, last_obs, last_state).
    """
    num_trials, q = L_params.shape
    total_sims = num_trials * spsa_batch_size
    
    # Generate random perturbations (+/- perturbation_scale)
    eta = (torch.randint(0, 2, (total_sims, q), device=device).float() * 2 - 1) * perturbation_scale
    L_expanded = L_params.unsqueeze(1).expand(-1, spsa_batch_size, -1).reshape(total_sims, q)
    
    L_plus = torch.round(F.relu(L_expanded + eta))
    L_minus = torch.round(F.relu(L_expanded - eta))

    # Expand warm-start states if provided
    obs_exp, state_exp = _expand_warm_start(init_obs, init_state, spsa_batch_size, total_sims)

    # Run simulations for both perturbations
    J_plus, _, _ = simulate_trajectory_batch(
        env_name, L_plus, T, total_sims, device,
        policy_type=policy_type, buffer_cost=buffer_cost,
        init_obs=obs_exp, init_state=state_exp, temp=temp
    )
    J_minus, last_obs_exp, last_state_exp = simulate_trajectory_batch(
        env_name, L_minus, T, total_sims, device,
        policy_type=policy_type, buffer_cost=buffer_cost,
        init_obs=obs_exp, init_state=state_exp, temp=temp
    )

    # Contract results back to trial dimension
    last_obs, last_state = _contract_states(
        last_obs_exp, last_state_exp, num_trials, spsa_batch_size
    )

    # Compute gradient estimates
    grad = _compute_spsa_gradient(J_plus, J_minus, eta, num_trials, spsa_batch_size, q)

    # Detach states
    last_obs = Obs(*[x.detach() for x in last_obs])
    last_state = EnvState(*[x.detach() for x in last_state])

    return grad, last_obs, last_state


def _expand_warm_start(
    init_obs: Optional[Obs],
    init_state: Optional[EnvState],
    spsa_batch_size: int,
    total_sims: int
) -> Tuple[Optional[Obs], Optional[EnvState]]:
    """Expand warm-start observation and state for SPSA batch."""
    if init_obs is None or init_state is None:
        return None, None

    q_exp = init_obs.queues.unsqueeze(1).expand(-1, spsa_batch_size, -1).reshape(total_sims, -1)
    t_exp = init_obs.time.unsqueeze(1).expand(-1, spsa_batch_size, -1).reshape(total_sims, -1)
    obs_exp = Obs(q_exp, t_exp)

    s_exp = [x.unsqueeze(1).expand(-1, spsa_batch_size, -1).reshape(total_sims, -1) for x in init_state]
    state_exp = EnvState(*s_exp)

    return obs_exp, state_exp


def _contract_states(
    last_obs_exp: Obs,
    last_state_exp: EnvState,
    num_trials: int,
    spsa_batch_size: int
) -> Tuple[Obs, EnvState]:
    """Contract expanded states back to trial dimension using mean."""
    q_contract = last_obs_exp.queues.view(num_trials, spsa_batch_size, -1).mean(dim=1)
    t_contract = last_obs_exp.time.view(num_trials, spsa_batch_size, -1).mean(dim=1)
    last_obs = Obs(q_contract, t_contract)

    s_contract = [x.view(num_trials, spsa_batch_size, -1).mean(dim=1) for x in last_state_exp]
    last_state = EnvState(*s_contract)

    return last_obs, last_state


def _compute_spsa_gradient(
    J_plus: torch.Tensor,
    J_minus: torch.Tensor,
    eta: torch.Tensor,
    num_trials: int,
    spsa_batch_size: int,
    q: int
) -> torch.Tensor:
    """Compute SPSA gradient estimate from perturbation costs."""
    J_plus = J_plus.view(num_trials, spsa_batch_size)
    J_minus = J_minus.view(num_trials, spsa_batch_size)
    eta = eta.view(num_trials, spsa_batch_size, q)
    
    # Create validity mask
    mask = (~torch.isnan(J_plus)) & (~torch.isnan(J_minus))
    mask = mask.float().unsqueeze(2)

    # Replace NaN with 0
    J_plus = torch.nan_to_num(J_plus, 0.0).unsqueeze(2)
    J_minus = torch.nan_to_num(J_minus, 0.0).unsqueeze(2)
    
    # SPSA gradient: (f(x+η) - f(x-η)) / (2η)
    grad_estimates = 0.5 * (J_plus - J_minus) * (1.0 / eta) * mask

    # Average over valid samples
    valid_counts = mask.sum(dim=1)
    valid_counts = torch.where(valid_counts == 0, torch.ones_like(valid_counts), valid_counts)
    
    return grad_estimates.sum(dim=1) / valid_counts


# =============================================================================
# Optimization Loop
# =============================================================================


def run_optimization_vectorized(
    env_name: str,
    method: str,
    batch_size: int,
    num_trials: int,
    iterations: int,
    device: str,
    policy_type: str,
    buffer_cost: float
) -> Dict[str, Any]:
    """
    Run buffer optimization using either Pathwise or SPSA gradient estimation.

    Args:
        env_name: Environment configuration file name.
        method: Optimization method ('PATHWISE' or 'SPSA').
        batch_size: Batch size for gradient estimation.
        num_trials: Number of parallel optimization trials.
        iterations: Number of optimization iterations.
        device: Computation device.
        policy_type: Scheduling policy type.
        buffer_cost: Cost penalty for overflow.

    Returns:
        Dictionary containing optimization results.
    """
    env_config = _load_env_config(env_name)
    q = len(env_config['h'])
    
    # Initialize buffer limits
    L_float = torch.ones((num_trials, q), device=device)
    L = L_float.clone()
    L.requires_grad = True
    
    lr = DEFAULT_LEARNING_RATE
    history = []
    last_obs, last_state = None, None
    grad = torch.zeros((num_trials, q), device=device)

    # Optimization loop
    for _ in tqdm(range(iterations), desc=f"{env_name} {method} B={batch_size}", leave=False):
        if method == 'PATHWISE':
            grad, last_obs, last_state = compute_pathwise_grad_L_batch(
                env_name, L, T=SIMULATION_HORIZON, device=device,
                policy_type=policy_type, buffer_cost=buffer_cost,
                init_obs=last_obs, init_state=last_state
            )
        elif method == 'SPSA':
            grad, last_obs, last_state = compute_spsa_grad_L_batch(
                env_name, L.detach(), T=SIMULATION_HORIZON, spsa_batch_size=batch_size,
                device=device, policy_type=policy_type, buffer_cost=buffer_cost,
                init_obs=last_obs, init_state=last_state
            )

        # Update buffer limits using sign gradient descent
        with torch.no_grad():
            grad = torch.nan_to_num(grad, 0.0)
            update = torch.sign(grad)
            L_float = F.relu(L_float - lr * update)
            L = torch.round(L_float).clone()
            L.requires_grad = True
        
        history.append(L.detach().cpu().tolist())

    # Final evaluation with long horizon
    final_costs, _, _ = simulate_trajectory_batch(
        env_name, L.detach(), T=EVALUATION_HORIZON,
        total_batch_size=num_trials, device=device,
        policy_type=policy_type, buffer_cost=buffer_cost
    )

    return {
        'env': env_name,
        'method': method,
        'batch': batch_size,
        'final_costs': final_costs.detach().cpu().tolist(),
        'final_Ls': L.detach().cpu().tolist()
    }


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Pathwise and SPSA gradient estimators for buffer optimization."
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help="Computation device ('cpu' or 'cuda')"
    )
    parser.add_argument(
        '--num_trials', type=int, default=5,
        help="Number of parallel optimization trials"
    )
    parser.add_argument(
        '--policy', type=str, default='MaxWeight', choices=['LBFS', 'MaxWeight'],
        help="Scheduling policy to use"
    )
    parser.add_argument(
        '--buffer_cost', type=float, default=DEFAULT_BUFFER_COST,
        help="Cost penalty for buffer overflow"
    )
    parser.add_argument(
        '--hyper', action='store_true', default=False,
        help="Use hyper versions of environment configs"
    )
    return parser.parse_args()


def main() -> None:
    """Main experiment execution."""
    args = parse_args()

    # Environment families and configurations
    families = {
        'reentrant_1': 'reentrant',
        'reentrant_2': 're-reentrant'
    }
    class_counts = [6, 9, 12, 15, 18, 21]
    
    all_results = []
    summary = defaultdict(lambda: defaultdict(dict))
    
    for paper_name, file_prefix in families.items():
        for K in class_counts:
            layers = K // 3
            suffix = "_hyper.yaml" if args.hyper else ".yaml"
            env_filename = f"{file_prefix}_{layers}{suffix}"
            
            # Check if config exists
            if not os.path.exists(f"./configs/env/{env_filename}") and \
               not os.path.exists(f"../configs/env/{env_filename}"):
                continue
            
            print(f"Processing {env_filename} ({paper_name}, K={K})...")
            
            # Run Pathwise optimization (B=1)
            res_pw = run_optimization_vectorized(
                env_filename, 'PATHWISE', 1, args.num_trials,
                NUM_ITERATIONS, args.device, args.policy, args.buffer_cost
            )
            all_results.append(res_pw)
            summary[env_filename]['PATHWISE_B1'] = {
                'mean': np.nanmean(res_pw['final_costs']),
                'std': np.nanstd(res_pw['final_costs'])
            }
            
            # Run SPSA optimization with various batch sizes
            for b in [10, 100, 1000]:
                res_spsa = run_optimization_vectorized(
                    env_filename, 'SPSA', b, args.num_trials,
                    NUM_ITERATIONS, args.device, args.policy, args.buffer_cost
                )
                all_results.append(res_spsa)
                summary[env_filename][f'SPSA_B{b}'] = {
                    'mean': np.nanmean(res_spsa['final_costs']),
                    'std': np.nanstd(res_spsa['final_costs'])
                }

    # Save results
    os.makedirs('./results', exist_ok=True)

    with open('./results/admission_control_full.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    with open('./results/admission_control_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

    print("\nDone. Results saved.")


if __name__ == '__main__':
    main()

