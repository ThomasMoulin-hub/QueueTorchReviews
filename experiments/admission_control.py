#!/user/tmm2219/.conda/envs/qt_env/bin/python
import torch
import numpy as np
import yaml
import argparse
import os
import json
from tqdm import tqdm
import sys
import torch.nn.functional as F
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from queuetorch.env import load_env, QueuingNetwork

# --- Helper Functions ---

def get_physical_mapping(env_type, K):
    # S0: Cols 0, 2. S1: Col 1.
    s0 = [k for k in range(K) if k % 3 != 1]
    s1 = [k for k in range(K) if k % 3 == 1]
    return [s0, s1]

def patch_env_for_stability(dq):
    original_arrival_rates = dq.arrival_rates
    def safe_arrival_rates(rng, t, batch):
        rates = original_arrival_rates(rng, t, batch)
        if isinstance(rates, np.ndarray):
            rates[rates == 0] = 1e-20
        elif torch.is_tensor(rates):
            rates[rates == 0] = 1e-20
        return rates
    dq.arrival_rates = safe_arrival_rates
    return dq

def get_total_cost(dq, state, obs, action, buffer):
    next_obs, next_state, holding_cost, overflow_cost, event_time = dq.step(state, action, buffer=buffer)
    total_step_cost = holding_cost + overflow_cost
    return total_step_cost, next_obs, next_state, event_time

# --- Simulation Core (Vectorized) ---

def simulate_trajectory_batch(env_name, buffer_params, T, total_batch_size, device, seed=None, policy_type='MaxWeight', buffer_cost=1000.0):
    """
    Runs simulation for a batch of configurations.
    buffer_params: (total_batch_size, q)
    Returns: (total_batch_size,) cost
    """
    config_path = f'./configs/env/{env_name}'
    if not os.path.exists(config_path): config_path = f'../configs/env/{env_name}'
    
    with open(config_path, 'r') as f:
        env_config = yaml.safe_load(f)
        
    dq = load_env(env_config, temp=1.0, batch=total_batch_size, seed=seed, device=device)
    dq.buffer_control = True
    
    # Use high buffer cost to match paper/buffer_control.py
    if not hasattr(dq, 'b') or dq.b is None:
        dq.b = (torch.ones(dq.q) * buffer_cost).float().to(device)
    else:
        # Override if it exists but we want to enforce the experiment param
        dq.b = (torch.ones(dq.q) * buffer_cost).float().to(device)
        
    dq = patch_env_for_stability(dq)
    
    obs, state = dq.reset(buffer=buffer_params)
    total_cumulative_cost = torch.zeros(total_batch_size).to(device)
    total_time = torch.zeros(total_batch_size).to(device)
    
    if 're-reentrant' in env_name: fam = 'reentrant_2'
    else: fam = 'reentrant_1'
    physical_mapping = get_physical_mapping(fam, dq.q)
    
    for t in range(T):
        queues, time = obs
        if torch.isnan(queues).any():
            return torch.ones(total_batch_size).to(device) * float('nan')

        action = torch.zeros((total_batch_size, dq.s, dq.q)).to(device)
        
        if policy_type == 'LBFS':
            for s_idx, group in enumerate(physical_mapping):
                group_indices = torch.tensor(group).to(device)
                group_queues = queues[:, group_indices]
                valid = (group_queues > 0.001).float()
                priorities = (torch.arange(len(group)).to(device).float() + 1) * valid
                best_idx_in_group = torch.argmax(priorities, dim=1)
                any_valid = torch.max(valid, dim=1).values > 0
                
                local_action = F.one_hot(best_idx_in_group, num_classes=len(group)).float()
                local_action = local_action * any_valid.unsqueeze(1)
                
                for i, k in enumerate(group):
                    if dq.s == dq.q: action[:, k, k] = local_action[:, i]
                    elif dq.s == 2: action[:, s_idx, k] = local_action[:, i]
                    elif s_idx < dq.s: action[:, s_idx, k] = local_action[:, i]
        
        elif policy_type == 'MaxWeight':
            # Heuristic from buffer_control.py: argmax(mu * h)
            # Since mu and h are usually 1, this is essentially static priority (lowest index first usually)
            logits = dq.mu * dq.h * (queues > 0.001).unsqueeze(1).float() * dq.network
            best_q = torch.argmax(logits, dim=2)
            action = F.one_hot(best_q, num_classes=dq.q).float()
            has_valid = torch.max(logits, dim=2).values > 0
            action = action * has_valid.unsqueeze(2)

        step_cost, obs, state, event_time = get_total_cost(dq, state, obs, action, buffer_params)
        total_cumulative_cost += step_cost.squeeze(1)
        total_time += event_time.squeeze(1)

    avg_cost = total_cumulative_cost / (total_time + 1e-8)
    return avg_cost

# --- Gradients (Vectorized) ---

def compute_pathwise_grad_L_batch(env_name, L_params, T, device, policy_type, buffer_cost):
    num_trials = L_params.shape[0]
    if L_params.grad is not None: L_params.grad.zero_()
        
    costs = simulate_trajectory_batch(env_name, L_params, T, total_batch_size=num_trials, device=device, policy_type=policy_type, buffer_cost=buffer_cost)
    
    mask = ~torch.isnan(costs)
    if not mask.any(): return torch.zeros_like(L_params)
        
    loss = costs[mask].sum()
    loss.backward()
    
    grad = L_params.grad.clone()
    grad[~mask] = 0.0
    return grad

def compute_spsa_grad_L_batch(env_name, L_params, T, spsa_batch_size, device, policy_type, buffer_cost, perturbation_scale=1.0):
    num_trials, q = L_params.shape
    total_sims = num_trials * spsa_batch_size
    
    eta = (torch.randint(0, 2, (total_sims, q)).float().to(device) * 2 - 1) * perturbation_scale
    L_expanded = L_params.unsqueeze(1).expand(-1, spsa_batch_size, -1).reshape(total_sims, q)
    
    L_plus = L_expanded + eta
    L_minus = L_expanded - eta
    
    J_plus = simulate_trajectory_batch(env_name, L_plus, T, total_sims, device, policy_type=policy_type, buffer_cost=buffer_cost)
    J_minus = simulate_trajectory_batch(env_name, L_minus, T, total_sims, device, policy_type=policy_type, buffer_cost=buffer_cost)
    
    J_plus = J_plus.view(num_trials, spsa_batch_size)
    J_minus = J_minus.view(num_trials, spsa_batch_size)
    eta = eta.view(num_trials, spsa_batch_size, q)
    
    mask = (~torch.isnan(J_plus)) & (~torch.isnan(J_minus)) 
    mask = mask.float().unsqueeze(2) 
    
    J_plus = torch.nan_to_num(J_plus, 0.0).unsqueeze(2) 
    J_minus = torch.nan_to_num(J_minus, 0.0).unsqueeze(2)
    
    grad_estimates = 0.5 * (J_plus - J_minus) * (1.0 / eta) * mask
    valid_counts = mask.sum(dim=1) 
    valid_counts = torch.where(valid_counts == 0, torch.ones_like(valid_counts), valid_counts)
    
    grad = grad_estimates.sum(dim=1) / valid_counts 
    return grad

# --- Optimization Loop (Vectorized) ---

def run_optimization_vectorized(env_name, method, batch_size, num_trials, iterations, device, policy_type, buffer_cost):
    config_path = f'./configs/env/{env_name}'
    if not os.path.exists(config_path): config_path = f'../configs/env/{env_name}'
    with open(config_path, 'r') as f: env_config = yaml.safe_load(f)
    q = len(env_config['h'])
    
    init_val = 20.0
    L = torch.ones((num_trials, q), device=device) * init_val
    L.requires_grad = True
    
    lr = 1.0
    history = [] 
    
    for t in tqdm(range(iterations), desc=f"{env_name} {method} B={batch_size}", leave=False):
        if method == 'PATHWISE':
            grad = compute_pathwise_grad_L_batch(env_name, L, T=100, device=device, policy_type=policy_type, buffer_cost=buffer_cost)
        elif method == 'SPSA':
            grad = compute_spsa_grad_L_batch(env_name, L.detach(), T=100, spsa_batch_size=batch_size, device=device, policy_type=policy_type, buffer_cost=buffer_cost)
        
        with torch.no_grad():
            grad = torch.nan_to_num(grad, 0.0)
            update = torch.sign(grad)
            L -= lr * update
            L.clamp_(min=1.0)
            if L.grad is not None: L.grad.zero_()
        
        history.append(L.detach().cpu().tolist())
        
    final_costs = simulate_trajectory_batch(env_name, L.detach(), T=10000, total_batch_size=num_trials, device=device, policy_type=policy_type, buffer_cost=buffer_cost)
    
    return {
        'env': env_name,
        'method': method,
        'batch': batch_size,
        'final_costs': final_costs.detach().cpu().tolist(), 
        'final_Ls': L.detach().cpu().tolist()
    }

# --- Main ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_trials', type=int, default=5)
    # Changed default policy to MaxWeight to match buffer_control.py and paper results
    parser.add_argument('--policy', type=str, default='MaxWeight', choices=['LBFS', 'MaxWeight'])
    # Added buffer cost argument, default 1000
    parser.add_argument('--buffer_cost', type=float, default=1000.0)
    parser.add_argument('--hyper', action='store_true', default=False, help='Use hyper versions of env configs')
    args = parser.parse_args()
    
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
            
            if not os.path.exists(f"./configs/env/{env_filename}") and not os.path.exists(f"../configs/env/{env_filename}"):
                continue
            
            print(f"Processing {env_filename} ({paper_name}, K={K})...")
            
            # 1. PATHWISE B=1
            res_pw = run_optimization_vectorized(env_filename, 'PATHWISE', 1, args.num_trials, 100, args.device, args.policy, args.buffer_cost)
            all_results.append(res_pw)
            summary[env_filename]['PATHWISE_B1'] = {
                'mean': np.nanmean(res_pw['final_costs']),
                'std': np.nanstd(res_pw['final_costs'])
            }
            
            # 2. SPSA B=10, 100, 1000
            for b in [10, 100, 1000]:
                res_spsa = run_optimization_vectorized(env_filename, 'SPSA', b, args.num_trials, 100, args.device, args.policy, args.buffer_cost)
                all_results.append(res_spsa)
                summary[env_filename][f'SPSA_B{b}'] = {
                    'mean': np.nanmean(res_spsa['final_costs']),
                    'std': np.nanstd(res_spsa['final_costs'])
                }
                
    os.makedirs('./results', exist_ok=True)
    with open('./results/admission_control_full.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    with open('./results/admission_control_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
        
    print("\nDone. Results saved.")
