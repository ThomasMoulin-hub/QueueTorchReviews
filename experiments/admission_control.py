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
import torch.distributions.one_hot_categorical as one_hot_sample

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

def simulate_trajectory_batch(env_name, buffer_params, T, total_batch_size, device, seed=None, policy_type='MaxWeight', buffer_cost=1000.0, init_obs=None, init_state=None, temp=1.0):
    """
    Runs simulation for a batch of configurations.
    buffer_params: (total_batch_size, q)
    Returns: (total_batch_size,) cost, last_obs, last_state
    """
    config_path = f'./configs/env/{env_name}'
    if not os.path.exists(config_path): config_path = f'../configs/env/{env_name}'
    
    with open(config_path, 'r') as f:
        env_config = yaml.safe_load(f)
        
    dq = load_env(env_config, temp=temp, batch=total_batch_size, seed=seed, device=device)
    dq.buffer_control = True
    
    # Use high buffer cost to match paper/buffer_control.py
    if not hasattr(dq, 'b') or dq.b is None:
        dq.b = (torch.ones(dq.q) * buffer_cost).float().to(device)
    else:
        # Override if it exists but we want to enforce the experiment param
        dq.b = (torch.ones(dq.q) * buffer_cost).float().to(device)
        
    # dq = patch_env_for_stability(dq) # Removed to match buffer_control.py
    
    if init_obs is not None and init_state is not None:
        obs, state = init_obs, init_state
        # Ensure initial queues are within buffer limits (re-clipping for warm-start)
        queues, time = obs
        queues = torch.min(torch.stack((queues, buffer_params), dim=2), dim=2).values
        obs = obs.__class__(queues, time)
        state = state.__class__(queues, *state[1:])
    else:
        obs, state = dq.reset(buffer=buffer_params)
        
    total_cumulative_cost = torch.zeros(total_batch_size).to(device)
    total_time = torch.zeros(total_batch_size).to(device)
    
    if 're-reentrant' in env_name: fam = 'reentrant_2'
    else: fam = 'reentrant_1'
    physical_mapping = get_physical_mapping(fam, dq.q)
    
    for t in range(T):
        queues, time = obs
        if torch.isnan(queues).any():
            return torch.ones(total_batch_size).to(device) * float('nan'), obs, state

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
            # Matching buffer_control.py logic exactly
            # Line 63: argmax(mu * h * (queues > 0))
            best_q = torch.argmax(dq.mu * dq.h * (queues > 0.).unsqueeze(1), dim=2)
            pr = F.one_hot(best_q, num_classes=dq.q).float()
            
            # Line 64: pr = torch.minimum((pr * dq.network), queues.unsqueeze(1).repeat(1, dq.s, 1))
            pr = torch.minimum(pr * dq.network, queues.unsqueeze(1).expand(-1, dq.s, -1))
            
            # Line 65: fallback to uniform over dq.network if all 0
            is_all_zero = torch.all(pr == 0., dim=2).reshape(total_batch_size, dq.s, 1)
            pr = pr + is_all_zero * dq.network
            
            # Line 66: normalization (with relu and epsilon for stability)
            pr = F.relu(pr)
            pr = pr / (torch.sum(pr, dim=-1, keepdim=True) + 1e-8)
            
            # Line 71: Sampling
            action = one_hot_sample.OneHotCategorical(probs=pr).sample()

        step_cost, obs, state, event_time = get_total_cost(dq, state, obs, action, buffer_params)
        total_cumulative_cost += step_cost.squeeze(1)
        total_time += event_time.squeeze(1)

    avg_cost = total_cumulative_cost / (total_time + 1e-8)
    return avg_cost, obs, state

# --- Gradients (Vectorized) ---

def compute_pathwise_grad_L_batch(env_name, L_params, T, device, policy_type, buffer_cost, init_obs=None, init_state=None, temp=0.1):
    num_trials = L_params.shape[0]
    if L_params.grad is not None: L_params.grad.zero_()
        
    costs, last_obs, last_state = simulate_trajectory_batch(env_name, L_params, T, total_batch_size=num_trials, device=device, policy_type=policy_type, buffer_cost=buffer_cost, init_obs=init_obs, init_state=init_state, temp=temp)
    
    mask = ~torch.isnan(costs)
    if not mask.any(): return torch.zeros_like(L_params), last_obs, last_state
        
    loss = costs[mask].sum()
    loss.backward()
    
    grad = L_params.grad.clone()
    grad[~mask] = 0.0

    # Detach state to avoid backpropping through simulation history in next iteration
    last_obs = last_obs.__class__(*[x.detach() for x in last_obs])
    last_state = last_state.__class__(*[x.detach() for x in last_state])

    return grad, last_obs, last_state

def compute_spsa_grad_L_batch(env_name, L_params, T, spsa_batch_size, device, policy_type, buffer_cost, perturbation_scale=1.0, init_obs=None, init_state=None, temp=1.0):
    num_trials, q = L_params.shape
    total_sims = num_trials * spsa_batch_size
    
    eta = (torch.randint(0, 2, (total_sims, q)).float().to(device) * 2 - 1) * perturbation_scale
    L_expanded = L_params.unsqueeze(1).expand(-1, spsa_batch_size, -1).reshape(total_sims, q)
    
    L_plus = torch.round(F.relu(L_expanded + eta))
    L_minus = torch.round(F.relu(L_expanded - eta))

    # Handle warm-start expansions
    if init_obs is not None and init_state is not None:
        # Expand obs and state for the spsa batch
        # Obs: (queues, time) where queues is (num_trials, q)
        q_exp = init_obs.queues.unsqueeze(1).expand(-1, spsa_batch_size, -1).reshape(total_sims, -1)
        t_exp = init_obs.time.unsqueeze(1).expand(-1, spsa_batch_size, -1).reshape(total_sims, -1)
        obs_exp = init_obs._make((q_exp, t_exp))
        
        # State: queues, time, service_times, arrival_times
        s_exp = []
        for x in init_state:
            s_exp.append(x.unsqueeze(1).expand(-1, spsa_batch_size, -1).reshape(total_sims, -1))
        state_exp = init_state._make(s_exp)
    else:
        obs_exp, state_exp = None, None
    
    J_plus, _, _ = simulate_trajectory_batch(env_name, L_plus, T, total_sims, device, policy_type=policy_type, buffer_cost=buffer_cost, init_obs=obs_exp, init_state=state_exp, temp=temp)
    J_minus, last_obs_exp, last_state_exp = simulate_trajectory_batch(env_name, L_minus, T, total_sims, device, policy_type=policy_type, buffer_cost=buffer_cost, init_obs=obs_exp, init_state=state_exp, temp=temp)
    
    # Contract back last_obs and last_state (just take the mean or first of the spsa batch)
    # buffer_control.py just takes whatever came out of the last call.
    q_contract = last_obs_exp.queues.view(num_trials, spsa_batch_size, -1).mean(dim=1)
    t_contract = last_obs_exp.time.view(num_trials, spsa_batch_size, -1).mean(dim=1)
    last_obs = last_obs_exp.__class__(q_contract, t_contract)
    
    s_contract = []
    for x in last_state_exp:
        s_contract.append(x.view(num_trials, spsa_batch_size, -1).mean(dim=1))
    last_state = last_state_exp.__class__(*s_contract)

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

    # Detach state for consistency and safety
    last_obs = last_obs.__class__(*[x.detach() for x in last_obs])
    last_state = last_state.__class__(*[x.detach() for x in last_state])

    return grad, last_obs, last_state

# --- Optimization Loop (Vectorized) ---

def run_optimization_vectorized(env_name, method, batch_size, num_trials, iterations, device, policy_type, buffer_cost):
    config_path = f'./configs/env/{env_name}'
    if not os.path.exists(config_path): config_path = f'../configs/env/{env_name}'
    with open(config_path, 'r') as f: env_config = yaml.safe_load(f)
    q = len(env_config['h'])
    
    init_val = 1.0
    L_float = torch.ones((num_trials, q), device=device) * init_val
    L = L_float.clone()
    L.requires_grad = True
    
    lr = 1.0
    history = [] 
    last_obs, last_state = None, None
    
    for t in tqdm(range(iterations), desc=f"{env_name} {method} B={batch_size}", leave=False):
        if method == 'PATHWISE':
            # Use T=1000 to match buffer_control.py
            grad, last_obs, last_state = compute_pathwise_grad_L_batch(env_name, L, T=1000, device=device, policy_type=policy_type, buffer_cost=buffer_cost, init_obs=last_obs, init_state=last_state)
        elif method == 'SPSA':
            grad, last_obs, last_state = compute_spsa_grad_L_batch(env_name, L.detach(), T=1000, spsa_batch_size=batch_size, device=device, policy_type=policy_type, buffer_cost=buffer_cost, init_obs=last_obs, init_state=last_state)
        
        with torch.no_grad():
            grad = torch.nan_to_num(grad, 0.0)
            update = torch.sign(grad)
            L_float = F.relu(L_float - lr * update)
            L = torch.round(L_float).clone()
            L.requires_grad = True
        
        history.append(L.detach().cpu().tolist())
        
    # Final evaluation with T=50000 to match buffer_control.py
    final_costs, _, _ = simulate_trajectory_batch(env_name, L.detach(), T=50000, total_batch_size=num_trials, device=device, policy_type=policy_type, buffer_cost=buffer_cost)
    
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
    with open('./results/admission_control_like_buffer_full.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    with open('./results/admission_control_like_buffer_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
        
    print("\nDone. Results saved.")
