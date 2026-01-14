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

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from queuetorch.env import load_env, QueuingNetwork

def generate_reentrant_config(env_type, num_classes):
    """
    Generates configuration for Re-entrant lines dynamically.

    Args:
        env_type: 'reentrant_1' (1 server) or 'reentrant_2' (2 servers)
        num_classes: Number of job classes (queues)
    """
    K = num_classes

    # 1. Define Network (S x K)
    if env_type == 'reentrant_1':
        S = 1
        # Server 0 serves all classes
        network = torch.ones((S, K))
        mu = torch.ones((S, K)) * 1.0 # Service rate 1.0

    elif env_type == 'reentrant_2':
        S = 2
        network = torch.zeros((S, K))
        mu = torch.zeros((S, K))

        # Server 0 serves classes 0, 2, 4... (Indices 0-based)
        # Server 1 serves classes 1, 3, 5...
        for k in range(K):
            server_idx = k % 2
            network[server_idx, k] = 1.0
            mu[server_idx, k] = 1.0

    else:
        raise ValueError(f"Unknown env_type: {env_type}")

    # 2. Define Arrival Rates (Lambda)
    # External arrival only to class 0. 
    # Others set to epsilon to avoid div by zero (inf * 0 = nan)
    lam_val = np.ones(K) * 1e-20 
    lam_val[0] = 0.5 
    
    return {
        'name': f'{env_type}_{num_classes}',
        'lam_type': 'constant',
        'lam_params': {'val': lam_val.tolist()},
        'network': torch.eye(K).tolist(), 
        'mu': torch.eye(K).tolist(),      
        'h': [1.0] * K,
        'init_queues': [0] * K,
        'queue_event_options': generate_event_options(K).tolist(),
        'train_T': 10000,
        'test_T': 10000,
        'num_pool': 1,
        'physical_servers': get_physical_mapping(env_type, K)
    }

def get_physical_mapping(env_type, K):
    # Returns a list of lists. Each sublist is a group of virtual servers (indices)
    # that share a physical server.
    if env_type == 'reentrant_1':
        return [list(range(K))] 
    elif env_type == 'reentrant_2':
        s0 = [k for k in range(K) if k % 2 == 0]
        s1 = [k for k in range(K) if k % 2 == 1]
        return [s0, s1]
    return []

def generate_event_options(K):
    # Size: (Q + S, Q) = (2K, K)
    # Rows 0..K-1: Arrivals (only Q0 has real arrival, others 0)
    # Rows K..2K-1: Service completions

    options = torch.zeros((2 * K, K))

    # Arrivals
    # Arrival to Q_i adds 1 to Q_i.
    # Though lambda is 0 for i > 0, we define the event just in case.
    for i in range(K):
        options[i, i] = 1.0

    # Services
    # Virtual Server i (serving Q_i) finishes:
    # Q_i - 1
    # Q_{i+1} + 1 (if i < K-1)
    for i in range(K):
        row_idx = K + i
        options[row_idx, i] = -1.0
        if i < K - 1:
            options[row_idx, i+1] = 1.0
    return options

def get_total_cost(dq, state, obs, action, buffer):
    next_obs, next_state, holding_cost, overflow_cost, event_time = dq.step(state, action, buffer=buffer)
    total_step_cost = holding_cost + overflow_cost
    return total_step_cost, next_obs, next_state

def simulate_trajectory(env_config, buffer_params, T, batch_size, device, seed=None):
    if 'b' not in env_config:
        b_cost = torch.ones(len(env_config['h'])) * 10.0
    else:
        b_cost = torch.tensor(env_config['b'])
        
    # Load env manually constructed
    # We need to bypass load_env reading from file if we pass a dict
    # load_env takes 'env_config' dict, so it's fine.

    dq = QueuingNetwork(
        network=torch.tensor(env_config['network']),
        mu=torch.tensor(env_config['mu']),
        h=torch.tensor(env_config['h']),
        arrival_rates=lambda rng, t, batch: np.array(env_config['lam_params']['val']), 
        inter_arrival_dists=lambda state, batch: state.exponential(1, (batch, len(env_config['h']))),
        service_dists=lambda state, batch, t: state.exponential(1, (batch, len(env_config['h']))),
        queue_event_options=torch.tensor(env_config['queue_event_options']),
        batch=batch_size,
        temp=1.0, # Increased temp for stability
        seed=seed,
        device=torch.device(device),
        buffer_control=True,
        b=b_cost.float().to(device)
    )
    
    obs, state = dq.reset(buffer=buffer_params)
    total_cumulative_cost = 0
    
    physical_mapping = env_config['physical_servers']
    
    for t in range(T):
        queues, time = obs

        # POLICY: Last Buffer First Serve (LBFS) per physical server
        # For each physical server group, pick the non-empty queue with highest index.

        # Check for NaNs early
        if torch.isnan(queues).any():
            # print(f"NaN detected in queues at step {t}")
            return torch.tensor(float('nan'), device=device, requires_grad=True)

        action = torch.zeros((batch_size, dq.s, dq.q)).to(device)

        # dq.s is K (Virtual Servers)
        # dq.q is K
        # Action is (B, K, K). But since Network is Identity, effectively (B, K).
        # We just need to set the diagonal.

        for group in physical_mapping:
            # group is a list of indices [i1, i2, ...] sharing a server
            # We want to find max i in group such that queue[i] > 0

            # Extract queues for this group: (B, len(group))
            group_indices = torch.tensor(group).to(device)
            group_queues = queues[:, group_indices]
            valid = (group_queues > 0.001).float()

            # We want the right-most 1.
            # We can multiply valid by 1, 2, 3... then take argmax?
            # No, argmax takes first occurrence.
            # We want last. Reverse?

            # Let's assign priorities: Index i gets priority i.
            # Higher index = Higher priority.
            priorities = torch.arange(len(group)).to(device).float().unsqueeze(0) * valid
            # If all 0, priorities is 0.

            # We need to handle the case where all are empty -> Action 0.
            # If valid has any 1, max priority > 0 (if we shift indices to 1..G)

            priorities = (torch.arange(len(group)).to(device).float() + 1) * valid
            best_idx_in_group = torch.argmax(priorities, dim=1)
            any_valid = torch.max(valid, dim=1).values > 0
            local_action = F.one_hot(best_idx_in_group, num_classes=len(group)).float()

            # Zero out if no queue was valid
            local_action = local_action * any_valid.unsqueeze(1)

            # Place into global action matrix
            # Diagonal elements: action[b, k, k]
            for i, k in enumerate(group):
                action[:, k, k] = local_action[:, i]

        step_cost, obs, state = get_total_cost(dq, state, obs, action, buffer_params)
        total_cumulative_cost = total_cumulative_cost + step_cost

    return torch.mean(total_cumulative_cost)

def compute_pathwise_grad_L(env_config, L_param, T, device):
    L_param.grad = None
    loss = simulate_trajectory(env_config, L_param, T, batch_size=1, device=device)
    if torch.isnan(loss):
        return torch.zeros_like(L_param)
    loss.backward()
    return L_param.grad.clone()

def compute_spsa_grad_L(env_config, L_curr, T, batch_size, device, perturbation_scale=1.0):
    q = L_curr.shape[1]
    eta = (torch.randint(0, 2, (batch_size, q)).float().to(device) * 2 - 1) * perturbation_scale
    L_base = L_curr.repeat(batch_size, 1)
    L_plus = L_base + eta
    L_minus = L_base - eta
    
    # Helper to get batch costs
    def get_batch_costs(buffers):
        if 'b' not in env_config:
            b_cost = torch.ones(len(env_config['h'])) * 10.0
        else:
            b_cost = torch.tensor(env_config['b'])
            
        dq = QueuingNetwork(
            network=torch.tensor(env_config['network']),
            mu=torch.tensor(env_config['mu']),
            h=torch.tensor(env_config['h']),
            arrival_rates=lambda rng, t, batch: np.array(env_config['lam_params']['val']),
            inter_arrival_dists=lambda state, batch: state.exponential(1, (batch, len(env_config['h']))),
            service_dists=lambda state, batch, t: state.exponential(1, (batch, len(env_config['h']))),
            queue_event_options=torch.tensor(env_config['queue_event_options']),
            batch=batch_size,
            temp=1.0, # Increased temp
            seed=None,
            device=torch.device(device),
            buffer_control=True,
            b=b_cost.float().to(device)
        )
        
        obs, state = dq.reset(buffer=buffers)
        total_costs = torch.zeros(batch_size, 1).to(device)
        physical_mapping = env_config['physical_servers']
        
        for _ in range(T):
            queues, time = obs
            if torch.isnan(queues).any():
                return torch.ones(batch_size, 1).to(device) * float('nan')

            action = torch.zeros((batch_size, dq.s, dq.q)).to(device)
            for group in physical_mapping:
                group_indices = torch.tensor(group).to(device)
                group_queues = queues[:, group_indices]
                valid = (group_queues > 0.001).float()
                priorities = (torch.arange(len(group)).to(device).float() + 1) * valid
                best_idx_in_group = torch.argmax(priorities, dim=1)
                any_valid = torch.max(valid, dim=1).values > 0
                local_action = F.one_hot(best_idx_in_group, num_classes=len(group)).float()
                local_action = local_action * any_valid.unsqueeze(1)
                for i, k in enumerate(group):
                    action[:, k, k] = local_action[:, i]
            
            step_cost, obs, state = get_total_cost(dq, state, obs, action, buffers)
            total_costs += step_cost
        return total_costs

    J_plus = get_batch_costs(L_plus)
    J_minus = get_batch_costs(L_minus)
    
    # Handle NaNs in costs by zeroing gradient contribution
    mask = (~torch.isnan(J_plus)) & (~torch.isnan(J_minus))
    mask = mask.float()
    
    J_plus = torch.nan_to_num(J_plus, 0.0)
    J_minus = torch.nan_to_num(J_minus, 0.0)
    
    grad_estimates = 0.5 * (J_plus - J_minus) * (1.0 / eta) * mask
    grad = torch.mean(grad_estimates, dim=0, keepdim=True)
    return grad

def run_optimization(env_config, method, batch_size, iterations=100, device='cpu'):
    q = len(env_config['h'])
    init_val = 20.0
    L = torch.tensor([[init_val] * q], device=device, requires_grad=True)
    lr = 1.0 
    
    for t in tqdm(range(iterations), desc=f"{method} B={batch_size}", leave=False):
        if method == 'PATHWISE':
            grad = compute_pathwise_grad_L(env_config, L, T=100, device=device)
        elif method == 'SPSA':
            grad = compute_spsa_grad_L(env_config, L.detach(), T=100, batch_size=batch_size, device=device)
        
        # Check for NaN gradient
        if torch.isnan(grad).any():
            # print(f"NaN gradient at step {t}. Skipping update.")
            continue

        with torch.no_grad():
            update = torch.sign(grad)
            L -= lr * update
            L.clamp_(min=1.0)
            if L.grad is not None:
                L.grad.zero_()
                
    return L.detach()

def final_evaluation(env_config, L_final, device='cpu'):
    cost = simulate_trajectory(env_config, L_final, T=10000, batch_size=100, device=device)
    return cost.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    # Experiment Loop
    env_types = ['reentrant_1', 'reentrant_2']
    class_counts = [6, 9, 12, 15, 18, 21]
    
    results = {}
    
    for env_type in env_types:
        results[env_type] = {}
        print(f"\n=== Running {env_type} ===")
        
        for K in class_counts:
            print(f"  Classes: {K}")
            env_config = generate_reentrant_config(env_type, K)
            results[env_type][K] = {}
            
            # 1. PATHWISE (B=1)
            L_final_pw = run_optimization(env_config, 'PATHWISE', batch_size=1, device=args.device)
            cost_pw = final_evaluation(env_config, L_final_pw, device=args.device)
            results[env_type][K]['PATHWISE_B1'] = cost_pw
            print(f"    PW B=1: {cost_pw:.2f}")
            
            # 2. SPSA (B=10, 100, 1000)
            for b in [10, 100, 1000]:
                L_final_spsa = run_optimization(env_config, 'SPSA', batch_size=b, device=args.device)
                cost_spsa = final_evaluation(env_config, L_final_spsa, device=args.device)
                results[env_type][K][f'SPSA_B{b}'] = cost_spsa
                print(f"    SPSA B={b}: {cost_spsa:.2f}")
            
    # Save to file
    os.makedirs('./results', exist_ok=True)
    with open('./results/admission_control_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to ./results/admission_control_results.json")
