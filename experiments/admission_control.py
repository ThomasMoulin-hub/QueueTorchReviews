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

from queuetorch.env import load_env

def get_total_cost(dq, state, obs, action, buffer):
    """
    Executes one step and returns the total cost (Holding + Overflow).
    Corresponds to Equation 15 term for one step.
    """
    # Step returns: obs, next_state, cost (holding), buffer_cost (overflow), event_time
    next_obs, next_state, holding_cost, overflow_cost, event_time = dq.step(state, action, buffer=buffer)
    
    # Total cost = Holding Cost + Overflow Cost
    # Note: holding_cost is already (h^T x_k) * tau_{k+1}
    # overflow_cost is b^T o_k
    total_step_cost = holding_cost + overflow_cost
    
    return total_step_cost, next_obs, next_state

def simulate_trajectory(env_config, buffer_params, T, batch_size, device, seed=None):
    """
    Runs a simulation for T steps with a fixed buffer size L.
    Returns the mean total cost over the batch.
    """
    # Load env with buffer control enabled
    # We need to ensure 'b' (overflow cost) is in config or passed manually
    # For this experiment, we assume 'b' is defined or we set a default
    if 'b' not in env_config:
        # Default overflow cost if not in yaml (arbitrary high cost to discourage overflow)
        b_cost = torch.ones(len(env_config['h'])) * 10.0
    else:
        b_cost = torch.tensor(env_config['b'])
        
    dq = load_env(env_config, temp=0.1, batch=batch_size, seed=seed, device=device)
    dq.buffer_control = True
    dq.b = b_cost.float().to(device)
    
    # Reset
    obs, state = dq.reset(buffer=buffer_params)
    
    total_cumulative_cost = 0
    
    # Simple policy: Random routing or fixed routing?
    # The prompt says "Given a fixed routing policy". 
    # We will use a uniform random policy or a simple heuristic if not specified.
    # For simplicity and to focus on L, we use a uniform random policy masked by network.
    
    for _ in range(T):
        queues, time = obs
        
        # Uniform random action compatible with network
        # We can use the logic from env.py to generate valid actions or just random
        # Here we assume a simple valid action selection (e.g., random valid server-queue pair)
        # For speed in "Pathwise", we want this to be differentiable if the policy was learned,
        # but here the policy is fixed.
        
        # Generate random priorities
        pr = torch.rand((batch_size, dq.s, dq.q)).to(device)
        
        # Mask and normalize (Softmax policy logic)
        pr = pr * dq.network
        pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
        # Handle empty queues case to avoid NaN
        pr += 1 * torch.all(pr == 0., dim=2).reshape(batch_size, dq.s, 1).repeat(1, 1, dq.q) * dq.network
        pr /= torch.sum(pr, dim=-1, keepdim=True)
        
        # Sample action (Discrete)
        # For Pathwise to work on L, the action doesn't strictly need to be differentiable 
        # w.r.t policy parameters, but the state update w.r.t L must be maintained.
        # Since L affects the *next* state via min(), discrete actions are fine 
        # as long as we don't need gradients w.r.t the policy.
        action = pr # Use probabilities as "soft" action or sample?
        # The prompt implies standard simulation. Let's use soft action for differentiability 
        # stability or hard action. Given "Pathwise" usually implies STE or reparam, 
        # but here we differentiate w.r.t L, not action.
        # Let's use the probabilities directly (Fluid approximation-ish) or STE.
        # To be safe and consistent with previous experiments:
        action = torch.round(pr) # Hard action
        
        step_cost, obs, state = get_total_cost(dq, state, obs, action, buffer_params)
        total_cumulative_cost = total_cumulative_cost + step_cost

    return torch.mean(total_cumulative_cost)

def compute_pathwise_grad_L(env_config, L_param, T, device):
    """
    Computes gradient w.r.t L using Pathwise estimator (B=1).
    """
    L_param.grad = None
    
    # We need to ensure L is treated as continuous during the forward pass
    # The environment handles L in: queues = min(queues + delta, L)
    # This is differentiable.
    
    loss = simulate_trajectory(env_config, L_param, T, batch_size=1, device=device)
    loss.backward()
    
    return L_param.grad.clone()

def compute_spsa_grad_L(env_config, L_curr, T, batch_size, device, perturbation_scale=1.0):
    """
    Computes SPSA gradient estimator (Eq 17).
    """
    # L_curr is a tensor of shape (1, q)
    q = L_curr.shape[1]
    
    # 1. Generate perturbations eta (Bernoulli +/- 1)
    # Shape: (batch_size, q)
    eta = (torch.randint(0, 2, (batch_size, q)).float().to(device) * 2 - 1) * perturbation_scale
    
    # 2. Prepare perturbed parameters
    # We need to broadcast L to batch size
    L_base = L_curr.repeat(batch_size, 1)
    
    L_plus = L_base + eta
    L_minus = L_base - eta
    
    # 3. Evaluate J(L + eta) and J(L - eta)
    # We can run these in parallel batches if the simulator supports passing a batch of L
    # The current env.py reset() takes 'buffer'. 
    # If we pass a (B, q) buffer to reset, does it handle it?
    # env.py: queues = torch.min(torch.stack((queues, buffer), dim = 2), dim = 2).values
    # If buffer is (B, q), stack works. Yes, it seems supported!
    
    loss_plus = simulate_trajectory(env_config, L_plus, T, batch_size, device)
    loss_minus = simulate_trajectory(env_config, L_minus, T, batch_size, device)
    
    # Note: simulate_trajectory returns MEAN loss over batch. 
    # But for SPSA we need individual losses to multiply by individual eta.
    # We need to modify simulate_trajectory or run it differently.
    # Actually, let's look at simulate_trajectory again. It returns torch.mean.
    # We need the vector of costs.
    
    # Let's inline the simulation here for SPSA to get vector costs
    def get_batch_costs(buffers):
        if 'b' not in env_config:
            b_cost = torch.ones(len(env_config['h'])) * 10.0
        else:
            b_cost = torch.tensor(env_config['b'])
        
        dq = load_env(env_config, temp=0.1, batch=batch_size, seed=None, device=device)
        dq.buffer_control = True
        dq.b = b_cost.float().to(device)
        
        obs, state = dq.reset(buffer=buffers)
        total_costs = torch.zeros(batch_size, 1).to(device)
        
        for _ in range(T):
            queues, time = obs
            pr = torch.rand((batch_size, dq.s, dq.q)).to(device)
            pr = pr * dq.network
            pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
            pr += 1 * torch.all(pr == 0., dim=2).reshape(batch_size, dq.s, 1).repeat(1, 1, dq.q) * dq.network
            pr /= torch.sum(pr, dim=-1, keepdim=True)
            action = torch.round(pr)
            
            step_cost, obs, state = get_total_cost(dq, state, obs, action, buffers)
            total_costs += step_cost
            
        return total_costs

    J_plus = get_batch_costs(L_plus)
    J_minus = get_batch_costs(L_minus)
    
    # Eq 17: Gradient = mean( 0.5 * (J+ - J-) * eta^(-1) )
    # Since eta is +/- 1 (times scale), eta^(-1) = eta / scale^2
    # If scale=1, eta^(-1) = eta.
    # The prompt says multiply by eta. If eta is +/- 1, dividing by eta is same as multiplying.
    
    # Shape: (B, 1) * (B, q) -> (B, q)
    grad_estimates = 0.5 * (J_plus - J_minus) * (1.0 / eta)
    
    # Average over batch
    grad = torch.mean(grad_estimates, dim=0, keepdim=True)
    
    return grad

def run_optimization(env_name, method, batch_size, iterations=100, device='cpu'):
    print(f"Running Optimization: Env={env_name}, Method={method}, B={batch_size}")
    
    # Load config
    config_path = f'./configs/env/{env_name}'
    if not os.path.exists(config_path):
         config_path = f'../configs/env/{env_name}'
    
    with open(config_path, 'r') as f:
        env_config = yaml.safe_load(f)
        
    # Initialize L (Buffer Sizes)
    # Start with a reasonable value, e.g., 20
    q = len(env_config['h'])
    init_val = 20.0
    L = torch.tensor([[init_val] * q], device=device, requires_grad=True)
    
    # Optimization Loop (Sign SGD)
    lr = 1.0 # Learning rate for integer steps
    
    history = []
    
    for t in tqdm(range(iterations)):
        if method == 'PATHWISE':
            grad = compute_pathwise_grad_L(env_config, L, T=100, device=device)
        elif method == 'SPSA':
            # Detach L for SPSA as we don't need autograd
            grad = compute_spsa_grad_L(env_config, L.detach(), T=100, batch_size=batch_size, device=device)
        
        # Sign SGD Update (Eq 16)
        # L_{t+1} = L_t - sign(grad)
        with torch.no_grad():
            update = torch.sign(grad)
            L -= lr * update
            
            # Constraint: L >= 1
            L.clamp_(min=1.0)
            
            # For logging, we might want to evaluate the "true" cost of current L
            # But that's expensive. We'll just log the L values.
            history.append(L.clone().detach().cpu().numpy())
            
            # Reset grad
            if L.grad is not None:
                L.grad.zero_()
                
    return L.detach(), history

def final_evaluation(env_name, L_final, device='cpu'):
    """
    Rigorous evaluation: N=10^4, 100 trajectories
    """
    config_path = f'./configs/env/{env_name}'
    if not os.path.exists(config_path):
         config_path = f'../configs/env/{env_name}'
    with open(config_path, 'r') as f:
        env_config = yaml.safe_load(f)
        
    # Run long simulation
    cost = simulate_trajectory(env_config, L_final, T=10000, batch_size=100, device=device)
    return cost.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    # Environments to test
    # Assuming reentrant_2.yaml exists. 
    # If reentrant_1 doesn't exist, we'll use reentrant_2 and another one.
    envs = ['reentrant_2.yaml', 'reentrant_3.yaml'] 
    
    results = {}
    
    for env_name in envs:
        results[env_name] = {}
        
        # 1. PATHWISE (B=1)
        L_final_pw, _ = run_optimization(env_name, 'PATHWISE', batch_size=1, device=args.device)
        cost_pw = final_evaluation(env_name, L_final_pw, device=args.device)
        results[env_name]['PATHWISE_B1'] = {'L': L_final_pw.cpu().tolist(), 'cost': cost_pw}
        
        # 2. SPSA (B=10, 100, 1000)
        for b in [10, 100, 1000]:
            L_final_spsa, _ = run_optimization(env_name, 'SPSA', batch_size=b, device=args.device)
            cost_spsa = final_evaluation(env_name, L_final_spsa, device=args.device)
            results[env_name][f'SPSA_B{b}'] = {'L': L_final_spsa.cpu().tolist(), 'cost': cost_spsa}
            
    # Print Results
    print("\n=== Final Comparative Results (Figure 11 Replication) ===")
    print(json.dumps(results, indent=4))
    
    # Save to file
    os.makedirs('./results', exist_ok=True)
    with open('./results/admission_control_results.json', 'w') as f:
        json.dump(results, f, indent=4)
