#!/user/tmm2219/.conda/envs/qt_env/bin/python
import torch
import numpy as np
import yaml
import argparse
import os
import json
from tqdm import tqdm
import torch.nn.functional as F
import sys

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from queuetorch.env import load_env
from queuetorch.policies import SoftPriorityPolicy, SoftMaxWeightPolicy, SoftMaxPressurePolicy
import torch.distributions.one_hot_categorical as one_hot_sample

def get_policy(policy_type, s, q):
    if policy_type == 'sPR':
        return SoftPriorityPolicy(s, q)
    elif policy_type == 'sMW':
        return SoftMaxWeightPolicy(s, q)
    elif policy_type == 'sMP':
        return SoftMaxPressurePolicy(s, q)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

def _compute_reinforce_grad_core(net, env_config, batch_size, T, gamma, device):
    """
    Core function to compute REINFORCE gradient for a single chunk.
    """
    # Ensure gradients are zeroed before starting
    net.zero_grad()
    
    dq = load_env(env_config, temp=1.0, batch=batch_size, seed=None, device=device)
    obs, state = dq.reset()
    
    log_probs = []
    rewards = []
    
    # Collect trajectories
    for _ in range(T):
        queues, time = obs
        probs = net(queues)
        
        # Masking for validity
        probs = probs * dq.network
        probs = torch.minimum(probs, queues.unsqueeze(1).repeat(1, dq.s, 1))
        probs += 1 * torch.all(probs == 0., dim=2).reshape(batch_size, dq.s, 1).repeat(1, 1, dq.q) * dq.network
        probs /= torch.sum(probs, dim=-1, keepdim=True)
        
        dist = one_hot_sample.OneHotCategorical(probs=probs)
        action = dist.sample()
        
        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)
        
        obs, state, cost, event_time = dq.step(state, action)
        rewards.append(-cost.squeeze(1)) # Reward = -Cost

    # Compute returns
    policy_loss = 0
    R = torch.zeros(batch_size).to(device)
    
    # Backward pass for returns (Monte Carlo)
    for t in reversed(range(T)):
        R = rewards[t] + gamma * R
        # REINFORCE loss: - log_prob * Return
        policy_loss = policy_loss - log_probs[t] * R

    # Average loss over batch
    loss = policy_loss.mean()
    
    # Compute gradients
    loss.backward()
    
    grads = []
    for param in net.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1).detach().cpu())
        else:
            grads.append(torch.zeros_like(param.view(-1)).cpu())
        
    return torch.cat(grads)

def compute_reinforce_grad(net, env_config, batch_size, T, gamma=0.999, device='cpu'):
    """
    Computes the REINFORCE gradient estimator with mini-batching to avoid OOM.
    """
    MAX_CHUNK = int(10*500)  # Safe batch size for 16GB VRAM (conservative)
    
    if batch_size <= MAX_CHUNK:
        return _compute_reinforce_grad_core(net, env_config, batch_size, T, gamma, device)
    
    total_grads = None
    remaining = batch_size
    
    # Only show progress bar if it's a large batch (Ground Truth)
    disable_tqdm = batch_size < 2000
    pbar = tqdm(total=batch_size, desc="Computing Grad (Accumulating)", leave=False, disable=disable_tqdm)
    
    while remaining > 0:
        current_batch = min(remaining, MAX_CHUNK)
        
        # Compute gradients for this chunk
        grads = _compute_reinforce_grad_core(net, env_config, current_batch, T, gamma, device)
        
        # Weighted accumulation
        weight = current_batch / batch_size
        if total_grads is None:
            total_grads = grads * weight
        else:
            total_grads += grads * weight
            
        remaining -= current_batch
        pbar.update(current_batch)
        
        # Clear CUDA cache to ensure memory is freed
        if 'cuda' in device:
            torch.cuda.empty_cache()
            
    pbar.close()
    return total_grads

def compute_pathwise_grad(net, env_config, batch_size, T, device='cpu'):
    """
    Computes the Pathwise (STE) gradient estimator.
    """
    net.zero_grad()
    
    dq = load_env(env_config, temp=0.1, batch=batch_size, seed=None, device=device) # Low temp for STE approximation
    obs, state = dq.reset()
    
    total_cost = 0
    
    for _ in range(T):
        queues, time = obs
        probs = net(queues)
        
        # Masking
        probs = probs * dq.network
        probs = torch.minimum(probs, queues.unsqueeze(1).repeat(1, dq.s, 1))
        probs += 1 * torch.all(probs == 0., dim=2).reshape(batch_size, dq.s, 1).repeat(1, 1, dq.q) * dq.network
        probs /= torch.sum(probs, dim=-1, keepdim=True)
        
        action = probs # Direct usage for STE
        
        obs, state, cost, event_time = dq.step(state, action)
        total_cost = total_cost + cost

    loss = torch.mean(total_cost / T)
    
    loss.backward()
    
    grads = []
    for param in net.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1).detach().cpu())
        else:
            grads.append(torch.zeros_like(param.view(-1)).cpu())
        
    return torch.cat(grads)

def cosine_similarity(g1, g2):
    if torch.norm(g1) == 0 or torch.norm(g2) == 0:
        return 0.0
    return F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()

def run_experiment(args):
    device = args.device
    
    # Load base config
    config_path = f'./configs/env/{args.env}'
    if not os.path.exists(config_path):
        # Try finding it relative to project root if running from experiments folder
        config_path = f'../configs/env/{args.env}'
        
    with open(config_path, 'r') as f:
        env_config = yaml.safe_load(f)
        
    # Setup results storage
    results = []
    
    # Policy loop
    for policy_name in ['sPR', 'sMW', 'sMP']:
        print(f"Testing Policy: {policy_name}")
        
        # Random parameter sampling loop
        for i in tqdm(range(args.num_samples)):
            # Initialize policy with random weights
            temp_dq = load_env(env_config, temp=1, batch=1, seed=None, device='cpu')
            s, q = temp_dq.s, temp_dq.q
            
            net = get_policy(policy_name, s, q).to(device)
            
            # Save initial state dict to reset between runs if needed (though we create new envs)
            # But we need the SAME net weights for all 3 calculations
            
            # 1. Ground Truth (REINFORCE with huge batch)
            gt_grad = compute_reinforce_grad(net, env_config, batch_size=args.gt_batch, T=args.horizon, device=device)
            
            # 2. Pathwise Estimator (B=1)
            pw_grad = compute_pathwise_grad(net, env_config, batch_size=1, T=args.horizon, device=device)
            
            # 3. REINFORCE Estimator (B=1000)
            rf_grad = compute_reinforce_grad(net, env_config, batch_size=1000, T=args.horizon, device=device)
            
            # Calculate Similarities
            sim_pw = cosine_similarity(pw_grad, gt_grad)
            sim_rf = cosine_similarity(rf_grad, gt_grad)
            
            results.append({
                'policy': policy_name,
                'sample_idx': i,
                'sim_pathwise': sim_pw,
                'sim_reinforce': sim_rf
            })
            
    # Save results
    output_dir = './results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(f'{output_dir}/gradient_comparison.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_dir}/gradient_comparison.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='mm1.yaml')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--horizon', type=int, default=1000)
    parser.add_argument('--gt_batch', type=int, default=10000, help="Batch size for Ground Truth")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of random theta samples")
    parser.add_argument('--intensity', type=float, default=1.0)
    
    args = parser.parse_args()
    run_experiment(args)
