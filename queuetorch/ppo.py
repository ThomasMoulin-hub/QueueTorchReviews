import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PPOBuffer:
    def __init__(self, batch_size, steps, device):
        self.batch_size = batch_size
        self.steps = steps
        self.device = device
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_gae(self, next_value, gamma=0.99, gae_lambda=0.95):
        """
        Generalized Advantage Estimation
        """
        values = self.values + [next_value]
        rewards = self.rewards
        dones = self.dones
        
        advantages = []
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[t].float()
            delta = rewards[t] + gamma * values[t+1] * non_terminal - values[t]
            last_gae_lam = delta + gamma * gae_lambda * non_terminal * last_gae_lam
            advantages.insert(0, last_gae_lam)
            
        return torch.stack(advantages)

    def get_batch(self, advantages):
        states_q = torch.cat([s.queues for s in self.states], dim=0)
        states_t = torch.cat([s.time for s in self.states], dim=0)
        actions = torch.cat(self.actions, dim=0)
        log_probs = torch.cat(self.log_probs, dim=0)
        values = torch.cat(self.values, dim=0)
        returns = values + advantages
        
        return states_q, states_t, actions, log_probs, returns, advantages

def ppo_loss(actor, critic, states_q, states_t, actions, old_log_probs, returns, advantages, clip_ratio=0.2, entropy_coef=0.01):
    # Get current policy distribution and values
    probs = actor(states_q, states_t)
    
    # Mask invalid actions (optional, depending on implementation) but here we assume actor handles it or learns it
    # For PPO we need categorical distribution
    dist = torch.distributions.OneHotCategorical(probs=probs)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()
    
    # Value loss
    new_values = critic(states_q, states_t).squeeze(-1)
    value_loss = F.mse_loss(new_values, returns)
    
    # Policy loss
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    total_loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
    
    return total_loss, policy_loss, value_loss, entropy
