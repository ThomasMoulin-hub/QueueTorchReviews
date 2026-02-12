import torch
import numpy as np
import yaml
import argparse
import os
import json
from tqdm import tqdm, trange
from torch import nn
import torch.nn.functional as F
import sys
from collections import defaultdict
import pathos.multiprocessing as mp

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from queuetorch.env import load_env
# from queuetorch.policies import SoftPriorityPolicy, SoftMaxWeightPolicy, SoftMaxPressurePolicy
import torch.distributions.one_hot_categorical as one_hot_sample

# def get_policy(policy_type, s, q):
#     if policy_type == 'sPR':
#         return SoftPriorityPolicy(s, q)
#     elif policy_type == 'sMW':
#         return SoftMaxWeightPolicy(s, q)
#     elif policy_type == 'sMP':
#         return SoftMaxPressurePolicy(s, q)
#     else:
#         raise ValueError(f"Unknown policy type: {policy_type}")

class ValueNet(nn.Module):
    def __init__(self, q, layers, hidden_dim, x_stats = None, y_stats = None):
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
        
        # Input layer
        if self.x_stats is not None:
            x = (x - self.x_stats[0]) / self.x_stats[1]
            
        x = F.relu(self.input_fc(x))

        # Hidden layer
        for l in range(self.layers):
            x = F.relu(self.layers_fc[l](x))

        # Output layer
        x = self.output_fc(x)
        return x
    
def evaluate_iterate(priority, env_config, batch = 100, eval_T = 10000):

    dq = load_env(env_config, temp = 0.5, batch = batch, seed = 42, device = 'cpu')
    torch.manual_seed(42)
    obs, state = dq.reset(seed = 42)
    
    total_cost = torch.tensor([[0.]]*batch)

    for _ in trange(eval_T):

        queues, time = obs

        pr = F.softmax(priority.repeat(dq.batch,dq.s,1), -1)
        pr = F.one_hot(torch.argmax(pr * 1.*(queues > 0.).unsqueeze(1), dim = 2), num_classes = dq.q)
        pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
        pr += 1*torch.all(pr == 0., dim = 2).reshape(dq.batch,dq.s,1).repeat(1,1,dq.q) * dq.network
        pr /= torch.sum(pr, dim = -1).reshape(dq.batch, dq.s, 1) 
        
        action = one_hot_sample.OneHotCategorical(probs = pr).sample()

        #action = pr
        obs, state, cost, event_time = dq.step(state, action)
        total_cost += cost
    
    return float(torch.mean(total_cost / state.time))

def calculate_returns(rewards, discount_factor, normalize = True):
    
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

def pathwise_cmu(env_config, seed, num_iter, alpha, temp = 0.01, T = 1000, eval_T = 10000):

    dq = load_env(env_config, temp = temp, batch = 1, seed = seed, device = 'cpu')
    priority = torch.zeros((1,dq.q)).float()
    sum_priority = priority.clone()

    priority.requires_grad = True
    st_steps = [priority.detach()]
    avg_iterate = [sum_priority.clone()]
    num = 1

    # Optimization loop
    for i in trange(num_iter):
        # Calculate gradient
        dq = load_env(env_config, temp = temp, batch = 1, seed = seed, device = 'cpu')

        if i > 0:
            obs, state = dq.reset(seed = seed, init_queues = init_queues)
        else:
            obs, state = dq.reset(seed = seed)
        
        total_cost = torch.tensor([[0.]]*dq.batch)
    
        for _ in range(T):
            queues, time = obs
            pr = F.softmax(priority.repeat(dq.batch,dq.s,1), -1) * dq.network # soft_priority with only theta
            pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1)) # work_conserving
            pr += 1*torch.all(pr == 0., dim = 2).reshape(dq.batch,dq.s,1) * dq.network # deal with extreme scenarios where pr becomes all zero
            pr /= torch.sum(pr, dim = -1).reshape(dq.batch, dq.s, 1) 

            action = pr
            obs, state, cost, event_time = dq.step(state, action)
            total_cost += cost

        init_queues = queues.detach()
        avg_cost = torch.mean(total_cost / state.time)
        avg_cost.backward()

        normalized_grad = priority.grad / torch.linalg.norm(priority.grad)
        
        priority = priority.detach() - alpha * normalized_grad
        
        st_steps.append(priority.detach())
        sum_priority += priority.detach()
        num += 1
        avg_iterate.append(sum_priority.clone() / num)
        
        priority.requires_grad = True

    # Eval
    avg_cost = evaluate_iterate(avg_iterate[-1], env_config, eval_T = eval_T)

    return {'last_iterate':avg_iterate[-1].detach().tolist(),
            'avg_cost': avg_cost}
    


def reinforce_cmu(env_config, seed, num_iter, alpha,  T = 1000, gamma = 0.99, batch = 1000, policy_temp = 5, eval_T = 10000):
    
    dq = load_env(env_config, temp = 1.0, batch = 1, seed = seed, device = 'cpu')
    priority = torch.zeros((1,dq.q)).float()
    priority.requires_grad = True

    sum_priority = priority.detach().clone()
    reinforce_steps = [priority]
    reinforce_avg_iterate = [sum_priority.clone()]
    num = 1

    for i in trange(num_iter):
        # Calculate gradient
        
        dq = load_env(env_config, temp = 1.0, batch = batch, seed = seed, device = 'cpu')

        if i > 0:
            obs, state = dq.reset(seed = seed, init_queues = init_queues)
        else:
            obs, state = dq.reset(seed = seed)
            
        torch.manual_seed(8838383+i)
        
        log_prob_buffer = []
        costs = []
        state_buffer = []

        for _ in range(T):

            queues, time = obs
            #pr = F.softmax(policy_temp*priority.repeat(dq.batch,dq.s,1), -1) * dq.network

            pr = F.softmax(priority.repeat(dq.batch,dq.s,1), -1) * dq.network
            pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
            pr += 1*torch.all(pr == 0., dim = 2).reshape(dq.batch,dq.s,1) * dq.network
            pr /= torch.sum(pr, dim = -1).reshape(dq.batch, dq.s, 1) 

            pr_sample = one_hot_sample.OneHotCategorical(probs = pr)
            action = pr_sample.sample()

            log_prob = torch.sum(torch.log(torch.sum(action.detach() * pr, dim = 2)), dim = 1, keepdims = True)
            log_prob_buffer.append(log_prob)

            obs, state, cost, event_time = dq.step(state, action)
            costs.append(cost.detach().tolist())
            state_buffer.append(queues.detach())

        init_queues = queues.detach()
            
        cost_buffer = torch.tensor(costs)
        cost_buffer -= torch.mean(cost_buffer)
        policy_loss = torch.tensor(0.)

        disc_costs = []
        for t in range(len(cost_buffer)):
            sum_discounted_costs = (1-gamma)*torch.sum(cost_buffer[t:] * torch.tensor([gamma ** j for j in range(0, len(cost_buffer) - t)]).unsqueeze(1).unsqueeze(2), 0)
            policy_loss = policy_loss + (sum_discounted_costs) * log_prob_buffer[t]
            
        torch.mean(policy_loss).backward()
        
        normalized_grad = (priority.grad / torch.linalg.norm(priority.grad))
        #normalized_grad = priority.grad
        
        priority = priority.detach() - alpha * normalized_grad
        reinforce_steps.append(priority)
        
        sum_priority += priority.detach()
        num += 1
        reinforce_avg_iterate.append(sum_priority.clone() / num)
        
        priority.requires_grad = True

    # Eval
    avg_cost = evaluate_iterate(reinforce_avg_iterate[-1], env_config, eval_T = eval_T)

    return {'last_iterate':reinforce_avg_iterate[-1].detach().tolist(),
            'avg_cost': avg_cost}

def reinforce_value_cmu(env_config, seed, num_iter, alpha,  T = 1000, gamma = 0.99, batch = 1000, policy_temp = 5, eval_T = 10000):
    
    dq = load_env(env_config, temp = 1.0, batch = 1, seed = seed, device = 'cpu')
    priority = torch.zeros((1,dq.q)).float()
    priority.requires_grad = True

    sum_priority = priority.detach().clone()
    reinforce_steps = [priority]
    reinforce_avg_iterate = [sum_priority.clone()]
    num = 1

    for i in trange(num_iter):
        # Calculate gradient
        
        dq = load_env(env_config, temp = 1.0, batch = batch, seed = seed, device = 'cpu')

        if i > 0:
            obs, state = dq.reset(seed = seed, init_queues = init_queues)
        else:
            obs, state = dq.reset(seed = seed)
            
        torch.manual_seed(8838383+i)
        
        log_prob_buffer = []
        costs = []
        state_buffer = []

        for t in range(T):

            queues, time = obs
            
            pr = F.softmax(priority.repeat(dq.batch,dq.s,1), -1) * dq.network
            pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
            pr += 1*torch.all(pr == 0., dim = 2).reshape(dq.batch,dq.s,1) * dq.network
            pr /= torch.sum(pr, dim = -1).reshape(dq.batch, dq.s, 1) 

            pr_sample = one_hot_sample.OneHotCategorical(probs = pr)
            action = pr_sample.sample()
            
            # pr = F.softmax(policy_temp*priority.repeat(dq.batch,dq.s,1), -1) * dq.network
            # pr_sample = one_hot_sample.OneHotCategorical(probs = pr)
            # action = pr_sample.sample()

            log_prob = torch.sum(torch.log(torch.sum(action.detach() * pr, dim = 2)), dim = 1, keepdims = True)
            log_prob_buffer.append(log_prob)

            obs, state, cost, event_time = dq.step(state, action)
            costs.append(cost.detach().tolist())
            state_buffer.append(torch.hstack((queues, t*torch.ones(dq.batch).unsqueeze(1))))

        # Save last state
        init_queues = queues.detach()
        
        # Returns
        cost_buffer = torch.tensor(costs)
        returns = calculate_returns(cost_buffer, gamma)

        # Replay buffer
        all_states = torch.cat(state_buffer, axis = 0)
        all_returns = torch.cat(returns)

        # Optimize
        if i == 0:
            state_mean = all_states.mean(0)
            state_std = all_states.std(0)
            
            value_net = ValueNet(dq.q+1, 2, 64)
            value_net.x_stats = [state_mean, state_std]

            adam = torch.optim.Adam(value_net.parameters(), lr = 0.001)

        return_dataset = torch.utils.data.TensorDataset(all_states, all_returns)
        return_dataloader = torch.utils.data.DataLoader(return_dataset, batch_size=1024, shuffle=True)

        # Train Value network
        for epoch in trange(3):
            total_loss = torch.tensor(0.)
            for count, (state_batch, return_batch) in enumerate(return_dataloader):
                adam.zero_grad()
                out = value_net(state_batch)
                value_loss = F.mse_loss(out, return_batch, reduction = 'sum')
        
                value_loss.backward()
                adam.step()
        
                total_loss += value_loss.item()
        
            print(total_loss / len(return_dataset))
            
        # Update policy
        policy_loss = torch.tensor(0.)
        for t in trange(len(returns)):
            state = state_buffer[t]
            policy_loss = policy_loss + (returns[t] - value_net(state).detach()) * log_prob_buffer[t]

        # Policy Mean
        torch.mean(policy_loss).backward()
    
        normalized_grad = (priority.grad / torch.linalg.norm(priority.grad))
        
        priority = priority.detach() - alpha * normalized_grad
        reinforce_steps.append(priority)
        
        sum_priority += priority.detach()
        num += 1
        reinforce_avg_iterate.append(sum_priority.clone() / num)
        
        priority.requires_grad = True

    # Eval
    avg_cost = evaluate_iterate(reinforce_avg_iterate[-1], env_config, eval_T = eval_T)

    return {'last_iterate':reinforce_avg_iterate[-1].detach().tolist(),
            'avg_cost': avg_cost}



if __name__ == "__main__":

    num_cores = 100
    num_trials = 950
    num_iter = 50 # 20
    alphas = [0.01, 0.1, 0.5, 1.0]
    gaps = [1, 0.5, 0.05, 0.01] # [1, 0.5, 0.1, 0.05, 0.01]
    rho = 0.99 # 0.95
    
    queue_class = 5 # 10
    gamma = 0.99
    reinforce_batch = 100
    eval_T = 20000

    T = 1000 # horizon N
    
    #parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('-e', type=str)
    #args = parser.parse_args()

    with open(f'/user/xz3355/QueueTorchReviews/configs/env/multiclass.yaml', 'r') as f:
        env_config = yaml.safe_load(f)

    name = env_config['name']

    # seeds = [int.from_bytes(os.urandom(4), 'big') for _ in range(10000)]
    # with open(f'/user/xz3355/QueueTorchReviews/cmu/seeds_cmu_5class.json', 'w') as f:
    #     json.dump(seeds, f)
    with open(f'/user/xz3355/QueueTorchReviews/cmu/seeds_cmu_5class.json', 'r') as f:
        seeds = json.load(f)
    
    reinforce_results = defaultdict(lambda: defaultdict(list))


    # with open(f'/user/xz3355/QueueTorchReviews/cmu/pathwise_wc_cmu_{name}10.json', 'r') as f:
    #     raw = json.load(f)

    # for k, v in raw.items():
    #     for kk, vv in v.items():
    #         pathwise_results[k][kk] = vv
    
    for alpha in alphas:
        print(f'alpha: {alpha}')
        
        for gap in gaps:
            print(f'gap: {gap}')
                

            # Setup network
            env_config['init_queues'] = [0]*queue_class
            env_config['network'] = [[1]*queue_class]
            env_config['queue_event_options'] = np.vstack((np.eye(queue_class),-np.eye(queue_class))).tolist()
            env_config['h'] = [1]*queue_class

            # mu and lambda
            env_config['mu'] = np.array([[1 + gap*i for i in range(1, queue_class+1)]])
            env_config['lam_params']['val'] = np.repeat(rho/np.sum(1/env_config['mu']), queue_class)
            
            # Do jobs
            reinforce_jobs = []
            for i in range(50, 50+num_trials):
                
                reinforce_jobs.append({
                        'env_config': env_config,
                        'seed': seeds[i], 
                        'num_iter': num_iter,
                        'alpha': alpha,
                        'gamma': gamma,
                        'batch': reinforce_batch,
                        'eval_T': eval_T})

            reinforce_cmu_mp = lambda x: reinforce_value_cmu(**x)

            print(f'Reinforce - {alpha} - {gap}')
            reinforce = []
            with mp.ProcessingPool(num_cores) as pool:
                reinforce = pool.amap(reinforce_cmu_mp, reinforce_jobs)

            reinforce_out = reinforce.get()
            reinforce_results[str(alpha)][str(gap)] = reinforce_out

            with open(f'/user/xz3355/QueueTorchReviews/cmu/wc_reinforce_baseline_cmu_B100_{name}5_all_eps_950_more_runs.json', 'w') as f:
                json.dump(reinforce_results, f)
        
