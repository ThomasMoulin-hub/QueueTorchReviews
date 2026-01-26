import numpy as np
from tqdm import trange
import torch
from torch import nn
import torch.nn.functional as F
from queuetorch.env import *
import json
import torch.distributions.one_hot_categorical as one_hot_sample
import yaml


if __name__ == '__main__':

    mw_costs = {}
    envs = ['reentrant_', 're-reentrant_']
    hypers = ['', '_hyper']

    for env in envs:
        for hyper in hypers:
            for L in range(2,3):

                name = f'{env}{L}{hyper}'
                print(name)
                with open(f'./configs/env/{name}.yaml', 'r') as f:
                    env_config = yaml.safe_load(f)    

                dq = load_env(env_config, temp = 1e-6, batch = 100, seed = 42, device = 'cpu')
                A = dq.queue_event_options[dq.q:]

                obs, state = dq.reset(seed = 23942934)

                total_cost = torch.tensor([[0.]]*dq.batch)
                for _ in trange(300000):

                    queues, time = obs

                    # c mu rule
                    
                    #pr = F.one_hot(torch.argmax(dq.mu*dq.h * 1.*(queues > 0.).unsqueeze(1), dim = 2), num_classes = dq.q)
                    #logits = dq.mu * torch.sum(-A.unsqueeze(0).repeat(dq.batch,1,1) * queues.unsqueeze(1) * dq.h.unsqueeze(1), 2).unsqueeze(1)
                    #pr = F.softmax(logits, -1)
                    pr = F.one_hot(torch.argmax(dq.mu*dq.h * queues.unsqueeze(1), dim = 2), num_classes = dq.q)
                    #pr = F.one_hot(torch.argmax(pr * dq.network, dim = 2), num_classes = dq.q)
                    
                    pr = torch.minimum((pr * dq.network), queues.unsqueeze(1).repeat(1, dq.s, 1))
                    pr += 1*torch.all(pr == 0., dim = 2).reshape(dq.batch,dq.s,1) * dq.network
                    pr /= torch.sum(pr, dim = -1).reshape(dq.batch, dq.s, 1)

                    action = one_hot_sample.OneHotCategorical(probs = pr).sample()
                    obs, state, cost, event_time = dq.step(state, action)
                    total_cost += cost

                mw_costs[name] = {'avg_cost': float(torch.mean(total_cost/state.time)),
                                'std_error': (2/np.sqrt(100)) * float(torch.std(total_cost/state.time))}
                
                with open(f'./PPO/cmu_results.json', 'w') as json_file:
                    json.dump(mw_costs, json_file)
