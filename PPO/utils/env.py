import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import torch.optim as optim
import gym
import sys
import itertools
from typing import NamedTuple
import pdb
import os
import gym
from gym import spaces
from stable_baselines3 import PPO
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gym
import argparse
import math
import os
import torch.optim as optim
import matplotlib.pyplot as plt

from typing import NamedTuple


class Obs(NamedTuple):
    queues: torch.Tensor
    time: torch.Tensor

class EnvState(NamedTuple):
    queues: torch.Tensor
    time: torch.Tensor
    service_times: torch.Tensor
    arrival_times: torch.Tensor

class STargmin(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.softmax = nn.Softmax(dim = 0)
    
    def forward(self, x):
        return F.one_hot(torch.argmin(x), num_classes = x.size()[1]) - self.softmax(-x/self.temp).detach() + self.softmax(-x/self.temp)
    
def allocator(action, mu, queue_service_times):
    
    adj_const = action.clone().detach()
    adj_const[adj_const < 1] = 1

    mu_with_grad = mu * action / adj_const
    a = mu * action
    num_q = a.size()[-1]
    
    # identify non-zero actions
    nonzero_inds = a[0].detach().numpy().nonzero()
    nonzero_inds = np.transpose(nonzero_inds).tolist()
    
    # for each queue, identify servers with positive action
    queue_nonzero_inds = {i:[] for i in range(num_q)}
    for ind in nonzero_inds:
        if ind[1] in queue_nonzero_inds.keys():
            for _server in range(int(torch.round(action[0][ind[0]][ind[1]]).item())):
                queue_nonzero_inds[ind[1]].append(ind)
    
    
    # build allocated action
    allocated_a = []
    num_allocated = []
    for q in range(num_q):
        q_allocated_a = []

        # sort the indices by work
        queue_nonzero_inds[q].sort(key = lambda x: float(mu_with_grad.detach()[0][x[0]][x[1]]), reverse = True)

        num_allocated_jobs = min(len(queue_service_times[q]), len(queue_nonzero_inds[q]))
        num_allocated.append(num_allocated_jobs)

        for j in range(num_allocated_jobs):
            # allocate job j in queue q to the jth non-empty server
            q_server = queue_nonzero_inds[q][j]
            q_allocated_a.append(mu_with_grad[0][q_server[0]][q_server[1]])
        
        # if num_allocated_jobs < len(queue_service_times[q]):
        #     num_unallocated_jobs = len(queue_service_times[q]) - num_allocated_jobs
        #     q_allocated_a = q_allocated_a + [torch.tensor(0) for _ in range(num_unallocated_jobs)]

        allocated_a.append(q_allocated_a)
    
    # print(queue_nonzero_inds)
    return allocated_a, queue_nonzero_inds, num_allocated
            
class P_DiffDiscreteEventSystem(gym.Env):
    def __init__(self, network, mu, h, draw_service, draw_inter_arrivals, init_time = 0, batch = 1, queue_event_options = None,
                 straight_through_min = False,
                 queue_lim = None, temp = 1, seed = 3003,
                 device = "cpu", f_hook = False, f_verbose = False, reset = False):
        

        self.device = device
        # print(f'batch size: {batch}')
        self.state = np.random.default_rng(seed)
        self.network = network.repeat(batch,1,1).to(self.device)
        self.mu = mu.repeat(batch,1,1).to(self.device)
        self.q = self.network.size()[-1]
        self.s = self.network.size()[-2]
        self.h = torch.tensor(h).float().to(device)
        self.temp = temp
        self.st_argmin = STargmin(temp = self.temp)
        self.f_hook = f_hook
        self.f_verbose = f_verbose
        self.straight_through_min = straight_through_min
        self.batch = batch
        self.seed = seed

        self.eps = 1e-8
        self.inv_eps = 1/self.eps
        
        if queue_event_options is None:
            self.queue_event_options = torch.cat((F.one_hot(torch.arange(0,self.q)), -F.one_hot(torch.arange(0,self.q)))).float().to(self.device)
        else:
            self.queue_event_options = queue_event_options.float().to(self.device)
        
        # self.queues = init_queues.float().to(self.device)
        self.free_servers = torch.ones((self.batch, self.s)).to(self.device)
        self.cost = torch.tensor([0]).to(self.device)

        if isinstance(init_time, torch.Tensor):
            self.time_elapsed = init_time.float().to(self.device)
        else:
            self.time_elapsed = torch.tensor([0.]).to(self.device)

        self.time_weight_queue_len = torch.zeros(self.network.size()[-1]).to(self.device)
        self.queue_len_dist = {}
        self.marg_queue_len_dist = [{} for _ in range(self.q)]
        self.terminated = False

        self.draw_service_core = draw_service
        self.draw_inter_arrivals_core = draw_inter_arrivals


        self.reset(time=self.time_elapsed, seed=seed)

    def draw_service(self, time):
        return self.draw_service_core(self, time)
    
    def draw_inter_arrivals(self, time):
        return self.draw_inter_arrivals_core(self, time)
    
        
    def reset(self, init_queues=None, time=None, seed = None, options:dict = None):
        #self.episode += 1
        cost = torch.tensor([0]).to(self.device)
        if time is None:
            time = torch.tensor([[0.]]).repeat(self.batch, 1).to(self.device)
        else:
            time = time.repeat(self.batch).unsqueeze(1).to(self.device)

        if init_queues is None:
            queues = torch.tensor([[0.]*self.q]).repeat(self.batch, 1).to(self.device)
        elif init_queues.size()[0] == 1:
            queues = init_queues.float().repeat(self.batch, 1).to(self.device)
        else:
            queues = init_queues.float().to(self.device)
        
        seed = self.seed
        if seed is not None:
            self.state = np.random.RandomState(seed)

        service_times = [[self.draw_service(time) for _ in range(int(queues_sample[q]))] for q in range(self.q) for queues_sample in queues]
        arrival_times = self.draw_inter_arrivals(time)

        self.obs = Obs(queues, time)
        self.env_state = EnvState(queues, time, service_times, arrival_times)
        
        return queues.cpu().numpy(), {}

    def step(self, action):
        
        # Compliance with network
        state = self.env_state
        queues, time, service_times, arrival_times = state    
        # print(queues)    
        # print(f'step batch size: {self.batch}')
        action = torch.tensor(action).float().to(self.device)
        action = action * self.network
        # print(f'network shape: {self.network.size()}')
        

        # action is zero if queues are zero
        #if self.f_preemptive:
        # print(f'action shape: {action.size()}')
        # print(f'queues shape: {queues.size()}')
        # action = torch.minimum(action, queues)
        action = torch.minimum(action, queues.unsqueeze(1).repeat(1,self.s,1))

        # work is action times mu

        # allocate work to jobs
        #allocated_work = work
        allocated_work, queue_nonzero_inds, num_allocated = allocator(action, self.mu, service_times)
        # print(allocated_work)


        if self.f_verbose:
            print(f"action:\t{action}")
            print(f"allocated work:\t{allocated_work}")
            print(f"queue_nonzero_inds:\t{queue_nonzero_inds}")

        # print(queues)
        # print(action)
        
        # effective service times are service_time divided by mu
        #eff_service_times = torch.stack([torch.min(torch.stack(self.service_times[q]) / torch.stack(allocated_work[q])) for q in range(self.q)])
        
        
        eff_service_times = [torch.tensor([self.inv_eps])]*self.q
        # print(f"eff service times shape: {torch.tensor(eff_service_times).size()}")
        for q in range(self.q):
            if num_allocated[q] > 0:
                eff_service_times[q] = torch.stack(service_times[q][:num_allocated[q]])[:,0,q] / torch.clip(torch.stack(allocated_work[q]), min = self.eps)
                #eff_service_times[q] = torch.stack(self.service_times[q])[:num_allocated[q]] / allocated_work[q]
        
        min_eff_service_times = torch.stack([torch.min(eff_service_times[q]) for q in range(self.q)])
        min_eff_service_times = min_eff_service_times.unsqueeze(0)
        
        
        # arrival times and service times are both q vectors
        # print(f"arrival times shape: {arrival_times.size()}")
        # print(f"min eff service times shape: {min_eff_service_times.size()}")
        event_times = torch.cat((arrival_times, min_eff_service_times), dim=1).float()

        if self.f_verbose:
            print(f"service:\t\t{service_times}")
            print(f"eff service:\t\t{eff_service_times}")
            print(f"eff service:\t\t{min_eff_service_times}")
            print(f"event times:\t\t{event_times}")
            print()

        # if a job was served, which job in which queue
        if True:
        # with torch.no_grad():
            which_job = [0]*self.q
            for q in range(self.q):
                if num_allocated[q] > 0:
                    which_job[q] = int(torch.argmin(torch.stack(service_times[q][:num_allocated[q]]).detach()[:,0,q] / torch.stack(allocated_work[q])).detach())
                    #which_job[q] = int(torch.argmin(torch.stack(self.service_times[q])[:num_allocated[q]].detach().squeeze(1) / allocated_work[q]).detach())

            which_queue = int(torch.argmin(min_eff_service_times).detach())
            
            if self.f_verbose:
                print(f"which_queue:\t\t{which_queue}")
                print(f"which_job:\t\t{which_job}")
                #print(f"which_arrival:\t\t{which_arrival}")
                print()
        
        
        # outcome is one_hot argmin of the event times
        outcome = self.st_argmin(event_times)
        
        
        # update state based on event time
        delta_q = torch.matmul(outcome, self.queue_event_options)
        
        # compute min event
        if not self.straight_through_min:
            event_time = torch.min(event_times)
        else:
            event_time = torch.sum(event_times * outcome)

        # if self.f_verbose:
        #     print(f"outcome:\t\t{outcome}")
        #     print(f"delta_q:\t\t{delta_q}")
        
        # if self.f_hook:
        #     if outcome.requires_grad:
        #         event_times.register_hook(lambda grad: print(f"event_times: {grad}"))
        #         outcome.register_hook(lambda grad: print(f"outcome_grad: {grad}"))
        #         event_time.register_hook(lambda grad: print(f"event time grad: {grad}"))
        #         delta_q.register_hook(lambda grad: print(f"delta grad: {grad}"))
        
        # # update joint state dist: state is concatenated string
        # with torch.no_grad():
        #     state_record = self.queues.data.numpy().astype("int")
        #     joint_state_key = tuple(state_record)
        #     if joint_state_key in self.queue_len_dist.keys():
        #         self.queue_len_dist[joint_state_key] += float(event_time.data.numpy())
        #     else:
        #         self.queue_len_dist[joint_state_key] = float(event_time.data.numpy())
            
        #     # update marginal state dist:
        #     for qu, qu_len in enumerate(state_record):
        #         if qu_len in self.marg_queue_len_dist[qu].keys():
        #             self.marg_queue_len_dist[qu][int(qu_len)] += float(event_time.data.numpy())
        #         else:
        #             self.marg_queue_len_dist[qu][int(qu_len)] = float(event_time.data.numpy())

        # time weighted queue length
        self.time_weight_queue_len = self.time_weight_queue_len + event_time * queues
        
        # update time elapsed, cost, queues
        time = time + event_time
        cost = torch.matmul(event_time * queues, self.h)
        
        
        queues = F.relu(queues + delta_q)
        # pdb.set_trace()

        
        
        if self.f_verbose:
            print(f"event_time:\t\t{event_time}")
            #print(f"eff_elapsed_time:{allocated_work * event_time}")
            print()
        
        # update service times for all jobs with positive work
        # for q in range(self.q):
        #     for j in range(num_allocated[q]):
        #         self.service_times[q][j] = F.relu(self.service_times[q][j] - allocated_work[q][j] * event_time)



        for q in range(self.q):
            if num_allocated[q] > 0:
                service_times[q][:num_allocated[q]] = list(torch.unbind(torch.stack(service_times[q][:num_allocated[q]]) - event_time * torch.stack(allocated_work[q]).unsqueeze(1).unsqueeze(-1).repeat((1,1,self.network.shape[-1]))))
                # service_times[q][:num_allocated[q]] = list(torch.unbind(torch.stack(service_times[q][:num_allocated[q]]) - event_time.detach() * torch.stack(allocated_work[q]).detach().unsqueeze(1).unsqueeze(-1).repeat((1,1,self.network.shape[-1]))))
        # update arrival times
        
        arrival_times = arrival_times - event_time

        if self.f_verbose:
            print(f"new service times:\t\t{service_times}")
            print(f"new arrival times:\t\t{arrival_times}")
            print()

        # Reset timers and add service
        # with torch.no_grad():
        if True:

            delta = delta_q.data.int()
            delta_arrived = torch.where(delta == 1, 1, 0)
            delta_left = torch.where(delta == -1, 1, 0)

            if torch.sum(delta != 0) == 0:
                arrival_times[0,torch.argmax(outcome)] = arrival_times[0,torch.argmax(outcome)] + 1e8
            # if a new job arrives
            if torch.sum(delta_arrived) > 0:
                # arrival occurs
                new_arrival_times = self.draw_inter_arrivals(time)
                new_service_time = self.draw_service(time)                
                
                # new arrival counter
                if torch.sum(delta_arrived) == 1:
                    arrival_times = arrival_times + torch.nan_to_num((new_arrival_times) * delta_arrived, nan = self.inv_eps)

                which_arrival = torch.argmax(delta_arrived)
                
                # service time of the new arrival
                service_times[which_arrival].append(new_service_time)

                if self.f_verbose:
                    print('Arrival!')
                    print(f"new service times:\t\t{service_times}")
                    print(f"new arrival times:\t\t{arrival_times}")
                    print()
            
            if torch.sum(delta_left) > 0:
                # remove a served job
                service_times[which_queue][which_job[which_queue]] = service_times[which_queue][which_job[which_queue]].detach()
                popped_job = service_times[which_queue].pop(which_job[which_queue])
                del popped_job

                if self.f_verbose:
                    print('Service!')
                    #print(f"popped job:\t\t{popped_job}")
                    print(f"service times:\t\t{service_times}")
                    print()

        
        next_state = EnvState(queues, time, service_times, arrival_times)
        obs = Obs(queues, time)

        self.env_state = next_state
        self.obs = obs

        done = False
        truncated = False
        reward = -cost

        info = {"obs": obs, "state": next_state, "cost": cost, "event_time": event_time, "queues": queues}

        return queues.cpu().numpy(), reward.cpu().numpy(), done, truncated, info
    
    def get_observation(self):
        return self.queues
        
    def print_state(self):
        print(f"Total Cost:\t{self.cost}")
        print(f"Time Elapsed:\t{self.time_elapsed}")
        print(f"Queue Len:\t{self.queues}")
        print(f"Service times:\t{self.service_times}")
        print(f"Arrival times:\t{self.arrival_times}")
        # else:
        #     print(f"Work:\t{self.work}")

class sbPGymDiffDiscreteEventSystem(P_DiffDiscreteEventSystem):
    """
    This subclass of GymDiffDiscreteEventSystem is compatible with Stable Baselines 3.
    """
    def __init__(self, 
                 network:torch.Tensor, 
                 mu:torch.Tensor, 
                 h:torch.Tensor,
                 draw_service,
                 draw_inter_arrivals,
                 init_time,
                 queue_event_options = None,
                 straight_through_min = False,
                 batch:int = 1, 
                 temp:float = 1,
                 seed:int = 3003,
                 device:torch.device = torch.device('cpu'), 
                 f_hook:bool = False, 
                 f_verbose:bool = False,
                 time_f = False,
                 reward_scale = 1.0,
                 policy_name = 'softmax',
                 action_map = None,
                 ):
        
        super().__init__(network = network,
                            mu = mu,
                            h = h,
                            draw_service= draw_service,
                            draw_inter_arrivals = draw_inter_arrivals,
                            init_time = init_time,
                            queue_event_options = queue_event_options,
                            straight_through_min = straight_through_min,
                            batch = batch,
                            temp = temp,
                            seed = seed,
                            device = device,
                            f_hook = f_hook,
                            f_verbose = f_verbose
                            )
        self.time_f = time_f
        self.policy_name = policy_name
        self.action_map = action_map

        process_id = os.getpid()
        parent_process_id = os.getppid()
        print(f"Environment initialized in process: {process_id}, parent process: {parent_process_id}")

        if self.policy_name == 'WC' or self.policy_name == 'vanilla':
            self.action_space = spaces.Box(low=0, high=1, shape=(self.s, self.q), dtype=np.float32)
        elif self.policy_name == 'discrete':
            self.action_space = spaces.Discrete(len(self.action_map))
            # print all actions in action map:
            # self.action_space = spaces.MultiDiscrete([self.q] * self.s)
            # action_space = spaces.Box(low=0, high=1, shape=(self.s, self.q), dtype=np.float32)
            #self.action_space = spaces.Box(low=0, high=1, shape=(self.s, self.q), dtype=np.float32)

        if time_f:
            self.observation_space = spaces.Box(
                low=0, 
                high=np.inf, 
                shape=(self.q + 1,),  # Add 1 for the time
                dtype=np.float32
            )
        
        else:
            self.observation_space = spaces.Box(
                low=0, 
                high=np.inf, 
                shape=(self.q,),  
                dtype=np.float32
            )

        self.obs = None
        self.reward_scale = reward_scale


def load_sb_p_env(env_config, temp, batch, seed, policy_name, device):

    name = env_config['name']

    if 'env_type' in env_config:
        env_type = env_config['env_type']
    else:
        env_type = name

    if env_config['network'] is None:
        env_config['network'] = np.load(f'./env_data/{name}/{name}_network.npy')
    env_config['network'] = torch.tensor(env_config['network']).float()


    if env_config['mu'] is None:
        env_config['mu'] = np.load(f'./env_data/{name}/{name}_mu.npy')
    env_config['mu'] = torch.tensor(env_config['mu']).float()

    orig_s, orig_q = env_config['network'].size()


    network = env_config['network'].repeat_interleave(1, dim = 0)
    mu = env_config['mu'].repeat_interleave(1, dim = 0)
    # if 'server_pool_size' in env_config.keys():
    #     env_config['server_pool_size'] = torch.tensor(env_config['server_pool_size']).to(model_config['env']['device'])
    # else:
    #     env_config['server_pool_size'] = torch.ones(orig_s).to(model_config['env']['device'])

    queue_event_options = env_config['queue_event_options']
    if queue_event_options is not None:
        if queue_event_options == 'custom':
            queue_event_options = torch.tensor(np.load(f'./env_data/{env_type}/{env_type}_delta.npy'))
        else:
            queue_event_options = torch.tensor(queue_event_options)



    lam_type = env_config['lam_type']
    lam_params = env_config['lam_params']
    h = torch.tensor(env_config['h'])

    if lam_params['val'] is None:
        lam_r = np.load(f'./env_data/{name}/{name}_lam.npy')
    else:
        lam_r = lam_params['val']
    def lam(t):
            if lam_type == 'constant':
                lam = lam_r
            elif lam_type == 'step':
                is_surge = 1*(t.data.cpu().numpy() <= lam_params['t_step'])
                lam = is_surge * np.array(lam_params['val1']) + (1 - is_surge) * np.array(lam_params['val2'])
            else:
                return 'Nonvalid arrival rate'
            
            return lam
        

    if env_config['queue_event_options'] == 'custom':
        env_config['queue_event_options'] = torch.tensor(np.load(f'./env_data/{name}/{name}_delta.npy'))


    def draw_inter_arrivals(self, time):

        def inter_arrival_dists(state, batch, t):
            exps = state.exponential(1, (batch, orig_q))
            lam_rate = lam(t)
            return exps / lam_rate

        interarrivals = torch.tensor(inter_arrival_dists(self.state, self.batch, time)).to(self.device)
        return interarrivals
    
    def draw_service(self, time):
        def service_dists(state, batch, t):
            return state.exponential(1, (batch, orig_q))
        service = torch.tensor(service_dists(self.state, self.batch, time)).to(self.device)
        return service

    # def draw_service(self, time):
    #     def service_dists(state, batch, t):
    #         return state.exponential(1, (batch, orig_q))
    #     service = torch.tensor(service_dists(self.state, self.batch, time)).to(self.device)
    #     return service

    # rho = 1.0
    # exp_arrival_1 = lambda state, t: state.exponential(1/(2.4*rho)) if t <= 100 else state.exponential(1/(0.4*rho))
    # exp_arrival_2 = lambda state, t: state.exponential(1/(0.6*rho)) if t <= 100 else state.exponential(1/(0.8*rho))
    # exp_arrival_3 = lambda state, t: state.exponential(1/(0.8*rho))
    # exp_arrival_4 = lambda state, t: state.exponential(1/(1.6*rho)) if t <= 100 else state.exponential(1/(0.8*rho))
    # exp_arrival_5 = lambda state, t: state.exponential(1/(0.6*rho))

    # def draw_inter_arrivals(self, time):

    #     interarrivals = torch.tensor([[exp_arrival_1(self.state, time), exp_arrival_2(self.state, time), exp_arrival_3(self.state, time), exp_arrival_4(self.state, time), exp_arrival_5(self.state, time)] for _ in range(self.batch)]).to(self.device)
        
    #     return interarrivals

    dq = sbPGymDiffDiscreteEventSystem(network, mu, h, 
                                       draw_service= draw_service, draw_inter_arrivals = draw_inter_arrivals, init_time = 0, 
                                    queue_event_options= queue_event_options,
                                    batch = batch, 
                                    temp = temp, seed = seed,
                                    time_f = False,
                                    reward_scale = 1.0,
                                    policy_name= policy_name,
                                    action_map = None,
                                    device = torch.device(device))

    return dq