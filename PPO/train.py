import sys
sys.path.append('../')
sys.path.append('..')
import gym
from gym import spaces
import numpy as np
import multiprocessing
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
from tqdm import trange
import torch
from torch import nn
import torch.nn.functional as F
import gym
import argparse
import math
import os
import torch.optim as optim

import json
import yaml
import json
import matplotlib.pyplot as plt
from utils.env import load_sb_p_env
# from utils.softmax_policy import Softmax_Policy
# from utils.vanilla_policy import Vanilla_Policy
from utils.policy_general import *
from utils.eval import parallel_eval
from utils.trainer import CustomPPOTrainer
from utils.rollout_buffer import CustomRolloutBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
# from queuetorch.env import *


import os
num_cpus = multiprocessing.cpu_count()
np.set_printoptions(precision=2)

def main():
    config_file_name = sys.argv[1]  
    env_config_name = sys.argv[2]
    print(f'config_file_name: {config_file_name}')

    if not config_file_name.endswith('.yaml'):
        config_file_name += '.yaml'

    config_file_path = './configs/' + config_file_name
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f'env_config: {env_config_name}')

    with open(f'../configs/env/{env_config_name}.yaml', 'r') as f:
        env_config = yaml.safe_load(f)

    name = env_config['name']
    print(f'name: {name}')
    
    if 'env_type' in env_config:
        env_type = env_config['env_type']
    else:
        env_type = name

    ## Environment Parameters
    # load network
    if env_config['network'] is None:
        network = np.load(f'./env_data/{env_type}/{env_type}_network.npy')
    else:
        network = env_config['network']

    print(f'network: {network}')

    if 'scale_factor' in env_config:
        scale_factor = env_config['scale_factor']
    else:
        scale_factor = 1

    # load mu
    if env_config['mu'] is None:
        mu = np.load(f'./env_data/{env_type}/{env_type}_mu.npy')
    else:
        mu = env_config['mu']

    network = torch.tensor(network).float()
    print(f'network: {network}')
    mu = torch.tensor(mu).float()
    print(f'mu: {mu}')

    orig_s, orig_q = network.size()

    # repeat if server pools
    num_pool = env_config['num_pool']
    network = network.repeat_interleave(num_pool, dim = 0)
    mu = mu.repeat_interleave(num_pool, dim = 0)

    queue_event_options = env_config['queue_event_options']
    if queue_event_options is not None:
        if queue_event_options == 'custom':
            queue_event_options = torch.tensor(np.load(f'./env_data/{env_type}/{env_type}_delta.npy'))
        else:
            queue_event_options = torch.tensor(queue_event_options)

    h = torch.tensor(env_config['h'])

    # arrival and service rates
    lam_type = env_config['lam_type']
    lam_params = env_config['lam_params']

    if lam_type == 'constant':
        if lam_params['val'] is None:
            lam_r = np.load(f'./env_data/{env_type}/{env_type}_lam.npy')
        else:
            lam_r = lam_params['val']

        def lam(rng, t, batch, lam_r):
            return lam_r
        arrival_rates = lambda rng, t, batch: lam(rng, t, batch, lam_r = lam_r)
    elif lam_type == 'step':
        def lam(rng, t, step, p, scale, val1, val2):
            if not rng:
                is_surge = 1*(t <= step)
                init_lam = is_surge * np.array(val1) + (1 - is_surge) * np.array(val2)
                return init_lam
            else:
                is_surge = 1*(t.detach().cpu().numpy() <= step)
                init_lam = is_surge * np.array(val1) + (1 - is_surge) * np.array(val2)
                switch = rng.binomial(1, p)
                return switch * (init_lam / (1 + scale)) + (1 - switch) * (init_lam / (1 - scale))
        arrival_rates = lambda rng, t, batch: lam(rng, t, step = lam_params['t_step'], 
                                                          p = 0.5, scale = lam_params['scale'],
                                                          val1 = lam_params['val1'], val2 = lam_params['val2'])
    elif lam_type == 'sawtooth':
        def lam(rng, t, batch, step, val1, val2):
            is_surge = 1*(np.floor((t.detach().cpu().numpy() / step)) % 2 == 0)
            return is_surge * np.array(val1) + (1 - is_surge) * np.array(val2)
        arrival_rates = lambda rng, t, batch: lam(rng, t, step = lam_params['t_step'], val1 = lam_params['val1'], val2 = lam_params['val2'])
    elif lam_type == 'hyper':
        if lam_params['val'] is None:
            lam_r = np.load(f'./env_data/{env_type}/{env_type}_lam.npy')
        else:
            lam_r = np.array(lam_params['val'])

        scale = lam_params['scale']

        def lam(rng, t, batch, p, lam_r, scale):
            if not rng:
                return lam_r
            else:
                lam_r = lam_r.reshape((1,len(lam_r))).repeat(batch, axis = 0)
                switch = rng.binomial(1, p, (batch, 1))
                return switch * (lam_r / (1 + scale)) + (1 - switch) * (lam_r / (1 - scale))
        arrival_rates = lambda rng, t, batch: lam(rng, t, batch, p = 0.5, lam_r = lam_r, scale = scale)
    else:
        print('Nonvalid arrival rate')
    
    def inter_arrival_dists(state, batch):
        exps = state.exponential(1, (batch, orig_q))
        return exps

    if 'service_type' in env_config:
        service_type = env_config['service_type']
    else:
        service_type = 'exp'

    def service_dists(state, batch, t):
        if service_type == 'exp':
            return state.exponential(1, (batch, orig_q))
        if service_type == 'lognormal':
            return state.lognormal(0, 1, (batch, orig_q)) / np.exp(1/2)
        if service_type == 'hyper':
            scale = 0.8
            coins = state.binomial(1,0.5, size = (batch, orig_q))
            a = state.exponential((1 + scale), (batch, orig_q))
            b = state.exponential((1 - scale), (batch, orig_q))
            return coins * a + (1 - coins) * b
        else:
            pass

    init_test_queues = torch.tensor([env_config['init_queues']]).float()

    # env hyperparameters
    device = config['env']['device']
    # use cuda
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_seed = config['env']['test_seed']
    train_seed = config['env']['train_seed']
    print(f'device: {device}')
    print(f'test_seed: {test_seed}')
    print(f'train_seed: {train_seed}')
    env_temp = config['env']['env_temp']
    straight_through_min = config['env']['straight_through_min']
    randomize = config['env']['randomize']
    time_f = config['env']['time_f']
    reward_scale = config['env']['reward_scale']
    policy_name = config['model']['policy_name']
    print(f'policy_name: {policy_name}')

    # training hyperparameters
    actors = config['training']['actors']
    normalize_advantage = config['training']['normalize_advantage']
    normalize_value = config['training']['normalize_value']
    normalize_reward = config['training']['normalize_reward']
    rescale_v = config['training']['rescale_v']
    truncation = config['training']['truncation']
    num_epochs = config['training']['num_epochs']
    amp_value = config['training']['amp_value']
    var_scaler = config['training']['var_scaler']
    per_iter_normal_obs = config['training']['per_iter_normal_obs']
    per_iter_normal_value = config['training']['per_iter_normal_value']

    # learning rates:
    lr = config['training']['lr']
    lr_policy = config['training']['lr_policy']
    lr_value = config['training']['lr_value']
    min_lr_policy = config['training']['min_lr_policy']
    min_lr_value = config['training']['min_lr_value']


    episode_steps = config['training']['episode_steps']
    gae_lambda = config['training']['gae_lambda']
    gamma = config['training']['gamma']
    target_kl = config['training']['target_kl']
    vf_coef = config['training']['vf_coef']
    ppo_batch_size = config['training']['batch_size']
    print(f'ppo_batch_size: {ppo_batch_size}')
    ppo_epochs = config['training']['ppo_epochs']
    train_batch = config['training']['train_batch']
    test_batch = config['training']['test_batch']
    clip_range_vf = config['training']['clip_range_vf']
    ent_coef = config['training']['ent_coef']
    bc = config['training']['behavior_cloning']

    # model hyperparameters:
    scale = config['model']['scale']

    # policy hyperparameters
    test_policy = config['policy']['test_policy']

    # total steps
    total_steps = num_epochs * episode_steps * actors
    eval_freq = episode_steps
    # reward_scale = reward_scale / episode_steps
    test_T = env_config['test_T']
    print('env_config_test_T', test_T)
    ############# Main Training Code: #############

    # Create a function that returns a new instance of the environment
    
    def make_env():
        return load_sb_p_env(env_config = env_config,
                       temp = env_temp, 
                       batch = 1,
                       seed = train_seed,
                       policy_name = policy_name,
                       device = torch.device(device))

    def make_test_env(seed):
        return load_sb_p_env(env_config = env_config,
                       temp = env_temp, 
                       batch = 1,
                       seed = seed,
                       policy_name = policy_name,
                       device = torch.device(device))
    
    # Train Env
    dq_raw = load_sb_p_env(env_config = env_config,
                       temp = env_temp, 
                       batch = 1,
                       seed = train_seed,
                       policy_name = policy_name,
                       device = torch.device(device))
    

    ### parallel training ###

    env_fns = [make_env for _ in range(actors)]
    # dq = SubprocVecEnv(env_fns, start_method='fork')
    raw_envs = [make_test_env(seed) for seed in range(train_seed, train_seed + actors)]
    dq = DummyVecEnv(env_fns)


    # Test Env
    dq_test_list = [make_test_env(seed) for seed in range(test_seed, test_seed + 100)]

    # model kwargs
    L = orig_q
    J = orig_s
    gmLJ = int(np.sqrt(L * J))
    pi_arch = [scale * L, scale * gmLJ, scale * J]
    vi_arch = [scale * L, scale * gmLJ, scale * J]
    print(f'pi_arch: {pi_arch}')



    # leakyrelu activation
    policy_kwargs = dict(
                    activation_fn=nn.Tanh,
                    network = network,
                    time_f = time_f,
                    randomize = randomize,
                    scale = scale,
                    rescale_v = rescale_v,
                    alpha = 0,
                    D = dq_raw.queue_event_options,
                    mu = mu,
                    net_arch=dict(pi=pi_arch, 
                                   vf=vi_arch))
    
    #target_kl = None
    # define sb model
    if policy_name == 'WC':
        policy = WCPolicy
    elif policy_name == 'vanilla':
        policy = VanillaPolicy
    # elif policy_name == 'discrete':
    #     policy = Discrete_Policy

    rollout_buffer_kwargs = dict(
                            q = orig_q,
                            normalize_advantage = normalize_advantage,
                            normalize_value = normalize_value,
                            normalize_reward = normalize_reward,
                            truncation = truncation,
                            var_scaler = var_scaler,
                            per_iter_normal_value = per_iter_normal_value,
    )

    model = CustomPPOTrainer(policy, dq, learning_rate=lr, lr_policy=lr_policy, lr_value=lr_value, min_lr_policy=min_lr_policy, amp_value = amp_value, min_lr_value=min_lr_value,n_steps=episode_steps, batch_size=ppo_batch_size, num_epochs = num_epochs, n_epochs=ppo_epochs, gamma=gamma, gae_lambda=gae_lambda, clip_range=0.2, clip_range_vf=clip_range_vf, normalize_advantage=normalize_advantage, raw_env = raw_envs, normalize_value = normalize_value, rescale_v = rescale_v, ent_coef=ent_coef, actors = actors, vf_coef=vf_coef, max_grad_norm=1.0, use_sde=False, sde_sample_freq=-1, rollout_buffer_class=CustomRolloutBuffer, rollout_buffer_kwargs=rollout_buffer_kwargs, target_kl=target_kl, stats_window_size=100, tensorboard_log=None, policy_kwargs=policy_kwargs, verbose=1, seed=None, device=device, _init_setup_model=True)
    # def eval call back

    eval_env = dq_test_list 
    eval_callback = parallel_eval(model = model, eval_env = eval_env, eval_freq = eval_freq, eval_t = test_T, test_policy = test_policy, test_seed = test_seed, init_test_queues = init_test_queues, test_batch = test_batch, device = device, num_pool = num_pool, time_f = time_f, randomize = randomize, policy_name = policy_name, per_iter_normal_obs = per_iter_normal_obs, env_config_name = env_config_name, bc = bc, verbose = 1)

    eval_callback.pre_train_eval()
                                     
    # Train model
    model.learn(total_timesteps=total_steps, log_interval=1, callback=eval_callback)

    test_cost_list = eval_callback.test_costs
    test_cost_std_list = eval_callback.test_costs_std
    queue_lengths_list = eval_callback.queue_lengths
    # final_cost_list = eval_callback.final_costs

    results = {'test_cost': test_cost_list, 
               'test_cost_std':test_cost_std_list,
               'queue_lengths': queue_lengths_list}
    
    # Store the lists
    if bc:
        with open(f"{policy_name}_bc_results.json", 'w') as f:
            json.dump(results, f)
    else:
        with open(f"{policy_name}_results.json", 'w') as f:
            json.dump(results, f)
    


if __name__ == '__main__':
    main()




