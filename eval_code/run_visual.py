# Copyright (c) 2020 Max Planck Gesellschaft

# !/usr/bin/env python
import sys
sys.path.append('../')
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
import pdb
#from baselines.ppo1 import mlp_policy
from eval_code import visual
from baselines import logger
import sys
import time

# This file is designed to visually evaluate the training results obtained in Simulation

def train(env_id, num_timesteps, seed, num_options,app, saves ,wsaves, epoch,dc,path,render,official,orig_ppo):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    # Episode len determines the length of the rollout. Remember: length of 20 means 1s.
    episode_len = 400
    from gym.envs.registration import register
    # add the current path to the repo -> we are loading exactly the repo it has been trained on!!!
    sys.path.append(path)
    print(sys.path)
    from src_code import mlp_policy
    # Depending on the environment argument, the right environment is selected
    if (env_id=='Pendulumnf-v0'):
        register(
            id='Pendulumnf-v0',
            entry_point='src_code.pendulum_nf:PendulumEnv',
            max_episode_steps=episode_len,
            #kwargs = vars(args),
        )
        env = gym.make('Pendulumnf-v0')
    # Potential Scalar Env
    elif (env_id=='Scalarnf-v0'):
        register(
            id='Scalarnf-v0',
            entry_point='src_code.gym_scalar_nf:GymScalarEnv',
            max_episode_steps=episode_len,
            #kwargs = vars(args),
        )
        env = gym.make('Scalarnf-v0')
    elif (env_id=='CartPole-v9'):
        register(
            id='CartPole-v9',
            entry_point='src_code.cartpole:CartPoleEnv',
            max_episode_steps=episode_len,
            #kwargs = vars(args),
        )
        env = gym.make('CartPole-v9')
    else:
        env = gym.make(env_id)

    # Create the policies needed
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, num_options=num_options, dc=dc) #was 64,32 or 15
    env = bench.Monitor(env, logger.get_dir() and 
        osp.join(logger.get_dir(), "monitor.json"))

    env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    if num_options ==1:
        optimsize=64
    elif num_options ==2:
        optimsize=32
    else:
        print("Only two options or primitive actions is currently supported.")
        sys.exit()

    # Start the visualization script
    visual.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=optimsize,
            gamma=0.99, lam=0.95, schedule='constant', num_options=num_options,
            app=app, saves=saves, wsaves=wsaves, epoch=epoch, seed=seed,dc=dc,
            episode_len=episode_len, path=path, render=render, official = official,
            orig_ppo=orig_ppo
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Pendulum-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--opt', help='number of options', type=int, default=2) 
    parser.add_argument('--app', help='Append to folder name', type=str, default='')        
    parser.add_argument('--saves', dest='saves', action='store_true', default=False)
    parser.add_argument('--wsaves', dest='wsaves', action='store_true', default=False)    
    parser.add_argument('--epoch', help='Epoch', type=int, default=-1) 
    parser.add_argument('--path', help='Path to the file to be loaded', type=str, default='')        
    parser.add_argument('--dc', type=float, default=0.1)
    parser.add_argument('--render', help='decision whether to render or not', type=int, default=0)
    parser.add_argument('--official', help='decision whether its an official implementation or not', type=int, default=0)
    parser.add_argument('--orig_ppo', help='decision whether its original ppo', type=int, default=0)


    args = parser.parse_args()

    train(args.env, num_timesteps=1e6, seed=args.seed, num_options=args.opt, app=args.app, saves=args.saves, wsaves=args.wsaves, \
        epoch=args.epoch,dc=args.dc,path=args.path,render=args.render,official=args.official,orig_ppo=args.orig_ppo)


if __name__ == '__main__':
    main()