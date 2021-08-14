import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np
import os

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
#from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from nfunk.helper.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from importlib import import_module
from baselines import logger, bench

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    # 获取gym内所有的env，存放在_game_envs这个字典里面
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args) # 从传入的参数中获取env_type和env_id, 这里有个快速看代码的技巧
                                          # get_env_type这个函数作用非常明确，且对于我们理解主线算法没有关系
                                          # 那我们可以就就不看这个函数到底怎么实现的，这样不影响梳理整个这个工程
                                          # 是怎么实现的
    print('env_type: {}'.format(env_type)) # 就打印一下env_type

    total_timesteps = int(args.num_timesteps) # 从传入参数获取timesteps
    seed = args.seed    # 获取随机数，在学习算法中往往都需要一个随机数

    learn = get_learn_function(args.alg) # 获取学习函数
    # This function "pulls" the default arguments from the ALGORITHMS learn function
    alg_kwargs = get_learn_function_defaults(args.alg, env_type) # 获取学习函数的默认输入参数
    # This one adds the extra arguments to the default ones or overwrites the default ones if they specified in the extra arguments!
    # -> in essence: extra arguments are passed onto the train function (except for the network...)
    alg_kwargs.update(extra_args) # 如果你不想用默认参数，这里是更新学习算法的参数，转用给定的参数
    if ('max_episode' in extra_args):
        max_episode = extra_args['max_episode']
    else:
        max_episode = None
    env = build_env(args, max_episode=max_episode) # build强化学习的env
    if args.save_video_interval != 0:
        # 是否保存video，不影响理解
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)


    # 强化学习网络
    if args.network:    # is only true if args.network is set...
        alg_kwargs['network'] = args.network # 自己给定的网络
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type) # 用默认的网络，有cnn和mlp，和环境的对应关系
                                                                  # 可看get_default_network这个函数

    ## FOCUS!!!
    ## 至此，强化学习的模型所需输入都已经设定好了，env也是生成好了
    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    # learn是刚刚上面获取的学习函数
    # 这里开始学习了!!!
    # 接下来我们的进去这个learn函数去看怎么学习的
    # 从nfunk->helper->cmd_util.py这个函数里我们可以看到默认给的算法是ppo2，那接下来就以这个算法为例
    # 继续深入阅读代码
    # ok，和万博沟通后发现，这个repo里好像没有ppo2算法，那他给的default值就差点意思了
    # 回过头来看，其实原始的ppo1算法以及这个repo作者提出的算法实现都在functioning_implementations这个文件夹内
    # 这里的learn函数其实就是对应各个算法里的learn函数
    # 以ppo1算法为例，learn函数的实现在functioning_implementations->originl ppo1->pposgd_simple.py这个函数里面
    # 其余几个算法同样
    # !!!FOCUS 后续你如果在这个代码里要用自己的算法，其实就是改动传入的arg.alg这个参数，并实现对应的算法
    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def build_env(args,max_episode=None):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu

    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        #config = tf.ConfigProto(allow_soft_placment=True,
        #                       intra_op_parallelism_threads=1,
        #                       inter_op_parallelism_threads=1)

        config = tf.ConfigProto(intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1)

        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        # entry point for creating customized environment
        if (env_type in {'nf'}):
            if env_id == 'Pendulumnf-v0':
                from gym.envs.registration import register
                register(
                    id='Pendulumnf-v0',
                    entry_point='nfunk.envs_nf.pendulum_nf:PendulumEnv',
                    max_episode_steps=max_episode or 200,
                    kwargs = vars(args),
                )
                env = gym.make(env_id)
                #env = bench.Monitor(env, None)
                return env

        # if we want to use "normal" gym environment setting ->"branch out here":
        if (env_type in {'nf-par'}):
            env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale,
                               flatten_dict_observations=flatten_dict_observations, env_kwargs=vars(args))
        else:
            env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    return env


def get_env_type(args):
    env_id = args.env # 这个env_id是由传入参数给定的

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)
    #print (args)
    arg_parser = common_arg_parser()    # 创建一个参数解析器
    # 解析传入的参数，比如python main.py -env a -env_id, 这里就是解析出来env，env_id这样的参数
    # 传入参数预设了很多，详细可看common_arg_parser()这个函数怎么实现
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args) # 还是在解析参数


    # MPI是一个并行运算用的python库，详细介绍可见https://mpi4py.readthedocs.io/en/stable/
    # 对于理解强化学习算法无太大相关关系，了解是干啥的就行，这个repo里可能是用来提高运行速度的
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        # 设置logger路径，logger就是用来打印日志的，和算法没有关系。。
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    if args.save_path is not None:
        extra_args['save_path'] = args.save_path

    # 从train函数返回model和env，看到这里就要进去train这个函数继续看了
    model, env = train(args, extra_args)

    # 保存model
    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    # play这个参数可能只是用来显示动画的，不重要
    if args.play:
        logger.log("Running trained model")
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = 0
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew[0] if isinstance(env, VecEnv) else rew
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done
            if done:
                print('episode_rew={}'.format(episode_rew))
                episode_rew = 0
                obs = env.reset()

    env.close()

    return model

if __name__ == '__main__':
    main(sys.argv)
