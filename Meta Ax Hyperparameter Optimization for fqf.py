import os
import gymnasium as gym
import random
import pickle
import numpy as np
import torch
from tianshou.data import (
    Collector,
    PrioritizedVectorReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.policy import FQFPolicy
from tianshou.utils.net.discrete import FractionProposalNetwork, FullQuantileFunction
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo
from ax.service.ax_client import AxClient, ObjectiveProperties
from tianshou.env import DummyVectorEnv

def set_random_seeds(seed: int, using_cuda: bool = False) -> None:
  """
  config random generators.
  """
  # config seed for Python random, NumPy, and Torch
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Set deterministic CUDA operations
  if using_cuda:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def test_fqf(env, config_kwargs, hparams) -> float:
    #Environment Check 
    assert isinstance(env.action_space, gym.spaces.Discrete)
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape

    train_envs = DummyVectorEnv([lambda: gym.make( config_kwargs.get("task")) for _ in range( config_kwargs.get("training_num"))])
    test_envs = DummyVectorEnv([lambda: gym.make( config_kwargs.get("task")) for _ in range( config_kwargs.get("test_num"))])
    train_envs.seed( config_kwargs.get("seed"))
    test_envs.seed( config_kwargs.get("seed"))
    
    # Set random seed
    set_random_seeds(config_kwargs["seed"], using_cuda=torch.cuda.is_available())

    # model
    feature_net = Net(
        state_shape,
        config_kwargs.get("hidden_sizes")[-1],
        hidden_sizes=config_kwargs.get("hidden_sizes")[:-1],
        device=config_kwargs.get("device"),
        softmax=False,
    )
    net = FullQuantileFunction(
        feature_net,
        action_shape,
        config_kwargs.get("hidden_sizes"),
        num_cosines=config_kwargs.get("num_cosines"),
        device=config_kwargs.get("device"),
    )
    optim = torch.optim.Adam(net.parameters(), lr=hparams['learning_rate'])
    fraction_net = FractionProposalNetwork(hparams["num_fractions"], net.input_dim)
    fraction_optim = torch.optim.RMSprop(fraction_net.parameters(), lr= 10 ** hparams['fraction_lr_exp'])
    policy: FQFPolicy = FQFPolicy(
        model=net,
        optim=optim,
        fraction_model=fraction_net,
        fraction_optim=fraction_optim,
        action_space=env.action_space,
        discount_factor=hparams['gamma'],
        num_fractions= hparams["num_fractions"],
        ent_coef=hparams["ent_coef"],
        estimation_step=hparams['multi_step_returns'],
        target_update_freq=hparams['target_network_update_freq'],
    ).to(config_kwargs.get("device"))
    
     
    # buffer
    buf_train: PrioritizedVectorReplayBuffer | VectorReplayBuffer
    if  config_kwargs.get("prioritized_replay"):
        buf_train = PrioritizedVectorReplayBuffer(
             config_kwargs.get("buffer_size"),
            buffer_num=config_kwargs.get("training_num"),
            alpha= hparams['priority_exponent'],
            beta= config_kwargs.get("beta"),
            weight_norm=True,
        )
    else:
        buf_train = VectorReplayBuffer( config_kwargs.get("buffer_size"), buffer_num=config_kwargs.get("training_num"))
        
    # collector
    train_collector = Collector(policy, train_envs, buf_train, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
 
    train_collector.reset()
    train_collector.collect(n_step= hparams['batch_size'] *  config_kwargs.get("training_num"))
    
    #Local Log path
    log_path = os.path.join(config_kwargs.get("logdir"),  config_kwargs.get("task"), "fqf")
    if not os.path.exists(log_path): os.makedirs(log_path)

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "best_policy.pth"))

    #decay_steps = (config_kwargs.get("epoch") - config_kwargs.get("eps_decay_epoch")) * config_kwargs.get("step_per_epoch")
    total_steps = config_kwargs.get("epoch") * config_kwargs.get("step_per_epoch")
    def train_fn(epoch: int, env_step: int) -> None:
        #Linearly decrease from 1.0 to 0.01
        #if env_step <= decay_steps:
        #    eps = config_kwargs.get("eps_train") - env_step / decay_steps *(
        #       config_kwargs.get("eps_train") - config_kwargs.get("eps_train_final"))
        eps = config_kwargs.get("eps_train_final")
        policy.set_eps(eps)
        # beta annealing, as discribed in the paper
        # Linearly increase beta from 0.4 to 1
        beta = config_kwargs.get("beta") + ((config_kwargs.get("beta_final") - config_kwargs.get("beta")) * env_step / total_steps)
        # Set beta in your buffer
        buf_train.set_beta(beta)   
            
    def test_fn(epoch: int, env_step: int | None) -> None:
        policy.set_eps( config_kwargs.get("eps_test"))


    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        ckpt_path = os.path.join(log_path, config_kwargs.get("model_name"), f"checkpoint_{env_step}.pth")
        torch.save(
            {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
            },
            ckpt_path,
        )
        buffer_path = os.path.join(log_path, config_kwargs.get("replay_buffer_name") , f"train_buffer_{env_step}.pkl")
        with open(buffer_path, "wb") as f:
            pickle.dump(train_collector.buffer, f)
        return ckpt_path


    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch= config_kwargs.get("epoch"),
        step_per_epoch= config_kwargs.get("step_per_epoch"),
        step_per_collect= config_kwargs.get("step_per_collect"),
        episode_per_test= config_kwargs.get("episode_per_test"),
        batch_size= hparams['batch_size'],
        update_per_step= 1/ config_kwargs.get("step_per_collect"),
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
    ).run()
    
    return result.best_reward

config_kwargs = {
    "task" : "CartPole-v1",
    "seed"  : 0,
    "eps_test"  : 0.001,
    "eps_train":1.0,
    'eps_train_final' : 0.01,
    "buffer_size"  : 1_000_000,
    "num_cosines":64,
    "step_per_collect"  : 10,
    'episode_per_test' : 10,
    "hidden_sizes" : [256, 256, 256],
    "training_num"  : 1,
    "test_num"  : 1,
    "logdir"  : "Tianshou-CartPole-v1-FQF-HPO",
    "prioritized_replay" : True,
    "beta"  : 0.4,
    "beta_final"  : 1.0,
    "device"  : "cuda" if torch.cuda.is_available() else "cpu",
    'model_name': "/users/mep22coo/Cantilever-Projects/Tianshou-fqf-CartPole-v1-FQF-HPC",
    'ax_experiment_name': "Tianshou-fqf-CartPole-v1-FQF",
    'ax_objective_name': "avg_ep_rew",
    'replay_buffer_name': "Tianshou-FQF",
    'num_trials': 50,
    "epoch"  : 50,
    "eval_episode" : 10,
    "step_per_epoch"  : 10_000,
    'verbose_ax': False,
    'verbose_trial': 1,
}

# Define the hyperparameters we want to optimize

hparams = [
    {"name": "learning_rate", "type": "range", "bounds": [1e-5, 1e-2], "log_scale": True},
    {"name": "fraction_lr_exp", "type": "range", "bounds": [-10, -6], "value_type": "float"},  # Exponent for fraction_lr
    {"name": "num_fractions", "type": "choice", "values": [8, 32, 64]},
    {"name": "ent_coef", "type": "range", "bounds": [0.1, 20.0], "value_type": "float", "log_scale": True},
    {"name": "gamma", "type": "range", "bounds": [0.9, 0.99]},
    {"name": "batch_size", "type": "choice", "values": [32, 64, 128]},
    {"name": "priority_exponent", "type": "range", "bounds": [0.4, 0.7]},
    {"name": "target_network_update_freq", "type": "range", "bounds": [1000, 50000], "value_type": "int", "log_scale": True},  # Target frequency often benefits from logarithmic scaling.
    {"name": "multi_step_returns", "type": "choice", "values": [3, 5]},
        ]

# Set parameter constraints
parameter_constraints = []

# Create our environment
try:
  env.close()
except NameError:
  pass

env = gym.make("CartPole-v1")

# Ax experiment snapshot file path
ax_snapshot_path = os.path.join(config_kwargs['logdir'], f"{config_kwargs['ax_experiment_name']}.json")


# Load experiment for resumption if previously saved
if os.path.exists(ax_snapshot_path):
  ax_client = AxClient.load_from_json_file(ax_snapshot_path)
else:
  ax_client = AxClient(
      random_seed=config_kwargs['seed'],
      verbose_logging=config_kwargs['verbose_ax']
  )
  ax_client.create_experiment(
      name=config_kwargs['ax_experiment_name'],
      parameters=hparams,
      objectives={config_kwargs['ax_objective_name']: ObjectiveProperties(minimize=False)},
      parameter_constraints=parameter_constraints,
  )

# Perform trials to optimize hyperparameters
while True:

  # Get successive hyperparameters
  next_hparams, trial_index = ax_client.get_next_trial()
  if trial_index >= config_kwargs['num_trials']:
    break

  if config_kwargs['verbose_trial'] > 0:
    print(f"...Experiment {trial_index}")

  # Perform trial
  avg_ep_rew = test_fqf(env, config_kwargs, next_hparams)
  ax_client.complete_trial(
      trial_index=trial_index,
      raw_data=avg_ep_rew,
  )

  # Save ax experiment snapshot
  ax_client.save_to_json_file(ax_snapshot_path)

best_param, values = ax_client.get_best_parameters()

#Log environment parameters
with open(os.path.join(config_kwargs["logdir"] , 'Config_file.txt'), 'w') as txt_file:
    for key, value in config_kwargs.items():
        txt_file.write(f'{key}: {value}\n')
        
print("Configuration logged to 'Config.txt' successfully!")
print(f"Best Parameters : {best_param}")
print(f"Best Average Reward : {values[0]}")






