import os
import gymnasium as gym
import random
import pickle
import numpy as np
import torch
import datetime
from tianshou.data import (
    Collector,
    PrioritizedVectorReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.policy import RainbowPolicy
from tianshou.policy.base import BasePolicy
from tianshou.policy.modelfree.rainbow import RainbowTrainingStats
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import NoisyLinear
from tianshou.utils.space_info import SpaceInfo
from utils import FlattenMultiDiscreteActions
from Rl_Environment import CantileverEnv_v0_1
from ax.service.ax_client import AxClient, ObjectiveProperties
from path_util import paths
from tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter 

def set_random_seeds(seed: int, using_cuda: bool = False) -> None:
  """
  Seed the different random generators.
  """
  # Set seed for Python random, NumPy, and Torch
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Set deterministic operations for CUDA
  if using_cuda:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def test_rainbow(env, config_kwargs, hparams) -> float:
    #Environment Check 
    assert isinstance(env.action_space, gym.spaces.Discrete)
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape

    train_envs = env
    test_envs = env
    
    # Set random seed
    set_random_seeds(config_kwargs["seed"], using_cuda=torch.cuda.is_available())
    
    def noisy_linear(x: int, y: int) -> NoisyLinear:
        return NoisyLinear(x, y,  config_kwargs.get("noisy_std"))

    net = Net(
        state_shape= state_shape,
        action_shape= action_shape,
        hidden_sizes= config_kwargs.get("hidden_sizes"),
        device= config_kwargs.get("device"),
        softmax=True,
        num_atoms= config_kwargs.get("num_atoms"),
        dueling_param= ({"linear_layer": noisy_linear}, {"linear_layer": noisy_linear}))
    optim = torch.optim.Adam(net.parameters(), 
                             lr= 0.00025 / hparams['learning_rate_denom'], eps = 1.5e-4)
    policy: RainbowPolicy[RainbowTrainingStats] = RainbowPolicy(
        model=net,
        optim=optim,
        discount_factor= hparams['gamma'],
        action_space=env.action_space,
        num_atoms= config_kwargs.get("num_atoms"),
        v_min= config_kwargs.get("v_min"),
        v_max= config_kwargs.get("v_max"),
        estimation_step=  hparams['multi_step_returns'],
        target_update_freq= hparams['target_network_update_freq'],
    ).to( config_kwargs.get("device"))
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
        
    
    buf_test = VectorReplayBuffer( config_kwargs.get("buffer_size"), buffer_num=config_kwargs.get("training_num")) 
    # collector
    train_collector = Collector(policy, train_envs, buf_train, exploration_noise=True)
    test_collector = Collector(policy, test_envs, buf_test,exploration_noise=True)
 
    train_collector.reset()
    train_collector.collect(n_step= hparams['batch_size'] *  config_kwargs.get("training_num"))
    
    # log time
    dt = datetime.datetime.now(datetime.timezone.utc)
    dt = dt.replace(microsecond=0, tzinfo=None)
    # logger    
    wandb_logger = WandbLogger(project= config_kwargs.get("wandb_project"),
                         name= str(dt),
                         config = hparams | config_kwargs)

    log_path = os.path.join( config_kwargs.get("logdir"),  config_kwargs.get("task"), "rainbow")
    if not os.path.exists(log_path): os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    wandb_logger.load(writer)
    

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "best_policy.pth"))

    total_steps = config_kwargs.get("epoch") * config_kwargs.get("step_per_epoch")
    def train_fn(epoch: int, env_step: int) -> None:
        #When using noisy net eps = 0.0
        eps = 0.0
        policy.set_eps(eps)
        # beta annealing, as discribed in the paper
        # Linearly increase beta from 0.4 to 1
        beta = config_kwargs.get("beta") + ((config_kwargs.get("beta_final") - config_kwargs.get("beta")) * env_step / total_steps)
        # Set beta in your buffer
        buf_train.set_beta(beta)   
    
    avg_ep_rew = []
    def test_fn(epoch: int, env_step: int | None) -> None:
        policy.set_eps( config_kwargs.get("eps_test"))
        if epoch >= 2:
            #Extract param from test buffer
            eps_rew = buf_test.get(np.arange(_env_gen_kwargs.get('eps_length') *config_kwargs.get("episode_per_test")),"rew" )
            list_avg_ep_rew = np.array_split(eps_rew, config_kwargs.get("episode_per_test"))
            avg_ep_rew.append(np.mean([ np.sum(ep_rew) for ep_rew in list_avg_ep_rew]))
            buf_test.reset()

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
        logger= wandb_logger,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
    ).run()
    wandb_logger.finalize()
    #Stats for last test
    eps_rew = buf_test.get(np.arange(_env_gen_kwargs.get('eps_length') *config_kwargs.get("episode_per_test")),"rew" )
    list_avg_ep_rew = np.array_split(eps_rew, config_kwargs.get("episode_per_test"))
    avg_ep_rew.append(np.mean([ np.sum(ep_rew) for ep_rew in list_avg_ep_rew]))
    buf_test.reset()
    return np.mean(avg_ep_rew)

config_kwargs = {
    "task" : "CantileverEnv_v0_1-Wrapped",
    "seed"  : 0,
    "eps_test"  : 0.0,
    "buffer_size"  : 1_000_000,
    "num_atoms" : 51, 
    "v_min"  : -10.0,
    "v_max" : 10.0,
    "noisy_std" : 0.5 if torch.cuda.is_available() else 0.1, 
    "step_per_collect"  : 4,
    "hidden_sizes" : [128, 128, 128], #Default
    "training_num"  : 1,
    "test_num"  : 1,
    "logdir"  : "Tianshou-cantilever-Mode-1_2_3-3_Sensors-Rainbow-HPO-Norm",
    "prioritized_replay" : True,
    "beta"  : 0.4,
    "beta_final"  : 1.0,
    "device"  : "cuda" if torch.cuda.is_available() else "cpu",
    'wandb_project': "Tianshou-cantilever-ax-hpo-Mode-1_2_3-3_Sensors-rainbow-HPO-Norm",
    'model_name': "Tianshou-rainbow-cantilever-Mode-1_2_3-3_Sensors-Rainbow-HPO-Norm",
    'ax_experiment_name': "Tianshou-rainbow-cantilever-experiment-Mode-1_2_3-3_Sensors-Rainbow-HPO-Norm",
    'ax_objective_name': "avg_ep_rew",
    'replay_buffer_name': "Tianshou-Rainbow-HPO-Norm-HPC",
    'num_trials': 50,
    "epoch"  : 50,
    'episode_per_test' : 10,
    "step_per_epoch"  : 10_000,
    'verbose_ax': False,
    'verbose_trial': 1,
}

# Define the hyperparameters we want to optimize

hparams = [
    {"name": "learning_rate_denom", "type": "choice", "values": [2, 4, 6]},
    {"name": "gamma", "type": "range", "bounds": [0.9, 0.99]},
    {"name": "batch_size", "type": "choice", "values": [32, 64, 128]},
    {"name": "priority_exponent", "type": "choice", "values": [0.4, 0.5, 0.7]},
    {"name": "target_network_update_freq", "type": "choice", "values": [320, 3200]},
    {"name": "multi_step_returns", "type": "choice", "values": [1, 3, 5]},
        ]

# Set parameter constraints
parameter_constraints = []

# Create our environment
try:
  env.close()
except NameError:
  pass


""" Enivronment Parameters
"""
#Path to ansys core
core_path, geo_path, mat_path = paths()
# environments setup
_env_gen_kwargs = {"elem_size": 0.005, #[m]
            "geo_path" : geo_path, 
            "core_path": core_path, 
            "seed" : 42,
            "max_step" : 5,
            "mat_path": mat_path,
            "sim_modes": [0,1,2],
            "num_sensors": 3,
            "mat_param" : [1.0, 1.0],
            "temp"       : 0,
            "render" : False,
            "eps_length" : 1000,
            "norm" : False,
            "mode_shape_folder_name" : "mode_shape_folder"
    }

env = FlattenMultiDiscreteActions(CantileverEnv_v0_1(_env_gen_kwargs))

# Construct path to Ax experiment snapshot file
ax_snapshot_path = os.path.join(config_kwargs['logdir'], f"{config_kwargs['ax_experiment_name']}.json")


# Load experiment from snapshot if it exists, otherwise create a new one
if os.path.exists(ax_snapshot_path):
  print(f"Loading experiment from snapshot: {ax_snapshot_path}")
  ax_client = AxClient.load_from_json_file(ax_snapshot_path)
else:
  print(f"Creating new experiment. Snapshot to be saved at {ax_snapshot_path}.")
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

# Choo choo! Perform trials to optimize hyperparameters
while True:

  # Get next hyperparameters and end experiment if we've reached max trials
  next_hparams, trial_index = ax_client.get_next_trial()
  if trial_index >= config_kwargs['num_trials']:
    break

  # Show that we're starting a new trial
  if config_kwargs['verbose_trial'] > 0:
    print(f"--- Trial {trial_index} ---")

  # Perform trial
  avg_ep_rew = test_rainbow(env, config_kwargs, next_hparams)
  ax_client.complete_trial(
      trial_index=trial_index,
      raw_data=avg_ep_rew,
  )

  # Save experiment snapshot
  ax_client.save_to_json_file(ax_snapshot_path)

best_param, values = ax_client.get_best_parameters()

#Log environment parameters
with open(os.path.join(config_kwargs["logdir"] , 'Config_file.txt'), 'w') as txt_file:
    for key, value in _env_gen_kwargs.items():
        txt_file.write(f'{key}: {value}\n')
    for key, value in config_kwargs.items():
        txt_file.write(f'{key}: {value}\n')
        
print("Configuration logged to 'Config.txt' successfully!")
print(f"Best Parameters : {best_param}")
print(f"Best Average Reward : {values[0]}")