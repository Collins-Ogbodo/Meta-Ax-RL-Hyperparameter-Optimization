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
from utils import FlattenMultiDiscreteActions
from Rl_Environment import CantileverEnv_v0_1
from path_util import paths
from tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter 
import tianshou as ts
from ax.service.ax_client import AxClient, ObjectiveProperties
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
 
    
def test_rainbow(train_envs, test_envs, config_kwargs, hparams, trial_index) -> float:
    #Environment Check 
    assert [isinstance(train_envs.action_space[i], gym.spaces.Discrete) 
            for i in range(config_kwargs["num_train_env"])]
    assert [isinstance(test_envs.action_space[i], gym.spaces.Discrete) 
            for i in range(config_kwargs["num_test_env"])]
    #space_info = SpaceInfo.from_env(env)
    state_shape = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].n
    
    # Set random seed
    set_random_seeds(config_kwargs["seed"], using_cuda=torch.cuda.is_available())
    
    def noisy_linear(x: int, y: int) -> NoisyLinear:
        return NoisyLinear(x, y,  config_kwargs.get("noisy_std"))
    
    Q_param = {"hidden_sizes": config_kwargs.get("hidden_sizes")}
    V_param = {"hidden_sizes": config_kwargs.get("hidden_sizes")}
    net = Net(
        state_shape= state_shape,
        action_shape= action_shape,
        hidden_sizes= config_kwargs.get("hidden_sizes"),
        device= config_kwargs.get("device"),
        softmax=True,
        num_atoms= config_kwargs.get("num_atoms"),
        dueling_param= (Q_param, V_param))
    optim = torch.optim.Adam(net.parameters(), 
                             lr= 0.00025 / hparams['learning_rate_denom'], eps = 1.5e-4)
    policy: RainbowPolicy[RainbowTrainingStats] = RainbowPolicy(
        model=net,
        optim=optim,
        discount_factor= config_kwargs['gamma'],
        action_space=train_envs.action_space[0],
        num_atoms= config_kwargs.get("num_atoms"),
        v_min= config_kwargs.get("v_min"),
        v_max= config_kwargs.get("v_max"),
        estimation_step=  hparams['multi_step_returns'],
        target_update_freq= config_kwargs['target_network_update_freq'],
    ).to( config_kwargs.get("device"))
    # buffer
    buf_train: PrioritizedVectorReplayBuffer | VectorReplayBuffer
    if  config_kwargs.get("prioritized_replay"):
        buf_train = PrioritizedVectorReplayBuffer(
             config_kwargs.get("buffer_size"),
            buffer_num=config_kwargs.get("num_train_env"),
            alpha= hparams['priority_exponent'],
            beta= config_kwargs.get("beta"),
            weight_norm=True,
        )
    else:
        buf_train = VectorReplayBuffer( config_kwargs.get("buffer_size"), buffer_num= config_kwargs.get("num_train_env"))
        
    
    buf_test = VectorReplayBuffer(config_kwargs.get("num_test_env") *_env_gen_kwargs.get('eps_length') 
                                 *config_kwargs.get("episode_per_test"), buffer_num= config_kwargs.get("num_test_env")) 
    # collector
    train_collector = Collector(policy, train_envs, buf_train, exploration_noise=True)
    test_collector = Collector(policy, test_envs, buf_test,exploration_noise=True)
 
    train_collector.reset()
    train_collector.collect(n_step= config_kwargs['batch_size'] * config_kwargs.get("num_train_env"))
    
    # log time
    dt = datetime.datetime.now(datetime.timezone.utc)
    dt = dt.replace(microsecond=0, tzinfo=None)
    # logger    
    wandb_logger = WandbLogger(project= config_kwargs.get("wandb_project"),
                         name= str(dt),
                         config = hparams | config_kwargs | {"Env Condition Node Id": train_envs.get_env_attr("env_conds_node_id", 0)})

    log_path = os.path.join( config_kwargs.get("logdir"),  config_kwargs.get("task"), "rainbow")
    if not os.path.exists(log_path): os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    wandb_logger.load(writer)
    

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, f"best_policy-{trial_index}.pth"))

    total_steps = config_kwargs.get("epoch") * config_kwargs.get("step_per_epoch")
    
    def train_fn(epoch: int, env_step: int) -> None:
        #When using noisy net eps = 0.0
        
        if env_step <= config_kwargs.get("decay_steps"):
            eps = config_kwargs.get("eps_train_initial") - (env_step / config_kwargs.get("decay_steps")) *(
            config_kwargs.get("eps_train_initial") - config_kwargs.get("eps_train_final"))
        else:
            eps = 0.01
        policy.set_eps(eps)
        # beta annealing, as discribed in the paper
        # Linearly increase beta from 0.4 to 1
        beta = config_kwargs.get("beta") + ((config_kwargs.get("beta_final") - config_kwargs.get("beta")) * env_step / total_steps)
        # Set beta in your buffer
        buf_train.set_beta(beta)  
     
    #Get final state reward metric
    def find_last_non_zero(lst): 
        arr = np.array(lst) 
        non_zero_indices = np.nonzero(arr)[0] 
        if len(non_zero_indices) == 0: 
            return None 
        return arr[non_zero_indices[-1]]
    
    avg_ep_rew = []
    def ax_data(env_step):
        #Extract reward from test buffer
        eps_rew = buf_test.get(np.arange(config_kwargs.get("num_test_env") *_env_gen_kwargs.get('eps_length') 
                                             *config_kwargs.get("episode_per_test")),"rew")
        list_avg_ep_rew = np.array_split(eps_rew, config_kwargs.get("episode_per_test"))
        avg_rew = np.mean([ np.sum(ep_rew) for ep_rew in list_avg_ep_rew])
        avg_ep_rew.append(avg_rew)
        
        #Extract reward metric from test buffer           
        rew_metric = buf_test.get(np.arange(config_kwargs.get("num_test_env") *_env_gen_kwargs.get('eps_length') 
                                            * config_kwargs.get("episode_per_test")),"info" )['reward_metric']
        list_test_reward_metric = np.array_split(rew_metric, config_kwargs.get("episode_per_test"))

        avg_ep_rew_metric_final = np.mean([ find_last_non_zero(ep_rew_metric) for ep_rew_metric in list_test_reward_metric])
        print([ find_last_non_zero(ep_rew_metric) for ep_rew_metric in list_test_reward_metric])
        avg_ep_rew_metric_sum = np.mean([ np.sum(ep_rew_metric) for ep_rew_metric in list_test_reward_metric])
        wandb_logger.write('test/env_step', env_step, {'avg_ep_rew_metric_sum': avg_ep_rew_metric_sum})
        wandb_logger.write('test/env_step', env_step, {'avg_ep_rew_metric_final': avg_ep_rew_metric_final})
        buf_test.reset()
        #print("avg_rew",avg_rew)
        #print("avg_ep_rew_metric_final",avg_ep_rew_metric_final)
        #print("avg_ep_rew_metric_sum",avg_ep_rew_metric_sum)
        
    
    def test_fn(epoch: int, env_step: int | None) -> None:
        policy.set_eps( config_kwargs.get("eps_test"))
        if epoch >= 2:
           #log data manually
           ax_data(env_step)

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
        batch_size= config_kwargs['batch_size'],
        update_per_step= 1/ config_kwargs.get("step_per_collect"),
        train_fn=train_fn,
        logger= wandb_logger,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
    ).run()
    wandb_logger.finalize()
    #Stats for last test
    eps_rew = buf_test.get(np.arange(config_kwargs.get("num_train_env") 
                        *_env_gen_kwargs.get('eps_length') *config_kwargs.get("episode_per_test")),"rew" )
    list_avg_ep_rew = np.array_split(eps_rew, config_kwargs.get("episode_per_test"))
    avg_ep_rew.append(np.mean([ np.sum(ep_rew) for ep_rew in list_avg_ep_rew]))
    torch.save(policy.state_dict(), os.path.join(log_path, f"final_policy-{trial_index}.pth"))
    buf_test.reset()
    return np.mean(avg_ep_rew) 

config_kwargs = {
    "task" : "CantileverEnv_v0_1-Wrapped",
    "batch_size": 32,
    "gamma" : 0.9,
    "seed"  : 0,
    "eps_train_initial"  : 1.0, 
    "eps_train_final"  : 0.01, 
    "eps_test"  : 0.0,
    "buffer_size"  : 1_000_000,
    "num_atoms" : 51, 
    "v_min"  : -10.0,
    "v_max" : 10.0,
    "noisy_std" : 0.5 if torch.cuda.is_available() else 0.1, 
    "step_per_collect"  : 4,
    "hidden_sizes" : [128, 128, 128], #Default
    "num_train_env"  : 2,
    "num_test_env"  : 2,
    "logdir"  : "Tianshou-severity-cantilever-0-1-2-damages-Mode-1-2-3-3_Sensors-HPO-PC-[128, 128, 128]",
    "prioritized_replay" : True,
    "beta"  : 0.4,
    "beta_final"  : 1.0,
    "device"  : "cpu", #"cuda" if torch.cuda.is_available() else "cpu",
    'wandb_project': "Tianshou-severity-cantilever-0-1-2-damages-Mode-1-2-3-3_Sensors-HPO-PC-[128, 128, 128]",
    'model_name': "Tianshou-severity-cantilever-0-1-2-damages-Mode-1-2-3-3_Sensors-HPO-PC",
    'ax_experiment_name': "T-severity-canti-0-1-2-dam-Mode-1-2-3-3_Sensors-PC-[128, 128, 128]",
    'ax_objective_name': "avg_ep_rew",
    'replay_buffer_name': "Tianshou-HPO-PC",
    'num_trials': 27,
    "epoch"  : 100,
    'episode_per_test' : 3,
    "step_per_epoch"  : 10_000,
    "decay_steps" : 250_000,
    "target_network_update_freq" : 3200,
    'verbose_ax': False,
    'verbose_trial': 1,
}

# Define the hyperparameters we want to optimize

hparams = [
    {"name": "learning_rate_denom", "type": "choice", "values": [2, 4, 6]},
    {"name": "priority_exponent", "type": "choice", "values": [0.4, 0.5, 0.7]},
    {"name": "multi_step_returns", "type": "choice", "values": [1, 3, 5]},
        ]

# Set parameter constraints
parameter_constraints = []
    

# Create our environment
try:
  envs.close()
except NameError:
  pass


""" Enivronment Parameters
"""
#Path to ansys core
core_path, geo_path, mat_path = paths()
# environments setup
_env_gen_kwargs = {
            "geo_path" : geo_path, 
            "core_path": core_path, 
            "mat_path": mat_path,
            "sim_modes": [0,1,2],
            "num_sensors": 3,
            "num_conditions" : 2,
            "render" : False,
            "norm" : True,
            "eps_length" : 1000,
    }

envs = ts.env.DummyVectorEnv([lambda: FlattenMultiDiscreteActions(CantileverEnv_v0_1(_env_gen_kwargs)) 
                              for _ in range(config_kwargs.get("num_train_env"))])
test_envs = envs
train_envs = envs

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
  avg_ep_rew = test_rainbow(test_envs, train_envs, config_kwargs, next_hparams, trial_index)
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
