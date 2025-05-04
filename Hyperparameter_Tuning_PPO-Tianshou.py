import os
import random
import datetime
import pickle
import tianshou as ts
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box
from torch.utils.tensorboard import SummaryWriter
from utils import FlattenMultiDiscreteActions
from Rl_Environment import CantileverEnv_v0_1
from tianshou.utils import WandbLogger
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.policy.base import BasePolicy
from tianshou.policy.modelfree.ppo import PPOTrainingStats
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from tianshou.utils.net.discrete import Actor, Critic
from ax.service.ax_client import AxClient, ObjectiveProperties
from path_util import paths
from PyAnsys_Environment import Cantilever

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
    


def test_ppo(test_envs, train_envs, config_kwargs, hparams, trial_index) -> None:
    #Environment Check 
    assert [isinstance(train_envs.action_space[i], gym.spaces.Discrete) 
            for i in range(config_kwargs.get("num_train_env"))]
    assert [isinstance(test_envs.action_space[i], gym.spaces.Discrete) 
            for i in range(config_kwargs.get("num_train_env"))]
    #space_info = SpaceInfo.from_env(env)
    state_shape = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].n
    
    # Set random seed
    set_random_seeds(config_kwargs["seed"], using_cuda=torch.cuda.is_available())
    

    # model
    net = Net(state_shape=state_shape, hidden_sizes= config_kwargs.get("hidden_sizes"), device=config_kwargs.get("device"))
    actor: nn.Module
    critic: nn.Module
    if torch.cuda.is_available():
        actor = DataParallelNet(Actor(net, action_shape, device=config_kwargs.get("device")).to(config_kwargs.get("device")))
        critic = DataParallelNet(Critic(net, device=config_kwargs.get("device")).to(config_kwargs.get("device")))
    else:
        actor = Actor(net, action_shape, device=config_kwargs.get("device")).to(config_kwargs.get("device"))
        critic = Critic(net, device=config_kwargs.get("device")).to(config_kwargs.get("device"))
    actor_critic = ActorCritic(actor, critic)
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=hparams["lr"])
    dist = torch.distributions.Categorical
    policy: PPOPolicy[PPOTrainingStats] = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        action_scaling=isinstance(train_envs.action_space[0], Box),
        discount_factor=config_kwargs.get("gamma"),
        max_grad_norm=config_kwargs.get("max_grad_norm"),
        eps_clip=config_kwargs.get("eps_clip"),
        vf_coef=config_kwargs.get("vf_coef"),
        ent_coef=config_kwargs.get("ent_coef"),
        gae_lambda=config_kwargs.get("gae_lambda"),
        reward_normalization=config_kwargs.get("rew_norm"),
        dual_clip=config_kwargs.get("dual_clip"),
        value_clip=config_kwargs.get("value_clip"),
        action_space=train_envs.action_space[0],
        deterministic_eval=True,
        advantage_normalization=config_kwargs.get("norm_adv"),
        recompute_advantage=config_kwargs.get("recompute_adv"),
    )
    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(config_kwargs.get("buffer_size"), config_kwargs.get("num_train_env")),
    )
    buf_test = VectorReplayBuffer(config_kwargs.get("num_test_env") *_env_gen_kwargs.get('eps_length') 
                                 *config_kwargs.get("episode_per_test"), buffer_num= config_kwargs.get("num_test_env")) 
    test_collector = Collector(policy, test_envs, buf_test)

    # log time
    dt = datetime.datetime.now(datetime.timezone.utc)
    dt = dt.replace(microsecond=0, tzinfo=None)
    # logger    
    wandb_logger = WandbLogger(project= config_kwargs.get("wandb_project"),
                         name= str(dt),
                         config = config_kwargs | _env_gen_kwargs)

    log_path = os.path.join( config_kwargs.get("logdir"),  config_kwargs.get("task"), "ppo")
    if not os.path.exists(log_path): os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    wandb_logger.load(writer)

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        if env_step%config_kwargs.get("step_per_epoch") == 0:
            ckpt_path = os.path.join(log_path, "checkpoint",f"checkpoint_{env_step}.pth")
            torch.save(
                {
                    "model": policy.state_dict(),
                    "optim": optim.state_dict(),
                },
                ckpt_path,
            )
            buffer_path = os.path.join(log_path, "train_buffer.pkl")
            with open(buffer_path, "wb") as f:
                pickle.dump(train_collector.buffer, f)
            return ckpt_path
    
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
        if epoch >= 2:
           #log data manually
           ax_data(env_step)
    # trainer
    n_steps = 2 ** hparams['step_per_collecte_pow2']
    minibatch_size = (config_kwargs.get("num_train_env") * n_steps) // (2 ** hparams['batch_size_div_pow2'])
    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=config_kwargs.get("epoch"),
        step_per_epoch=config_kwargs.get("step_per_epoch"),
        repeat_per_collect=config_kwargs.get("repeat_per_collect"),
        episode_per_test= config_kwargs.get("episode_per_test"),
        batch_size=minibatch_size,
        step_per_collect=n_steps,
        save_best_fn=save_best_fn,
        test_fn=test_fn,
        save_checkpoint_fn = save_checkpoint_fn,
        logger=wandb_logger,
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
"task":"CantileverEnv_v0_1-Wrapped",
"reward_threshold":None,
"seed":0,
"buffer_size":500_000,
"lr":3e-4,
"gamma":0.99,
"epoch":10,
"step_per_epoch":100_000,
"step_per_collect":80,
"repeat_per_collect":10,
"batch_size":64,
"hidden_sizes":[128, 128],
"num_train_env":10,
"num_test_env": 10,
"episode_per_test":10,
"logdir":"Ts-PPO-severity-cantilever-0-1-2-damages-Mode-1-2-3-3_Sensors-HPO-PC",
"model_name":"Ts-PPO-severity-cantilever-0-1-2-damages-Mode-1-2-3-3_Sensors-HPO-PC",
"ax_experiment_name":"Ts-PPO-severity-cantilever-0-1-2-damages-Mode-1-2-3-3_Sensors-HPO-PC",
"ax_objective_name":"avg_ep_rew",
"wandb_project":"Ts-PPO-severity-cantilever-0-1-2-damages-Mode-1-2-3-3_Sensors-HPO-PC",
'replay_buffer_name': "Tianshou-HPO-PC",
"device" : "cuda" if torch.cuda.is_available() else "cpu",
"vf_coef":0.5, #Hyperparameter [0.5 and 1]
"ent_coef":0.0, #Hyperparameter [0.0, 0.01]
"eps_clip":0.2,
"max_grad_norm":0.5,
"gae_lambda":0.95, #Hyperparater [0.9,1]
"rew_norm":True,
"norm_adv":0,
"recompute_adv":0,
"dual_clip":None,
"value_clip":0,
'verbose_ax': False,
'verbose_trial': 1,
'num_trials': 50,
}

hparams =  [
    {"name": "lr", 
     "type": "range", 
     "bounds": [1e-5, 1e-3], 
     "log_scale": True,
     },
    {"name": "batch_size_div_pow2", 
     "type": "range", 
     'value_type': "int",
    'bounds': [5, 9],    # Inclusive, 2**n between [1, 8]
    'log_scale': False,
    },
    {"name": "step_per_collecte_pow2", 
     "type": "range", 
     "value_type" : "int", 
     "bounds": [5, 12], # Inclusive, 2**n between [32, 4096]
     'log_scale': False,
     }
    ]

'''hparams =  [
    {"name": "lr", "type": "range", "bounds": [1e-5, 1e-3], "log_scale": True},
    {"name": "batch_size", "type": "choice", "values": [32, 64, 128, 256, 512]},
    {"name": "gamma", "type": "range", "bounds": [0.9, 0.99]},
    {"name": "gae_lambda", "type": "range", "bounds": [0.9, 1.0]},
    {"name": "ent_coef", "type": "range", "bounds": [0.0, 1e-2], "log_scale": True},
    {"name": "vf_coef", "type": "range", "bounds": [0.1, 1.0]},
    {"name": "ppo_epochs", "type": "range", "bounds": [1, 10]},
    {"name": "mini_batch_size", "type": "choice", "values": [32, 64, 128, 256]},
    {"name": "n_steps", "type": "choice", "values": [2048, 4096, 8192]},
]'''

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
            "eps_length" : 200,
            "node_id": [90,1670],
            "mass": [0.2,0.2]
    }

gym_env = Cantilever(_env_gen_kwargs)
_env_gen_kwargs["gym_env"] = gym_env
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
  avg_ep_rew = test_ppo(test_envs, train_envs, config_kwargs, next_hparams, trial_index)
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
