import time
import datetime
import os
import random
import logging
from typing import Any, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import wandb

import stable_baselines3 as sb3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, Logger
from stable_baselines3.common.evaluation import evaluate_policy

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import init_notebook_plotting, render
from ax.plot.contour import interact_contour
import plotly.io as pio
pio.renderers.default = "jupyterlab"
init_notebook_plotting()

# Log in to Weights & Biases
wandb.login()

# Make wandb be quiet
os.environ["WANDB_SILENT"] = "true"
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

def set_random_seeds(seed: int, using_cuda: bool = False) -> None:
  """
  Seed the different random generators.
  """

  # Set seed for Python random, NumPy, and Torch
  random.seed(seed)
  np.random.seed(seed)
  th.manual_seed(seed)

  # Set deterministic operations for CUDA
  if using_cuda:
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

from Rl_Environment import CantileverEnv
""" Enivronment Parameters
"""
#Path to ansys core
path = 'C:\\Program Files\\ANSYS Inc\\ANSYS Student\\v242\\aisol\\bin\winx64\\AnsysWBU.exe'

geo_path = 'C:\\Users\\ogbod\\Documents\\PhD\\PhD Code\\Structural-Testing-Digital-Twin\\RL-Project\\Cantilever\\Geometry\\Cantilever-EMA.agdb'
mat_path = 'C:\\Users\\ogbod\\Documents\\PhD\\PhD Code\\Structural-Testing-Digital-Twin\\RL-Project\\Cantilever\\Material\\ANSYS GRANTA-Low-Alloy-Steel-4140-Normalised.xml'

# environments setup
_env_gen_kwargs = {"elem_size": 0.005, #[m]
             "geo_path" : geo_path, 
             "core_path": path, 
             "seed" : 42,
             "max_step" : 5,
             "mat_path": mat_path,
             "sim_modes": [0,1,2],
             "num_sensors": 2,
             "mat_param" : [1.0, 1.0],
             "temp"       : 10,
             "render" : False,
             "eps_length" : 1024,
             "mode_shape_folder_name" : "mode_shape_folder"
    }
#env = CantileverEnv(_env_gen_kwargs)

# Function that tests the model in the given environment
def test_agent(env, model, max_steps=0):
  #Test deterministic policy
  mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
  # Reset environment
  obs, info = env.reset()
  ep_len = 0
  ep_rew_list = []
  avg_step_time = 0.0
  eps_reward_metric_sum = 0.0
  # Run episode until complete
  while True:

    # Provide observation to policy to predict the next action
    timestamp = time.time()
    action, _ = model.predict(obs, deterministic=True)

    # Perform action, update total reward
    obs, reward, terminated, truncated, info = env.step(action)
    avg_step_time += time.time() - timestamp
    ep_rew_list.append(reward)
    eps_reward_metric_sum += info['reward_metric']
    # Increase step counter
    ep_len += 1
    if (max_steps > 0) and (ep_len >= max_steps):
      break

    # Check to see if episode has ended
    if terminated or truncated:
      break

  # Calculate average step time
  avg_step_time /= ep_len
  #Remwar Metric
  eps_reward_metric_final = info['reward_metric']
  ep_rew = np.sum(ep_rew_list)
  ep_std = np.std(ep_rew_list)
  return ep_len, ep_rew, ep_std,avg_step_time, eps_reward_metric_sum, eps_reward_metric_final

# Evaluate agent on a number of tests
def evaluate_agent(env, model, steps_per_test, num_tests):

  # Initialize metrics
  avg_ep_len = 0
  avg_ep_rew = 0
  avg_ep_std = 0
  avg_step_time = 0.0
  avg_reward_metric_sum = 0.0
  avg_reward_metric_final = 0.0
  # Test the agent a number of times
  for ep in range(num_tests):
    ep_len, ep_rew, ep_std, step_time, reward_metric_sum, reward_metric_final = test_agent(env, model, max_steps=steps_per_test)
    avg_ep_len += ep_len
    avg_ep_rew += ep_rew
    avg_ep_std += ep_std
    avg_step_time += step_time
    avg_reward_metric_sum += reward_metric_sum
    avg_reward_metric_final += reward_metric_final


  # Compute metrics
  avg_ep_len /= num_tests
  avg_ep_rew /= num_tests
  avg_step_time /= num_tests
  avg_reward_metric_sum /= num_tests
  avg_reward_metric_final /= num_tests

  return avg_ep_len, avg_ep_rew, avg_ep_std, avg_step_time, avg_reward_metric_sum, avg_reward_metric_final

class EvalAndSaveCallback(BaseCallback):
  """
  Evaluate and save the model every ``check_freq`` steps
  """

  # Constructor
  def __init__(
      self,
      check_freq,
      save_dir,
      model_name="model",
      replay_buffer_name=None,
      steps_per_test=0,
      num_tests=10,
      step_offset=0,
      verbose=1,
  ):
    super(EvalAndSaveCallback, self).__init__(verbose)
    self.check_freq = check_freq
    self.save_dir = save_dir
    self.model_name = model_name
    self.replay_buffer_name = replay_buffer_name
    self.num_tests = num_tests
    self.steps_per_test = steps_per_test
    self.step_offset = step_offset
    self.verbose = verbose

  # Create directory for saving the models
  def _init_callback(self):
    if self.save_dir is not None:
      os.makedirs(self.save_dir, exist_ok=True)

  # Save and evaluate model at a set interval
  def _on_step(self):
    if self.n_calls % self.check_freq == 0:

      # Set actual number of steps (including offset)
      actual_steps = self.step_offset + self.n_calls

      # Save model
      model_path = os.path.join(self.save_dir, f"{self.model_name}_{str(actual_steps)}")
      self.model.save(model_path)

      # Save replay buffer
      if self.replay_buffer_name != None:
        replay_buffer_path = os.path.join(self.save_dir, f"{self.replay_buffer_name}")
        self.model.save_replay_buffer(replay_buffer_path)

      # Evaluate the agent
      avg_ep_len, avg_ep_rew, avg_ep_std, avg_step_time, avg_reward_metric_sum, avg_reward_metric_final = evaluate_agent(
          env,
          self.model,
          self.steps_per_test,
          self.num_tests
      )
      if self.verbose:
        print(f"{str(actual_steps)} steps | average test length: {avg_ep_len},average test reward: {avg_ep_rew}, average test reward metric sum : {avg_reward_metric_sum}, average test reward metric final : {avg_reward_metric_final}")

      # Log metrics to WandB
      log_dict = {
          'avg_ep_len': avg_ep_len,
          'avg_ep_rew': avg_ep_rew,
          'avg_ep_std' : avg_ep_std,
          'avg_step_time': avg_step_time,
          'avg_reward_metric_sum' : avg_reward_metric_sum,
          'avg_reward_metric_final' : avg_reward_metric_final,
          'train/actor_loss': self.model.logger.name_to_value['train/n_updates'],
          'train/approx_k1': self.model.logger.name_to_value['train/approx_k1'],
          'train/clip_fraction': self.model.logger.name_to_value['train/clip_fraction'],
          'train/clip_range': self.model.logger.name_to_value['train/clip_range'],
          'train/critic_loss': self.model.logger.name_to_value['train/critic_loss'],
          'train/ent_coef': self.model.logger.name_to_value['train/ent_coef'],
          'train/ent_coef_loss': self.model.logger.name_to_value['train/ent_coef_loss'],
          'train/entropy_loss': self.model.logger.name_to_value['train/entropy_loss'],
          'train/explained_variance': self.model.logger.name_to_value['train/explained_variance'],
          'train/learning_rate': self.model.logger.name_to_value['train/learning_rate'],
          'train/loss': self.model.logger.name_to_value['train/loss'],
          'train/n_updates': self.model.logger.name_to_value['train/n_updates'],
          'train/policy_gradient_loss': self.model.logger.name_to_value['train/policy_gradient_loss'],
          'train/value_loss': self.model.logger.name_to_value['train/value_loss'],
          'train/std': self.model.logger.name_to_value['train/std'],
      }
      wandb.log(log_dict, commit=True, step=actual_steps)

    return True


class WandBWriter(KVWriter):
  """
  Log metrics to Weights & Biases when called by .learn()
  """

  # Initialize run
  def __init__(self, run, verbose=1):
    super().__init__()
    self.run = run
    self.verbose = verbose

  # Write metrics to W&B project
  def write(
    self,
    key_values: Dict[str, Any],
    key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
    step: int = 0,
  ) -> None:

    log_dict = {}

    # Go through each key/value pairs
    for (key, value), (_, excluded) in zip(
      sorted(key_values.items()), sorted(key_excluded.items())):

      if self.verbose >= 2:
        print(f"step={step} | {key} : {value} ({type(value)})")

      # Skip excluded items
      if excluded is not None and "wandb" in excluded:
        continue

      # Log integers and floats
      if isinstance(value, np.ScalarType):
        if not isinstance(value, str):
          wandb.log(data={key: value}, step=step)
          log_dict[key] = value

    # Print to console
    if self.verbose >= 1:
      print(f"Log for steps={step}")
      print("--------------")
      for (key, value) in sorted(log_dict.items()):
        print(f"  {key}: {value}")
      print()

  # Close the W&B run
  def close(self) -> None:
    self.run.finish()
    

def do_trial(settings, hparams):
  """
  Training loop used to evaluate a set of hyperparameters
  """

  # Set random seed
  set_random_seeds(settings['seed'], using_cuda=th.cuda.is_available())

  # Create new W&B run
  config = {}
  dt = datetime.datetime.now(datetime.timezone.utc)
  dt = dt.replace(microsecond=0, tzinfo=None)
  run = wandb.init(
      project=settings['wandb_project'],
      name=str(dt),
      config=config,
      allow_val_change=True,
      settings=wandb.Settings(silent=(not settings['verbose_wandb']))
  )
  # Print run info
  if settings['verbose_trial'] > 0:
    print(f"WandB run ID: {run.id}")
    print(f"WandB run name: {run.name}")

  # Log hyperparameters to W&B
  wandb.config.update(hparams, allow_val_change=True)

  # Set custom Logger with our custom writer
  wandb_writer = WandBWriter(run, verbose=settings['verbose_log'])
  loggers = Logger(
      folder=None,
      output_formats=[wandb_writer]
  )

  # Calculate derived hyperparameters
  n_steps = 2 ** hparams['steps_per_update_pow2']
  minibatch_size = (hparams['n_envs'] * n_steps) // (2 ** hparams['batch_size_div_pow2'])
  layer_1 = 2 ** hparams['layer_1_pow2']
  layer_2 = 2 ** hparams['layer_2_pow2']


  # Set completed steps to checkpoint number (in filename) or 0 to start over
  # TODO: how to resume if trial is paused/cancelled
  completed_steps = 0

  # Load or create new model
  if completed_steps != 0:
    model_path = os.path.join(settings['save_dir'], f"{settings['model_name']}_{str(completed_steps)}.zip")
    model = sb3.SAC.load(model_path, env)
    steps_to_complete = settings['total_steps'] - completed_steps
  else:
    model = sb3.SAC(
        'MlpPolicy',
        env,
        learning_rate=hparams['learning_rate'],
        train_freq=n_steps,
        learning_starts = 10000,
        batch_size=minibatch_size,
        gradient_steps = int(0.1 * n_steps),
        gamma=hparams['gamma'],
        ent_coef=hparams['entropy_coef'],
        action_noise = action_noise,
        policy_kwargs={'net_arch': [layer_1, layer_2]},
        use_sde=hparams['use_sde'],
        sde_sample_freq=hparams['sde_freq'],
        verbose=settings['verbose_train'],
        device = "cuda" if th.cuda.is_available() else "cpu"
    )
    steps_to_complete = settings['total_steps']

  # Set up checkpoint callback
  checkpoint_callback = EvalAndSaveCallback(
      check_freq=settings['checkpoint_freq'],
      save_dir=settings['save_dir'],
      model_name=settings['model_name'],
      replay_buffer_name=settings['replay_buffer_name'],
      steps_per_test=settings['steps_per_test'],
      num_tests=settings['tests_per_check'],
      step_offset=(settings['total_steps'] - steps_to_complete),
      verbose=settings['verbose_test'],
  )

  # Choo choo train
  model.learn(total_timesteps=steps_to_complete,
              callback=[checkpoint_callback])

  # Get dataframe of run metrics
  history = wandb.Api().run(f"{run.project}/{run.id}").history()

  # Get index of evaluation with maximum reward
  max_idx = np.argmax(history.loc[:, 'avg_ep_rew'].values)

  # Find number of steps required to produce that maximum reward
  max_rew_steps = history['_step'][max_idx]
  _avg_reward_metric_final = history['avg_reward_metric_final'][max_idx]
  _avg_reward_metric_sum = history['avg_reward_metric_sum'][max_idx]
  if settings['verbose_trial'] > 0:
    print(f"Steps with max reward: {max_rew_steps}")
    print(f"reward metric final at Steps with max reward: {_avg_reward_metric_final}")
    print(f"reward metric sum at Steps with max reward: {_avg_reward_metric_sum}")

  # Load model with maximum reward from previous run
  model_path = os.path.join(settings['save_dir'], f"{settings['model_name']}_{str(max_rew_steps)}.zip")
  model = sb3.SAC.load(model_path, env)

  # Evaluate the agent
  avg_ep_len, avg_ep_rew, avg_ep_std, avg_step_time, avg_reward_metric_sum, avg_reward_metric_final = evaluate_agent(
      env,
      model,
      settings['steps_per_test'],
      settings['tests_per_check'],
  )

  # Log final evaluation metrics to WandB run
  wandb.run.summary['Average test episode length'] = avg_ep_len
  wandb.run.summary['Average test episode reward'] = avg_ep_rew
  wandb.run.summary['Average test episode std'] = avg_ep_std
  wandb.run.summary['Average test step time'] = avg_step_time
  wandb.run.summary['Average test reward metric sum'] = avg_reward_metric_sum
  wandb.run.summary['Average test reward metric final'] = avg_reward_metric_final

  # Print final run metrics
  if settings['verbose_trial'] > 0:
    print('---')
    print(f"Best model: {settings['model_name']}_{str(max_rew_steps)}.zip")
    print(f"Average episode length: {avg_ep_len}")
    print(f"Average episode reward: {avg_ep_rew}")
    print(f"Average episode std: {avg_ep_std}")
    print(f"Average step time: {avg_step_time}")
    print(f"Average test reward metric sum: {avg_reward_metric_sum}")
    print(f"Average test reward metric final: {avg_reward_metric_final}")

  # Close W&B run
  run.finish()

  return avg_ep_rew

# Project settings that do not change
settings = {
    'wandb_project': "SAC-cantilever-ax-hpo-Mode-1-2-3",
    'model_name': "SAC-cantilever-Mode-1-2-3",
    'ax_experiment_name': "SAC-cantilever-experiment-Mode-1-2-3",
    'ax_objective_name': "avg_ep_rew",
    'replay_buffer_name': 'SAC_replay_buffer',
    'save_dir': "SAC-checkpoints-Mode-1-2-3",
    'checkpoint_freq': 5120,
    'steps_per_test': 1024,
    'tests_per_check': 2,
    'total_steps': 51_200,
    'num_trials': 20,
    'seed': 42,
    'verbose_ax': False,
    'verbose_wandb': False,
    'verbose_train': 0,
    'verbose_log': 0,
    'verbose_test': 0,
    'verbose_trial': 1,
}

# Define the hyperparameters we want to optimize
hparams = [
  {
    'name': "n_envs",
    'type': "fixed",
    'value_type': "int",
    'value': 1,
  },
  {
    'name': "learning_rate",
    'type': "range",
    'value_type': "float",
    'bounds': [1e-5, 1e-2],
    'log_scale': True,
  },
  {
    'name': "steps_per_update_pow2",
    'type': "range",
    'value_type': "int",
    'bounds': [6, 12],    # Inclusive, 2**n between [64, 4096]
    'log_scale': False,
  },
  {
    'name': "batch_size_div_pow2",
    'type': "range",
    'value_type': "int",
    'bounds': [5, 8],    # Inclusive, 2**n between [1, 8]
    'log_scale': False,
  },
  {
    'name': "gamma",
    'type': "range",
    'value_type': "float",
    'bounds': [0.9, 0.99],
    'log_scale': False,
  },
  {
    'name': "entropy_coef",
    'type': "range",
    'value_type': "float",
    'bounds': [0.1, 0.3],
    'log_scale': False,
  },
  {
    'name': "layer_1_pow2",
    'type': "range",     #'range'
    'value_type': "int",
    'bounds' :[5, 8],    # Inclusive, 2**n between [32, 256] input 'bounds' :[5, 8]
    'log_scale': False,
  },
  {
    'name': "layer_2_pow2",
    'type': "range", #'range'
    'value_type': "int",
    'bounds' :[5, 8],    # Inclusive, 2**n between [32, 256] input 'bounds' :[5, 8]
    'log_scale': False,
  },
]

# Set parameter constraints
parameter_constraints = []

# Create our environment
try:
  env.close()
except NameError:
  pass
env = CantileverEnv(_env_gen_kwargs)
action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[0]), sigma=0.1 * np.ones(env.action_space.shape[0]))

# Construct path to Ax experiment snapshot file
ax_snapshot_path = os.path.join(settings['save_dir'], f"{settings['ax_experiment_name']}.json")

# Load experiment from snapshot if it exists, otherwise create a new one
if os.path.exists(ax_snapshot_path):
  print(f"Loading experiment from snapshot: {ax_snapshot_path}")
  ax_client = AxClient.load_from_json_file(ax_snapshot_path)
else:
  print(f"Creating new experiment. Snapshot to be saved at {ax_snapshot_path}.")
  ax_client = AxClient(
      random_seed=settings['seed'],
      verbose_logging=settings['verbose_ax']
  )
  ax_client.create_experiment(
      name=settings['ax_experiment_name'],
      parameters=hparams,
      objectives={settings['ax_objective_name']: ObjectiveProperties(minimize=False)},
      parameter_constraints=parameter_constraints,
  )

# Choo choo! Perform trials to optimize hyperparameters
while True:

  # Get next hyperparameters and end experiment if we've reached max trials
  next_hparams, trial_index = ax_client.get_next_trial()
  if trial_index >= settings['num_trials']:
    break

  # Show that we're starting a new trial
  if settings['verbose_trial'] > 0:
    print(f"--- Trial {trial_index} ---")

  # Perform trial
  avg_ep_rew = do_trial(settings, next_hparams)
  ax_client.complete_trial(
      trial_index=trial_index,
      raw_data=avg_ep_rew,
  )

  # Save experiment snapshot
  ax_client.save_to_json_file(ax_snapshot_path)
  #model = ax_client.generation_strategy.model
  #render(interact_contour(model=model, metric_name=settings['ax_objective_name']))

#Log environment parameters
with open(settings["save_dir"] +'\\Environment_config.txt', 'w') as txt_file:
    for key, value in _env_gen_kwargs.items():
        txt_file.write(f'{key}: {value}\n')
print("EnvironmentConfiguration logged to 'Environment_config.txt' successfully!")

#%%
# Get from W&B dashboard
settings = {
    'wandb_project': "cantilever-ax-hpo-Mode-1-2-3-evaluation",
    'model_name': "ppo-cantilever-Mode-1-2-3-evaluation",
    'ax_experiment_name': "ppo-cantilever-experiment-Mode-1-2-3-evaluation",
    'ax_objective_name': "avg_ep_rew",
    'replay_buffer_name': None,
    'save_dir': "checkpoints-Mode-1-2-3-evaluation",
    'checkpoint_freq': 5120,
    'steps_per_test': 1024,
    'tests_per_check': 2,
    'total_steps': 102_400,
    'num_trials': 50,
    'seed': 42,
    'verbose_ax': False,
    'verbose_wandb': False,
    'verbose_train': 0,
    'verbose_log': 0,
    'verbose_test': 0,
    'verbose_trial': 1,
}

hparams = {
    'n_envs': 1,
    'learning_rate': 0.00025355300129260107,
    'steps_per_update_pow2': 6,
    'batch_size_div_pow2': 0,
    'gamma': 0.935130460798928,
    'entropy_coef': 0.07607439564868153,
    'layer_1_pow2': 8,
    'layer_2_pow2': 8,
    'layer_3_pow2': 8,
    'layer_4_pow2': 8,
    'layer_5_pow2': 8,
    'gae_lambda' : 0.95,
    "vf_coef" : 0.6500169535752682
    }

# Create our environment
# environments setup
_env_gen_kwargs = {"elem_size": 0.005, #[m]
             "geo_path" : geo_path, 
             "core_path": path, 
             "seed" : 42,
             "max_step" : 5,
             "mat_path": mat_path,
             "sim_modes": [0,1,2],
             "num_sensors": 2,
             "mat_param" : [1.0, 1.0],
             "temp"       : 10,
             "render" : False,
             "eps_length" : 1024,
             "mode_shape_folder_name" : "mode_shape_folder"
    }
#%%
try:
  env.close()
except NameError:
  pass
env = CantileverEnv(_env_gen_kwargs)
  
# Train model
_ = do_trial(settings, hparams)
  
#%%

# Model and video settings
MODEL_FILENAME = "checkpoints-Mode-1-2-3-evaluation/ppo-cantilever-Mode-1-2-3-evaluation_40960.zip"
# Create our environment
#try:
#  env.close()
#except NameError:
#  pass

_env_gen_kwargs = {"elem_size": 0.005, #[m]
             "geo_path" : geo_path, 
             "core_path": path, 
             "seed" : 42,
             "max_step" : 5,
             "mat_path": mat_path,
             "sim_modes": [0,1,2],
             "num_sensors": 2,
             "mat_param" : [1.0, 1.0],
             "temp"       : 10,
             "render" : False,
             "eps_length" : 1024,
             "mode_shape_folder_name" : "mode_shape_folder"
    }
print('Start Time', datetime.datetime.now(datetime.timezone.utc))
try:
  env.close()
except NameError:
  pass
env = CantileverEnv(_env_gen_kwargs)

# Load the model
model = sb3.PPO.load(MODEL_FILENAME)
print('Stop Time', datetime.datetime.now(datetime.timezone.utc))
#%%
# Test the model
print('Start Time', datetime.datetime.now(datetime.timezone.utc))
ep_len, ep_rew, avg_ep_std, avg_step_time, eps_reward_metric_sum, eps_reward_metric_final = test_agent(
    env,
    model,
    max_steps=102_400_000,
)
print('Stop Time', datetime.datetime.now(datetime.timezone.utc))
print(f"Episode length: {ep_len}, reward: {ep_rew}, avg step time: {avg_step_time}, Episode test reward metric sum : {eps_reward_metric_sum}, Episode test reward metric final : {eps_reward_metric_final}")