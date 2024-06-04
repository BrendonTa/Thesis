import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import matplotlib.pyplot as plt
import numpy as np
#off-policy optimization for HER
model_class = SAC
# Available strategies (cf paper): future, final, episode
goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

class FigureRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        # Plot values (here a random variable)
        figure = plt.figure()
        figure.add_subplot().plot(np.random.random(3))
        # Close the figure after logging it
        self.logger.record("trajectory/figure", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()
        return True
#create the environment
env = gym.make("BinPick-v2",render_mode="human")
env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
assert reward == env.unwrapped.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
assert truncated == env.unwrapped.compute_truncated(obs["achieved_goal"], obs["desired_goal"], info)
assert terminated == env.unwrapped.compute_terminated(obs["achieved_goal"], obs["desired_goal"], info)


model = model_class(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=1,
        goal_selection_strategy=goal_selection_strategy,
    ),
    verbose=1,
    tensorboard_log="Thesis/Thesis-Mujoco/log/HER_Dense"
)
#comment out the PPO model call if 
model = sb3.PPO("MultiInputPolicy",env=env,verbose=2,tensorboard_log="Thesis/Thesis-Mujoco/log/PPO_Dense")
model.learn(total_timesteps=10_000,callback=FigureRecorderCallback())
#renders the learning proccess
env.render()
#goes through multiple
for _ in range(1000):
    
    action,_state = model.predict(obs,deterministic=True)  # agent policy that uses the observation and info
    
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        observation, info = env.reset()
env.close()