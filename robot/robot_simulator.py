import gymnasium as gym
import evogym.envs
import os, math
import numpy as np 
from . import network_manager

from evogym.utils import get_full_connectivity 


def _get_env_mode_env(bot, robot, type_env) :
    connections = get_full_connectivity(robot)
    env_name = bot.params["env_name"][type_env] 
    env = gym.make(env_name, body = robot, connections = connections)
    return env.unwrapped

def safe_observation_size_mode_env(bot, robot, type_env) :
    env = _get_env_mode_env(bot, robot, type_env)
    try :
        observation, _ = env.reset()
        return len(observation)
    finally :
        env.close()
        del env

def simulate_render_mode_env(bot, type_env, params) : 
    print('\n ----- Beginning simulation -----\n')

    robot = bot.shape
    env = _get_env_render_mode_env(robot, type_env, params)
    fitness = 0
    observation, _ = env.reset()

    actuators = env.get_actuator_indices("robot")
    inputs_size = math.ceil(math.sqrt(len(observation)))

    images = []
    controller_nodes_evals, controller_input_nodes, controller_output_nodes = bot.controller_nodes_evals, bot.controller_input_nodes, bot.controller_output_nodes
    for _ in range (params["sim_step"]) : 

        images.append(env.render())

        observation.resize(inputs_size**2)
        all_actions = network_manager.activate(controller_nodes_evals, controller_input_nodes, controller_output_nodes, observation)
        action = np.array([all_actions[i] for i in actuators])
        observation, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated 
        fitness += reward
        if done : 
            break 

    env.close()
    del env 
    print('\n==============================')
    print(f'Individual fitness : {fitness}')
    print('==============================\n')
    print('----- End of simulation -----\n')
    return images, fitness 

def _get_env_render_mode_env(robot, type_env, params) : 
    connections = get_full_connectivity(robot)
    env = gym.make(params["env_name"][type_env], body=robot, connections=connections, render_mode="rgb_array")
    return env.unwrapped

