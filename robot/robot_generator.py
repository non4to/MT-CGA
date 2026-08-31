import numpy as np 
from . import network_manager
from evogym import is_connected, has_actuator, get_full_connectivity 


TYPES_OF_VOXELS = ["empty", "rigid", "soft", "horizontal", "vertical"]

def is_valid_robot(robot) :
    return (is_connected(robot) and has_actuator(robot))

def robot_get_full_connectivity(robot) :
    return get_full_connectivity(robot)

def generate_robot_body_from_network_and_env(bot, type_env) :
        body_nodes_evals, body_input_nodes, body_output_nodes = bot.body_nodes_evals, bot.body_input_nodes, bot.body_output_nodes
        body_shape = bot.params["body_shape"]
        bias = np.sqrt(2)/2
        if type_env == 0 or type_env == 1 :
            body_outputs = network_manager.activate(body_nodes_evals, body_input_nodes, body_output_nodes, [bias])
            formated = np.reshape(body_outputs, (body_shape[0], body_shape[1], len(TYPES_OF_VOXELS)))
            robot = np.argmax(formated, 2)
        # elif type_env == 1 :
        #     body_outputs = network_manager.activate(body_nodes_evals, body_input_nodes, body_output_nodes, [-bias])
        #     formated = np.reshape(body_outputs, (body_shape[0], body_shape[1], len(TYPES_OF_VOXELS)))
        #     robot = np.argmax(formated, 2)
        return robot

