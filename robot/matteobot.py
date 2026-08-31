import numpy as np
import math
import copy
import time 
import json
import pickle as pkl 
import imageio as io
import os 
from . import genome_operator, network_manager, substrate
from . import hyper_encoding, robot_generator, robot_simulator
from . import activations
# from .parallel_tool import ParallelTool


# To read 
# I think a modif need to be done with hamming distance because a body of a child might not differ from his parent but his brain could be different, so it need to be evaluated
# maybe just evaluate again even if same body could be the solution 

# To do 
# the function s are working the same as yours except the get random, you need to give as parameters the parameters.json loaded, and the type of env in integer of the env
# let me explain env_name = ["Walker-v0", "BridgeWalker-v0"] so 0 is for the first so that env = env_name[0] and same with 1. 
# Ill let you do the modif for the name I used and repeted in the paranmeters.json 
# The best would be to use integers always for the creation of the first gen where you use get random. 
# then the env is an attribute and the child gets the env of the first parent, so the case he is in, so everything is good 
# I let you modify the evaluate function for the environment again, because I dont understand how yours work, now my function takes the name of the env and transfo4rm it into integer 0 or 1 inside. 
# the best would be that each of my function that need the env take an integer 0 or 1

# General 
# I do the same as you, if the body is not valid I try and try again until one is good (I do this for the crossover, mutate and also the get random)
# Everything is stored as attribute, each neural network and so one, to look at the neural network, see the setattr that i put
# for ex. the nodes_evals is for the cppn, the body_nodes_evals is for the body neural network and the controller_nodes_evals is for the controller
# Also go look at the other files I created, I put them in the same folder to make my life easy but dont hesitate to create a sub folder to have all the repository clean. 

# Implemented 
# Right now, the same body will be generated for the two environments , but the controller is different, due to that and because I didnt want to parallel process of parallel proccess, the ram might increase to the infinite as the sim goes on 
# It should be fine for 500 gen if you do not use too much for the algo and that you clean the memory. 
# But if the exp stops because of that, we'll need to find a solution for that. 
# I think everything is good, I highly advise you to read this file first and then go to the the other file that serve as tools. I guess the name of my functions explain well enough what they do. 

# Also my evaluate returns the same as you with fitness and time, modify it if you dont want the time 
# To save the genome (which is deterministic so we can only save that) well need to use pickle because we cqannot put iot into json, but we need to be careful to rebuilt the robot and dont change the positions of the file

# Also, if you want the distance Im using go look in the distance.py tool of my repository sgr-ecs-matteo, you should find everything. 

# Then good luck and don't forget to ask me questions if you have any 
        
def evaluate(bot:"Mattebot", environment, n_steps, info:list) -> float:
    """function that evaluates the robot and returns its fitness value
    """
    stime = time.time()
    type_env = 0 if environment == bot.params["env_name"][0] else 1
    former_type_env = copy.deepcopy(bot.type_env)
    if type_env != bot.type_env : # here I associated when creating body brain the env to the robot, when you check in cga if the robot need to be evaluated again, I also check here to create a new controller associated to the good task 
        bot.generate_body_brain(type_env) # the body doesnt change, just the controller due to the difference in the obs vector 
        setattr(bot, "type_env", former_type_env) # the body brain function write over the type of env so I put it back 
    robot = bot.shape 
    controller_nodes_evals, controller_input_nodes, controller_output_nodes = bot.controller_nodes_evals, bot.controller_input_nodes, bot.controller_output_nodes
    try:
        env = robot_simulator._get_env_mode_env(bot, robot, type_env) 
    except:
        filename = f"{info[1]}{os.sep}badbots{os.sep}badbot{bot.id}-gen{info[0]}.json"
        with open(filename, "w") as f:
            json.dump({
                "id": bot.id,
                "shape": bot.shape.tolist(),
                "generation":info[0],
                "type_env": type_env
            }, f)
        return -999999, 0 #evogym is weird, this bot is broken!

    try : 
        reward = 0  

        observation, _ = env.reset()
        actuators = env.get_actuator_indices("robot")
        inputs_size = math.ceil(math.sqrt(len(observation)))
        finished = False 

        for _ in range (n_steps) : 
            observation.resize(inputs_size**2)
            all_actions = network_manager.activate(controller_nodes_evals, controller_input_nodes, controller_output_nodes, observation)
            action = np.array([all_actions[i] for i in actuators])
            observation, step_reward, terminated, truncated, _ = env.step(action)

            reward += step_reward

            done = terminated or truncated 

            if done : 
                finished = True 
                break 
        return reward, time.time() - stime

    except Exception:
        filename = f"{info[1]}{os.sep}badbots{os.sep}badbot{bot.id}-gen{info[0]}.json"
        with open(filename, "w") as f:
            json.dump({
                "id": bot.id,
                "shape": bot.shape.tolist(),
                "generation":info[0],
                "type_env": type_env
            }, f)
        return -999999, 0 #evogym is weird, this bot is broken!

    finally : 
        env.close()
        del env

def evaluate_both_last(folder_dir, gen=500, n_steps=500, tasks=["Walker-v0", "BridgeWalker-v0"]) :
    pkl_path = os.path.join(folder_dir, "bots", f"generation_{gen}.pkl")
    with open(pkl_path, "rb") as file : 
        whole_gen = pkl.load(file)

    dict_of_results = {}

    for robot in whole_gen.values(): 
        task_list = robot.fit.keys()
        task = list(task_list)[0]

        pos = 0 if task == tasks[0] else 1
        other_pos = 1 if pos == 0 else 0

        other_task = tasks[other_pos]

        state = [501, folder_dir]
        reward, _ = evaluate(robot, other_task, n_steps, state)

        dict_of_results[robot.id] = {}
        dict_of_results[robot.id]["original"] = task
        dict_of_results[robot.id]["other"] = other_task

        dict_of_results[robot.id][task] = robot.fit[task]
        dict_of_results[robot.id][other_task] = reward

    new_json_path = os.path.join(folder_dir, "last_gen.json")

    with open(new_json_path, "w") as file :
        json.dump(dict_of_results, file)

    return dict_of_results # just for beauty of things 

def render(filename, out_dir, pos, type_env, params): 
    # need to install ffmepg 
    print('\n----- Loading the object -----\n')
    with open(filename, "rb") as file : 
        grid = pkl.load(file)

    # pos being like (x,y)
    bot = grid[pos]

    bot.generate_body_brain(type_env)

    images, _ = robot_simulator.simulate_render_mode_env(bot, type_env, params)

    print(' ----- Saving the Simulation -----\n')
    id = bot.id

    os.makedirs(out_dir, exist_ok=True)
    videogifpath = os.path.join(out_dir, 'id_{}_pos_{}_env_{}.gif'.format(id, pos, type_env))
    videomp4path = os.path.join(out_dir, 'id_{}_pos_{}_env_{}.mp4'.format(id, pos, type_env))
    io.mimwrite(videomp4path, images, fps=30, macro_block_size=1)
    io.mimwrite(videogifpath, images, fps=30)

    print(' ----- Simulation saved -----\n')
    

def get_random(params, type_env, w = 5, h = 5, rng=None):
    r = HaploidMattebot(params, rng)
    r.randomize(type_env, w, h)
    return r

class HaploidMattebot():
    def __init__(self, params, rng:np.random=None): # params is the dictionary of the parameters.json so just load the json into obj and give this obj as params
        """Bots have a random generator as input
        """
        self.params = params
        self.shape = np.array([[1]])
        self._rng = rng if rng is not None else np.random.default_rng()
        # self.fit = -9999
        self.fit = {}
        self.id = -1
        self.age = 0
        # self.parallel_tool = ParallelTool(self, self.params["cpus_for_controller"]) # not used now, see what happens with ram 
        self.function_pool = self._create_pool()
        
    def randomize(self, type_env, w = 5, h = 5) :
        count = 0
        while True:
            self.first_generation_genome()
            self.generate_body_brain(type_env)
            if self.valid:
                break
            count += 1
            if (count > 5000):
                raise Exception("Can't find a valid random robot after 5000 tries!")

    def first_generation_genome(self) -> "Mattebot":
        """Robot calls this function
        Output is the first generation of the robot
        """
        nodes_by_layer = genome_operator.nodes_by_layer(self)
        setattr(self, "nodes_by_layer", nodes_by_layer)
        
        connections, biases, activation_functions, = genome_operator.generate_genome_first_generation(self)

        setattr(self, "cppn_connections", connections)
        setattr(self, "biases", biases)
        setattr(self, "functions", activation_functions)


    def generate_body_brain(self, type_env) -> "Mattebot": 
        nodes_evals, input_nodes, output_nodes = network_manager.create_cppn(self)

        setattr(self, "type_env", type_env)
        setattr(self, "nodes_evals", nodes_evals)
        setattr(self, "input_nodes", input_nodes)
        setattr(self, "output_nodes", output_nodes)

        body_substrate_shape = substrate.extract_body_network_shape(self)
        body_substrate = substrate.shape_into_coordinates(body_substrate_shape)
        act_func = np.tanh
        output_act_func = np.tanh

        body_nodes_evals, body_input_nodes, body_output_nodes = hyper_encoding.create_phenotype_network(self, body_substrate, output_act_func, act_func, 0)

        setattr(self, "body_nodes_evals", body_nodes_evals)
        setattr(self, "body_input_nodes", body_input_nodes)
        setattr(self, "body_output_nodes", body_output_nodes)

        robot_grid = robot_generator.generate_robot_body_from_network_and_env(self, type_env)
        valid = robot_generator.is_valid_robot(robot_grid)

        setattr(self, "valid", valid)
        setattr(self, "shape", robot_grid)
        if valid :
            connections = robot_generator.robot_get_full_connectivity(robot_grid)
            setattr(self, "connections", connections)
            # I put the following lines like that because we could have parallel inside parallel and hard to manage, what well see is basically that the ram is increasing but for 500 gen should be ok, if not work, well have to find solutions 
            # chunk = [(self, robot_grid, type_env)]
            # results = self.parallel_tool.run(robot_simulator.safe_observation_size_mode_env, chunk)
            # size_of = [size for size in results]
            # size = size_of[0]
            size = robot_simulator.safe_observation_size_mode_env(self, robot_grid, type_env)

            setattr(self, "size", size)

            act_func = np.tanh
            out_act_func = activations.controller_out
            if size <= 0 : 
                setattr(self, "valid", False)
            else : 
                grid_input_size = math.ceil(math.sqrt(size))
                controller_substrate_shape = substrate.extract_controller_network_shape(self, grid_input_size)
                controller_substrate = substrate.shape_into_coordinates(controller_substrate_shape)
                controller_nodes_evals, controller_input_nodes, controller_output_nodes = hyper_encoding.create_phenotype_network(self, controller_substrate, out_act_func, act_func, 1)

                setattr(self, "controller_nodes_evals", controller_nodes_evals)
                setattr(self, "controller_input_nodes", controller_input_nodes)
                setattr(self, "controller_output_nodes", controller_output_nodes)

        

    def crossover(self, mate, type_env=None) -> "Mattebot":
        """Robot calls this function
        Imput is another robot to crossover
        Output is the bot resulted from the crossover
        """ 
        if type_env is None: # this because the env is the one from parent 1 
            type_env = self.type_env
        cppn_connections = {}
        biases = {}
        functions = {}

        count = 0

        while True :
            count += 1 
            
            for index_of_layer in range(len(self.nodes_by_layer)) :
                if index_of_layer == 0 :
                    previous_layer = self.nodes_by_layer[index_of_layer]
                    continue

                current_layer = self.nodes_by_layer[index_of_layer]

                for node, previous_nodes_by_layer in genome_operator.incoming_by_node(previous_layer, current_layer) :
                    choice = self._rng.integers(0, 2)
                    if choice == 0 :
                        biases[node] = copy.deepcopy(self.biases[node])
                        functions[node] = copy.deepcopy(self.functions[node])
                        for previous_node in previous_nodes_by_layer :
                            cppn_connections[(previous_node, node)] = copy.deepcopy(self.cppn_connections[(previous_node, node)])
                    elif choice == 1 :
                        biases[node] = copy.deepcopy(mate.biases[node])
                        functions[node] = copy.deepcopy(mate.functions[node])
                        for previous_node in previous_nodes_by_layer :
                            cppn_connections[(previous_node, node)] = copy.deepcopy(mate.cppn_connections[(previous_node, node)])
                previous_layer = current_layer

            nodes_by_layer = copy.deepcopy(self.nodes_by_layer)

            child_params = copy.deepcopy(self.params)

            child = HaploidMattebot(child_params, self._rng)

            setattr(child, "cppn_connections", cppn_connections)
            setattr(child, "biases", biases)
            setattr(child, "functions", functions)
            setattr(child, "nodes_by_layer", nodes_by_layer)

            child.generate_body_brain(type_env)

            if child.valid :
                return child

            if count > 5000 :
                raise Exception("Can't find a valid random robot after 5000 tries!") 
        
    def mutate(self, type_env=None) -> "Mattebot":
        """Robot call this function
        Output is the bot resulted from the mutation
        """
        if type_env is None:
            type_env = self.type_env
        params = copy.deepcopy(self.params)

        new_bot = HaploidMattebot(params, self._rng)

        cppn_connections, biases, functions, nodes_by_layer = copy.deepcopy(self.cppn_connections), copy.deepcopy(self.biases), copy.deepcopy(self.functions), copy.deepcopy(self.nodes_by_layer)

        setattr(new_bot, "cppn_connections", cppn_connections)
        setattr(new_bot, "biases", biases)
        setattr(new_bot, "functions", functions)
        setattr(new_bot, "nodes_by_layer", nodes_by_layer)

        count = 0 
        while True :
            count += 1 

            cppn_connections, biases, functions, nodes_by_layer = copy.deepcopy(self.cppn_connections), copy.deepcopy(self.biases), copy.deepcopy(self.functions), copy.deepcopy(self.nodes_by_layer)

            output_nodes = [node for branch in new_bot.nodes_by_layer[-1] for node in branch]

            for node in new_bot.cppn_connections :
                if self._rng.uniform() < new_bot.params["threshold_weight"] :
                    new_bot.cppn_connections[node] += self._rng.normal(loc = 0, scale = new_bot.params["sigma_weight"])
                    if abs(new_bot.cppn_connections[node]) > new_bot.params["max_weight_cppn"] :
                        new_bot.cppn_connections[node] = np.sign(new_bot.cppn_connections[node]) * new_bot.params["max_weight_cppn"]

            for node in new_bot.biases :
                if self._rng.uniform() < new_bot.params["threshold_bias"] :
                    new_bot.biases[node] += self._rng.normal(loc = 0, scale = new_bot.params["sigma_bias"])
                    if abs(new_bot.biases[node]) > new_bot.params["max_bias_cppn"] :
                        new_bot.biases[node] = np.sign(new_bot.biases[node]) * new_bot.params["max_bias_cppn"]

            list_of_keys = list(new_bot.function_pool.keys())
            for node in new_bot.functions :
                if node in output_nodes :
                    continue
                if self._rng.uniform() < new_bot.params["threshold_function"] :
                    choice = self._rng.integers(0, len(list_of_keys))
                    new_function = new_bot.function_pool[list_of_keys[choice]]
                    while new_bot.functions[node] == new_function :
                        choice = self._rng.integers(0, len(list_of_keys))
                        new_function = new_bot.function_pool[list_of_keys[choice]]
                    new_bot.functions[node] = new_function

            new_bot.generate_body_brain(type_env)

            if new_bot.valid :
                return new_bot

            setattr(new_bot, "cppn_connections", cppn_connections)
            setattr(new_bot, "biases", biases)
            setattr(new_bot, "functions", functions)
            setattr(new_bot, "nodes_by_layer", nodes_by_layer)

            if count > 5000 :
                raise Exception("Can't find a valid random robot after 5000 tries!")



    def _create_pool(self):
        return {name: activations.REGISTRY[name] for name in self.params["function_pool"]}  

    
    
