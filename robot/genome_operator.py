from . import activations


def nodes_by_layer(bot) : 
    
    shape_of_cppn = bot.params["shape_of_cppn"]
    node = 0 
    nodes_by_layer = []
    for index_of_layer in range(len(shape_of_cppn)) :
        branches = len(shape_of_cppn[index_of_layer])
        nodes_on_layer = []
        for branch in range(branches) :
            nodes_on_branch = []
            for _ in range(shape_of_cppn[index_of_layer][branch]) :
                nodes_on_branch.append(node)
                node += 1
            nodes_on_layer.append(nodes_on_branch)
        nodes_by_layer.append(nodes_on_layer)
    return nodes_by_layer        


def incoming_by_node(previous_layer, current_layer) :
    pairs = []
    if len(previous_layer) == 1 and len(current_layer) == 1 :
        for node in current_layer[0] :
            pairs.append((node, previous_layer[0]))
    elif len(previous_layer) == 1 and len(current_layer) > 1 :
        for branch in range(len(current_layer)) :
            for node in current_layer[branch] :
                pairs.append((node, previous_layer[0]))
    elif len(previous_layer) > 1 and len(current_layer) > 1 :
        for branch in range(len(current_layer)) :
            for node in current_layer[branch] :
                pairs.append((node, previous_layer[branch]))
    return pairs

def generate_genome_first_generation(bot) : 
    output_activation_function = activations.identity
    nodes_by_layer = bot.nodes_by_layer
    function_pool = bot.function_pool

    connections = {}
    biases = {}
    activation_functions = {}
    list_of_keys = list(function_pool.keys())

    for index_of_layer in range(len(nodes_by_layer)) :
        if index_of_layer == 0 :
            previous_layer = nodes_by_layer[index_of_layer]
            continue
        current_layer = nodes_by_layer[index_of_layer]
        for node, previous_nodes in incoming_by_node(previous_layer, current_layer) :
            for previous_node in previous_nodes :
                weight = bot.params["range_weight"] * bot._rng.uniform(-1, 1)
                connections[(previous_node, node)] = weight
            b = bot.params["range_bias"] * bot._rng.uniform(-1, 1)
            c = bot._rng.integers(0, len(list_of_keys))
            act_function = function_pool[list_of_keys[c]]
            biases[node] = b
            activation_functions[node] = act_function if index_of_layer != len(nodes_by_layer) - 1 else output_activation_function
        previous_layer = current_layer
    return connections, biases, activation_functions

    