


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

def create_cppn(bot) : 

    nodes_by_layer = bot.nodes_by_layer
    connections = bot.cppn_connections
    bias = bot.biases
    functions = bot.functions
    agg = sum
    response = 1

    
    node_evals = []
    input_nodes = [node for node in nodes_by_layer[0][0]]
    output_nodes = [node for branch in nodes_by_layer[-1] for node in branch]

    for index_of_layer in range(len(nodes_by_layer)) :
        if index_of_layer == 0 :
            previous_layer = nodes_by_layer[index_of_layer]
            continue

        current_layer = nodes_by_layer[index_of_layer]
        for node, previous_nodes in incoming_by_node(previous_layer, current_layer) :
            inputs_of_node = []
            for previous_node in previous_nodes :
                weight = connections[(previous_node, node)]
                inputs_of_node.append((previous_node, weight))
            activation_function = functions[node]
            node_bias = bias[node]
            node_evals.append((node, activation_function, agg, node_bias, response, inputs_of_node))

        previous_layer = current_layer

    return node_evals, input_nodes, output_nodes   


def activate(node_evals, input_nodes, output_nodes, inputs):
    values = {}
    for key, value in zip(input_nodes, inputs) : 
        values[key] = value
    
    for node, activation_function, agregation_function, bias, response, inputs_of_node in node_evals : 
        node_inputs = []
        for previous_node, weight in inputs_of_node : 
            node_inputs.append(values[previous_node] * weight)
        entering_node = agregation_function(node_inputs)
        values[node] = activation_function(bias + response * entering_node)
    return [values[node] for node in output_nodes]