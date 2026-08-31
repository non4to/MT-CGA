from . import network_manager 
import numpy as np 

def create_phenotype_network(bot, all_layers_coordinates, out_activation_function, activation_function, type_output) :
    agg = sum
    response = 1

    max_weight = bot.params["max_weight"]
    
    node_dict = {}
    idx = 0
    for layer in all_layers_coordinates :
        for node in layer :
            node_dict[tuple(node)] = idx 
            idx += 1

    node_evals = []
    idx_current_source_layer = 0 
    for layer in all_layers_coordinates[1:] :
        source_layer = all_layers_coordinates[idx_current_source_layer]
        for hidden_node in layer :
            one_towards_two = True 
            incomming_connections, bias = _connect_target_node_to_layer_without_bias_2_outputs(bot, hidden_node, source_layer, node_dict, one_towards_two, max_weight, type_output)
            act = out_activation_function if layer == all_layers_coordinates[-1] else activation_function
            node_evals.append(tuple( [node_dict[tuple(hidden_node)], act, agg, bias, response, incomming_connections] ))
        idx_current_source_layer += 1

    input_nodes = [node_dict[tuple(node_coor)] for node_coor in all_layers_coordinates[0]]
    output_nodes = [node_dict[tuple(node_coor)] for node_coor in all_layers_coordinates[-1]]

    return node_evals, input_nodes, output_nodes 

def _connect_target_node_to_layer_without_bias_2_outputs(bot, target_node_coordinates, source_layer_coordinates, node_dict, one_towards_two, max_weight, type_output) : 
    incomming_connections = []
    for source_node_coordinate in source_layer_coordinates :
        weight = _query_cppn_weight_2_outputs(bot, source_node_coordinate, target_node_coordinates, one_towards_two, max_weight, type_output)
        incomming_connections.append((node_dict[tuple(source_node_coordinate)], weight))
    bias = 0
    return incomming_connections, bias

def _query_cppn_weight_2_outputs(bot, coordinate1, coordinate2, one_towards_two, max_weight, type_output) :

    distance = _distance(coordinate1, coordinate2)

    nodes_evals, input_nodes, output_nodes = bot.nodes_evals, bot.input_nodes, bot.output_nodes

    if one_towards_two :
        inputs = [*coordinate1, *coordinate2, distance]
    else :
        inputs = [*coordinate2, *coordinate1, distance]

    # if one_towards_two :
    #     inputs = [*coordinate1, *coordinate2]
    # else :
    #     inputs = [*coordinate2, *coordinate1]

    output = network_manager.activate(nodes_evals, input_nodes, output_nodes, inputs)

    weight = output[type_output]

    if abs(weight) > 0.1 and abs(weight) < max_weight:
        return weight
    elif abs(weight) >= max_weight :
        return max_weight if weight > 0 else -max_weight
    else :
        return 0.0
    
def _distance(coordinate1, coordinate2) :
    array1 = np.array(coordinate1)
    array2 = np.array(coordinate2)
    return np.linalg.norm(array1 - array2)