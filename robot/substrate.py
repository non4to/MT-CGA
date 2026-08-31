import numpy as np 
import itertools as it


def extract_body_network_shape(bot):

    body_shape = bot.params["body_shape"]
    shape_of_substrate = [
        [1, 1, 1, 1, 1], 
        [2, body_shape[0], body_shape[1], 1, 1], # the last 2 dim must be the same if you want to implement a circle 
        [3, body_shape[0], body_shape[1], 5, 5]
    ]
    return shape_of_substrate

def shape_into_coordinates(shape) :
    """basically turn the shape in all of the coordinate for each layer, first indices is the layer"""
    number_of_layer = len(shape)
    first_indice = 1/number_of_layer
    all_interval = []

    if np.sign(shape[0][0]) == 1 :
        first_dim_locations = np.linspace(first_indice, 1.0, number_of_layer)

    elif np.sign(shape[0][0]) == -1 :
        first_dim_locations = np.linspace(-first_indice, -1.0, number_of_layer)

    for layer in shape :
        indice = abs(layer[0]) - 1
        # if im on body i must have last 2 coordinates acting as a circle and not 1 line, this is why i switched from 4 dimension to 5
        # to replace the 1 line by a surface so that the cppn identify each possibility of actuators for each voxels identically 
        # this was not the case before because i had lots of body full of 0 and 4, I think it work for tanaka due to the nature of neat itself, which was protecting 
        # different individuals, because mine does not do that, those full of 4 individuals take over the population too quickly (gen 1)
        # simply because of the nature of the subsrate itself i presume 


        location = first_dim_locations[indice]
        interval_array = []
        if np.sign(shape[0][0]) == 1  and indice != 0 :
            for x in layer[1:len(layer)-2] :
                interval_one_dimension = np.linspace(-1.0, 1, x) if (x>1) else [0.0]
                interval_array.append(interval_one_dimension)
            if layer[-2] == layer[-1] and layer[-1] != 1:
                
                number_of_points = layer[-1]
                angle = 2*np.pi/number_of_points
                radius = 1.0/2
                last_coords = []
                for i in range(number_of_points) :
                    last_coords.append([radius*np.cos(angle*i), radius*np.sin(angle*i)])
                
                raw_cart_product = list(it.product([location], *interval_array, last_coords))
                all_coords = [(*e[:-1], *e[-1]) for e in raw_cart_product]
                all_interval.append(all_coords)
            else :
                for x in layer[len(layer)-2:] :
                    interval_one_dimension = np.linspace(-1.0, 1, x) if (x>1) else [0.0]
                    interval_array.append(interval_one_dimension)
                all_coords = list(it.product([location], *interval_array))
                all_interval.append(all_coords)

        elif np.sign(shape[0][0]) == -1  and indice != 0 :
            for x in layer[1:len(layer)-2] :
                interval_one_dimension = np.linspace(-1.0, 1, x) if (x>1) else [0.0]
                interval_array.append(interval_one_dimension)
            if layer[-2] == layer[-1] and layer[-1] != 1 :
                
                number_of_points = layer[-1]
                angle = 2*np.pi/number_of_points
                radius = 1.0/2
                last_coords = []
                for i in range(number_of_points) :
                    last_coords.append([radius*np.cos(angle*i), radius*np.sin(angle*i)])
                
                raw_cart_product = list(it.product([location], *interval_array, last_coords))
                all_coords = [(*e[:-1], *e[-1]) for e in raw_cart_product]
                all_interval.append(all_coords)
            else :
                for x in layer[len(layer)-2:] :
                    interval_one_dimension = np.linspace(-1.0, 1, x) if (x>1) else [0.0]
                    interval_array.append(interval_one_dimension)
                all_coords = list(it.product([location], *interval_array))
                all_interval.append(all_coords)

        else :
            for x in layer[1:] :
                interval_one_dimension = np.linspace(-1.0, 1, x) if (x>1) else [0.0]
                interval_array.append(interval_one_dimension)


            all_coords = list(it.product([location], *interval_array))
            all_interval.append(all_coords)
    
    return all_interval


def extract_controller_network_shape(bot, grid_input_size) :

    body_shape = bot.params["body_shape"]
    controller_shape = [
        [-1, grid_input_size, grid_input_size, 1, 1],
        [-2, body_shape[0], body_shape[1], 1, 1], 
        [-3, body_shape[0], body_shape[1], 1, 1]
    ]
    return controller_shape