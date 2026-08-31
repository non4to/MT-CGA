import numpy as np 
import pickle as pkl 
import json, csv
import  os 
from multiprocessing import Pool
from datetime import datetime, timedelta
from collections import Counter

def hamming_distance(shape1: list, shape2: list) -> float:
    A = np.array(shape1).flatten()
    B = np.array(shape2).flatten()
    
    maxDist = max(A.size, B.size) #A and B MUST have the same size!
    dist = np.sum(A != B)
    return dist/maxDist

def same_activation(f1, f2) :
    samples = np.linspace(-5, 5, 10)
    out1 = np.round([f1(x) for x in samples], 2)
    out2 = np.round([f2(x) for x in samples], 2)
    return np.array_equal(out1, out2)

def distance_expressed_genome(bot1, bot2) : 
    act_function_distance = 0
    weight_distance = 0 
    bias_distance = 0 

    for node_eval1, node_eval2 in zip(bot1.nodes_evals, bot2.nodes_evals) :
        node1, activation_function1, agregation_function1, bias1, response1, inputs_of_node1 = node_eval1
        node2, activation_function2, agregation_function2, bias2, response2, inputs_of_node2 = node_eval2

        list_of_weight1 = [weight for previous_node, weight in inputs_of_node1]
        list_of_weight2 = [weight for previous_node, weight in inputs_of_node2]

        for weight1, weight2 in zip(list_of_weight1, list_of_weight2) :
                        weight_distance += np.linalg.norm(np.array(weight1) - np.array(weight2))

        bias_distance += np.linalg.norm(np.array(bias1) - np.array(bias2))

        act_function_distance += 0 if same_activation(activation_function1, activation_function2) else 1

        number_of_nodes = len(bot1.nodes_evals) 

    return act_function_distance, weight_distance, bias_distance, number_of_nodes

def gap_expressed_genome_compared_to_reference(bot1, bot2, act_function_distance, weight_distance, bias_distance) :
    other_act_function_distance, other_weight_distance, other_bias_distance, number_of_nodes = distance_expressed_genome(bot1, bot2)

    difference_act_function = other_act_function_distance - act_function_distance
    difference_weight = other_weight_distance - weight_distance
    difference_bias = other_bias_distance - bias_distance

    return difference_act_function, difference_weight, difference_bias

def mean_genome(filename) :
    # I guess supposed to take filename of the pickle of the generation to then load every ind 
    # now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    # print(f"[{now}] [PID {os.getpid()}] beggining {os.path.basename(filename)}", flush=True)

    with open(filename, "rb") as file : 
            grid = pkl.load(file)

    total_weight = 0
    total_bias = 0
    total_act_function = 0
    count = 0

    explored = set()

    for pos in grid.keys() :
        bot = grid[pos]

        if pos not in explored :
            explored.add(pos)

        for other_pos in grid.keys() :
            if other_pos not in explored : 
                other_bot = grid[other_pos]

                act_function_distance, weight_distance, bias_distance, number_of_nodes = distance_expressed_genome(bot, other_bot)
                count += 1

                total_weight += weight_distance 
                total_bias += bias_distance 
                total_act_function += act_function_distance 

    act_function_distance = total_act_function / count
    weight_distance = total_weight / count
    bias_distance = total_bias / count

    return act_function_distance, weight_distance, bias_distance 

def mean_all_genome(folder_dir, n_processors:int) :
    total_weight = 0
    total_bias = 0
    total_act_function = 0
    count = 0

    final_folder = os.path.join(folder_dir, "bots")
    pklArchives = []
    for filename in os.listdir(final_folder) :
        if filename.endswith(".pkl") : 
            adress = os.path.join(final_folder, filename)
            pklArchives.append(adress)
            
    with Pool(processes=n_processors) as pool:
        results = pool.map(mean_genome, pklArchives)
        
    for act_function_distance, weight_distance, bias_distance in results:
        total_weight += weight_distance
        total_bias += bias_distance
        total_act_function += act_function_distance
        count += 1
        
    act_function_distance = total_act_function / count
    weight_distance = total_weight / count
    bias_distance = total_bias / count
    return act_function_distance, weight_distance, bias_distance

# ex 
# genotypic_threshold_distance = {"act_function":3, "weight":10, "bias":5, "body":3.0}

def set_of_diff(folder_dir: str, distance_function, genotypic_threshold_distance:dict, genotypic_threshold_constant:int,type_of_distance:str, n_processors:int=5, threshold=0.7, max_gen = 500) : 

    dict_of_set = {} # with task separated 
    other_set = set() # with both task in one set 
    dict_of_species = {}   
    species = {}  
    dict_of_preval_id = {}

    json_path = os.path.join(folder_dir, "robots_log.jsonl")
    with open(json_path, "r") as file :
        robots = [json.loads(line) for line in file if line != "\n"]
    max = {}

    #get the maximum for each task 
    for robot in robots : 
        if robot["gen"] > max_gen :
            continue
        task_list = list(robot["fit"].keys())
        task = task_list[0]
        if task not in max.keys() :
            max[task] = 0

        if robot["fit"][task] > max[task] :
            max[task] = robot["fit"][task]

    for robot in robots : 
        if robot["gen"] > max_gen :
            continue
        task_list = list(robot["fit"].keys())
        task = task_list[0]
        if (robot["fit"][task] >= max[task] * threshold) and (robot["id"] not in dict_of_preval_id.keys()):
            dict_of_preval_id[robot["id"]] = robot 

    ##########################################
    # PHENOTYPIC
    ##########################################
    if type_of_distance == "phenotypic":
        lastNow = datetime.now()
        for i, key in enumerate(dict_of_preval_id.keys()):
            robot_data = dict_of_preval_id[key]
            task = list(robot_data["fit"].keys())[0]

            stop = False
            for specy in species.keys():
                other_key = species[specy]["representative"]
                other_bot = dict_of_preval_id[other_key]
                phenotype_distance = distance_function(robot_data["shape"], other_bot["shape"])
                if phenotype_distance <= genotypic_threshold_distance["body"]:
                    stop = True
                    break

            if not stop:
                other_set.add(key)
                leng = len(species.keys())
                species[leng + 1] = {}
                species[leng + 1]["representative"] = key
                species[leng + 1]["set"] = set([key])
            else:
                species[specy]["set"].add(key)

    ##########################################
    # GENOTYPIC
    ##########################################
    elif type_of_distance == "genotypic": 
    
        ref_dist_act_function , ref_dist_weight , ref_dist_bias = mean_all_genome(folder_dir, n_processors)
        genotypic_threshold_distance["act_function"] = ref_dist_act_function * genotypic_threshold_constant
        genotypic_threshold_distance["weight"] = ref_dist_weight * genotypic_threshold_constant
        genotypic_threshold_distance["bias"] = ref_dist_bias * genotypic_threshold_constant
        dict_of_species = {} # species by task {task : {specie : {representative : id, set of ids : set}} }
        species = {} # species regardless of the task 
        loaded = {}

        minimumGen = 230
        every_pkl = {}

        #this is to define which gen to on RAM (faster to compare)
        diff_gen = set()
        for key in dict_of_preval_id.keys() : 
            gen = dict_of_preval_id[key]["gen"]
            if (gen < minimumGen): continue
            if gen not in diff_gen :
                diff_gen.add(gen)
        #load gens on RAM
        for gen in diff_gen :
            now = datetime.now()
            if gen%100==0:
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] loading {gen}...")
            gen_file_path = os.path.join(folder_dir, "bots", "generation_{}.pkl".format(gen))
            with open(gen_file_path, "rb") as file : 
                whole_gen = pkl.load(file)
            every_pkl[gen] = whole_gen
        print(f"Amount of gens: {len(diff_gen)}")
        lastNow = datetime.now()

        #for each robot that have passed the fitness threshold
        for i, key in enumerate(dict_of_preval_id.keys()) :
            if datetime.now() - lastNow >= timedelta(seconds=30):
                lastNow = datetime.now()
                print(f"[{lastNow.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Robot {i}/{len(dict_of_preval_id.keys())}")

            #build dictionaries that need task as keys (this is not being used)
            task_list = list(dict_of_preval_id[key]["fit"].keys())
            task = task_list[0]
            if task not in dict_of_set.keys() : 
                dict_of_set[task] = set([key])
                dict_of_species[task] = {} #species inside task

            # load genome 
            gen = dict_of_preval_id[key]["gen"]
            # print(f"Loading gen {gen}...")
            
            if gen not in diff_gen:
                gen_file_path = os.path.join(folder_dir, "bots", f"generation_{gen}.pkl")
                with open(gen_file_path, "rb") as file : 
                    whole_gen = pkl.load(file)
                bot = [bot for bot in whole_gen.values() if bot.id == key][0]
            else:
                bot = [bot for bot in every_pkl[gen].values() if bot.id == key][0]

            stop = False 
            otherBotList = []
            for specy in species.keys() :
                other_key = species[specy]["representative"]
                if other_key not in loaded.keys() :
                    gen = dict_of_preval_id[other_key]["gen"]
                    if gen not in diff_gen:
                        gen_file_path = os.path.join(folder_dir, "bots", f"generation_{gen}.pkl")
                        with open(gen_file_path, "rb") as file : 
                            whole_gen = pkl.load(file) 
                        other_bot = [bot for bot in whole_gen.values() if bot.id == other_key][0]
                    else:
                        other_bot = [bot for bot in every_pkl[gen].values() if bot.id == other_key][0]
                    loaded[other_key] = other_bot
                else:
                    other_bot = loaded[other_key]

                act_function_distance, weight_distance, bias_distance, _ = distance_function(bot, other_bot)
                if act_function_distance <= genotypic_threshold_distance["act_function"] and weight_distance <= genotypic_threshold_distance["weight"] and bias_distance <= genotypic_threshold_distance["bias"] :
                    stop = True
                    break

            if not stop : 
                other_set.add(key)
                leng = len(species.keys())
                species[leng + 1] = {}
                species[leng + 1]["representative"] = key
                species[leng + 1]["set"] = set([key])
            else : 
                species[specy]["set"].add(key)

    return dict_of_set, other_set, dict_of_species, species

def print_dict(dictionary:dict):
    for key, content in dictionary.items():
        print(f"{key}: {content}")

def build_all_genotypic_speciation_archives(n_processors:int = 20, genotypic_threshold_constant:float=0.25, threshold:float=0.8):
    logFolder = "/home/flalipe/Projects/Airob/log"
    progressFile = os.path.join(logFolder, "speciationProgress.txt")
    folderList = []
    finishedFolders = ["mixed-randomSelectAge50-20x5_seed7_CGA_08271513",
                       "mixed-randomSelectAge50-20x5_seed49_CGA_08271831",
                       "mixed-randomSelectAge50-20x5_seed343_CGA_08281513",
                       "bridge-randomSelectAge50-20x5_seed49_CGA_08281122",
                        "bridge-randomSelectAge50-20x5_seed343_CGA_08290758", 
                        "bridge-randomSelectAge50-20x5_seed7_CGA_08271609",
                        "walker-randomSelectAge50-20x5_seed7_CGA_08271602", 
                        "walker-randomSelectAge50-20x5_seed49_CGA_08272200", 
                        "walker-randomSelectAge50-20x5_seed343_CGA_08280411", 
                        "walker-randomSelectAge50-20x5_seed2401_CGA_08290405", 
                        "walker-randomSelectAge50-20x5_seed16807_CGA_08282148", 
                        "bridge-randomSelectAge50-20x5_seed2401_CGA_08291229" ,
                        "bridge-randomSelectAge50-20x5_seed16807_CGA_08291553" 
                       ]

    for folder in os.listdir(logFolder):
        if folder in finishedFolders: continue
        if folder.startswith("bridge") or folder.startswith("mixed") or folder.startswith("walker"):
            folder = f"{logFolder}{os.sep}{folder}"
            folderList.append(folder)


    for folder in folderList:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            msg = f"[{now}] Working on {folder}... \n"
            print(msg)
            with open(progressFile, "a", encoding="utf-8") as f:
                f.write(msg)
            _, b, _, d = set_of_diff(folder_dir=folder,
                                    distance_function=distance_expressed_genome,
                                    type_of_distance="genotypic",
                                    n_processors=n_processors,
                                    genotypic_threshold_distance={"act_function":3, "weight":10, "bias":5, "body":3.0},
                                    genotypic_threshold_constant=genotypic_threshold_constant,
                                    threshold=threshold,
                                    max_gen=500)
            
            bPath = os.path.join(folder, "representatives.csv")
            with open(bPath, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["robot_id"])
                for robot_id in b:
                    writer.writerow([robot_id])

            dPath = os.path.join(folder, "speciation.csv")
            with open(dPath, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["species_id", "representative_id", "member_robot_ids"])  # Header
                for species_id, data in d.items():
                    members_str = ";".join(map(str, data["set"]))
                    writer.writerow([species_id, data["representative"], members_str])

def build_all_phenotypic_speciation_archives(n_processors:int = 20, phenotypic_threshold:float=0.15):
    logFolder = "/home/flalipe/Projects/Airob/log"
    progressFile = os.path.join(logFolder, "phenotypicSpeciationProgress.txt")
    folderList = []
    finishedFolders = []
    # finishedFolders = ["mixed-randomSelectAge50-20x5_seed7_CGA_08271513",
    #                 "mixed-randomSelectAge50-20x5_seed49_CGA_08271831",
    #                 "mixed-randomSelectAge50-20x5_seed343_CGA_08281513",
    #                 "bridge-randomSelectAge50-20x5_seed49_CGA_08281122",
    #                     "bridge-randomSelectAge50-20x5_seed343_CGA_08290758", 
    #                     "bridge-randomSelectAge50-20x5_seed7_CGA_08271609",
    #                     "walker-randomSelectAge50-20x5_seed7_CGA_08271602", 
    #                     "walker-randomSelectAge50-20x5_seed49_CGA_08272200", 
    #                     "walker-randomSelectAge50-20x5_seed343_CGA_08280411", 
    #                     "walker-randomSelectAge50-20x5_seed2401_CGA_08290405", 
    #                     "walker-randomSelectAge50-20x5_seed16807_CGA_08282148", 
    #                     "bridge-randomSelectAge50-20x5_seed2401_CGA_08291229" ,
    #                     "bridge-randomSelectAge50-20x5_seed16807_CGA_08291553" 
    #                 ]
    for folder in os.listdir(logFolder):
        if folder in finishedFolders: continue
        if folder.startswith("bridge") or folder.startswith("mixed") or folder.startswith("walker"):
            folder = f"{logFolder}{os.sep}{folder}"
            folderList.append(folder)

    # folderList = ["log/mixed-randomSelectAge50-20x5_seed7_CGA_08271513"]

    for folder in folderList:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            msg = f"[{now}] Working on {folder}... \n"
            print(msg)
            with open(progressFile, "a", encoding="utf-8") as f:
                f.write(msg)
            a, b, c, d = set_of_diff(folder_dir=folder,
                                    distance_function=hamming_distance,
                                    type_of_distance="phenotypic",
                                    n_processors=n_processors,
                                    genotypic_threshold_distance={"act_function":3, "weight":10, "bias":5, "body":phenotypic_threshold},
                                    genotypic_threshold_constant=0.25,
                                    threshold=0.6,
                                    max_gen=500)

            print(a)
            print(c)
            
            bPath = os.path.join(folder, "phenotypic_representatives.csv")
            with open(bPath, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["robot_id"])
                for robot_id in b:
                    writer.writerow([robot_id])

            dPath = os.path.join(folder, "phenotypic_speciation.csv")
            with open(dPath, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["species_id", "representative_id", "member_robot_ids"])  # Header
                for species_id, data in d.items():
                    members_str = ";".join(map(str, data["set"]))
                    writer.writerow([species_id, data["representative"], members_str])


if __name__ == "__main__":
    # build_all_genotypic_speciation_archives(25, 0.25, 0.6) #0.6 minimum fitness # 0.25 constant multiplying mean #25 cores to paralelize the mean calculus
    # build_all_phenotypic_speciation_archives(25,0.2)
    build_all_phenotypic_speciation_archives(n_processors=20, phenotypic_threshold=0.15)




                
         

            

            
            


