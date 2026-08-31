import pandas as pd
import numpy as np, json
import multiprocessing as mp
import re, ast
from Analysis.data import loaders, tools
# from sklearn.cluster import DBSCAN

TASKMAPS = {
    "walker":{"(0,0)":"Walker-v0","(1,0)":"Walker-v0","(2,0)":"Walker-v0","(3,0)":"Walker-v0","(4,0)":"Walker-v0","(5,0)":"Walker-v0","(6,0)":"Walker-v0","(7,0)":"Walker-v0","(8,0)":"Walker-v0","(9,0)":"Walker-v0","(10,0)":"Walker-v0","(11,0)":"Walker-v0","(12,0)":"Walker-v0","(13,0)":"Walker-v0","(14,0)":"Walker-v0","(15,0)":"Walker-v0","(16,0)":"Walker-v0","(17,0)":"Walker-v0","(18,0)":"Walker-v0","(19,0)":"Walker-v0","(0,1)":"Walker-v0","(1,1)":"Walker-v0","(2,1)":"Walker-v0","(3,1)":"Walker-v0","(4,1)":"Walker-v0","(5,1)":"Walker-v0","(6,1)":"Walker-v0","(7,1)":"Walker-v0","(8,1)":"Walker-v0","(9,1)":"Walker-v0","(10,1)":"Walker-v0","(11,1)":"Walker-v0","(12,1)":"Walker-v0","(13,1)":"Walker-v0","(14,1)":"Walker-v0","(15,1)":"Walker-v0","(16,1)":"Walker-v0","(17,1)":"Walker-v0","(18,1)":"Walker-v0","(19,1)":"Walker-v0","(0,2)":"Walker-v0","(1,2)":"Walker-v0","(2,2)":"Walker-v0","(3,2)":"Walker-v0","(4,2)":"Walker-v0","(5,2)":"Walker-v0","(6,2)":"Walker-v0","(7,2)":"Walker-v0","(8,2)":"Walker-v0","(9,2)":"Walker-v0","(10,2)":"Walker-v0","(11,2)":"Walker-v0","(12,2)":"Walker-v0","(13,2)":"Walker-v0","(14,2)":"Walker-v0","(15,2)":"Walker-v0","(16,2)":"Walker-v0","(17,2)":"Walker-v0","(18,2)":"Walker-v0","(19,2)":"Walker-v0","(0,3)":"Walker-v0","(1,3)":"Walker-v0","(2,3)":"Walker-v0","(3,3)":"Walker-v0","(4,3)":"Walker-v0","(5,3)":"Walker-v0","(6,3)":"Walker-v0","(7,3)":"Walker-v0","(8,3)":"Walker-v0","(9,3)":"Walker-v0","(10,3)":"Walker-v0","(11,3)":"Walker-v0","(12,3)":"Walker-v0","(13,3)":"Walker-v0","(14,3)":"Walker-v0","(15,3)":"Walker-v0","(16,3)":"Walker-v0","(17,3)":"Walker-v0","(18,3)":"Walker-v0","(19,3)":"Walker-v0","(0,4)":"Walker-v0","(1,4)":"Walker-v0","(2,4)":"Walker-v0","(3,4)":"Walker-v0","(4,4)":"Walker-v0","(5,4)":"Walker-v0","(6,4)":"Walker-v0","(7,4)":"Walker-v0","(8,4)":"Walker-v0","(9,4)":"Walker-v0","(10,4)":"Walker-v0","(11,4)":"Walker-v0","(12,4)":"Walker-v0","(13,4)":"Walker-v0","(14,4)":"Walker-v0","(15,4)":"Walker-v0","(16,4)":"Walker-v0","(17,4)":"Walker-v0","(18,4)":"Walker-v0","(19,4)":"Walker-v0"},
    "bridge":{"(0,0)":"BridgeWalker-v0","(1,0)":"BridgeWalker-v0","(2,0)":"BridgeWalker-v0","(3,0)":"BridgeWalker-v0","(4,0)":"BridgeWalker-v0","(5,0)":"BridgeWalker-v0","(6,0)":"BridgeWalker-v0","(7,0)":"BridgeWalker-v0","(8,0)":"BridgeWalker-v0","(9,0)":"BridgeWalker-v0","(10,0)":"BridgeWalker-v0","(11,0)":"BridgeWalker-v0","(12,0)":"BridgeWalker-v0","(13,0)":"BridgeWalker-v0","(14,0)":"BridgeWalker-v0","(15,0)":"BridgeWalker-v0","(16,0)":"BridgeWalker-v0","(17,0)":"BridgeWalker-v0","(18,0)":"BridgeWalker-v0","(19,0)":"BridgeWalker-v0","(0,1)":"BridgeWalker-v0","(1,1)":"BridgeWalker-v0","(2,1)":"BridgeWalker-v0","(3,1)":"BridgeWalker-v0","(4,1)":"BridgeWalker-v0","(5,1)":"BridgeWalker-v0","(6,1)":"BridgeWalker-v0","(7,1)":"BridgeWalker-v0","(8,1)":"BridgeWalker-v0","(9,1)":"BridgeWalker-v0","(10,1)":"BridgeWalker-v0","(11,1)":"BridgeWalker-v0","(12,1)":"BridgeWalker-v0","(13,1)":"BridgeWalker-v0","(14,1)":"BridgeWalker-v0","(15,1)":"BridgeWalker-v0","(16,1)":"BridgeWalker-v0","(17,1)":"BridgeWalker-v0","(18,1)":"BridgeWalker-v0","(19,1)":"BridgeWalker-v0","(0,2)":"BridgeWalker-v0","(1,2)":"BridgeWalker-v0","(2,2)":"BridgeWalker-v0","(3,2)":"BridgeWalker-v0","(4,2)":"BridgeWalker-v0","(5,2)":"BridgeWalker-v0","(6,2)":"BridgeWalker-v0","(7,2)":"BridgeWalker-v0","(8,2)":"BridgeWalker-v0","(9,2)":"BridgeWalker-v0","(10,2)":"BridgeWalker-v0","(11,2)":"BridgeWalker-v0","(12,2)":"BridgeWalker-v0","(13,2)":"BridgeWalker-v0","(14,2)":"BridgeWalker-v0","(15,2)":"BridgeWalker-v0","(16,2)":"BridgeWalker-v0","(17,2)":"BridgeWalker-v0","(18,2)":"BridgeWalker-v0","(19,2)":"BridgeWalker-v0","(0,3)":"BridgeWalker-v0","(1,3)":"BridgeWalker-v0","(2,3)":"BridgeWalker-v0","(3,3)":"BridgeWalker-v0","(4,3)":"BridgeWalker-v0","(5,3)":"BridgeWalker-v0","(6,3)":"BridgeWalker-v0","(7,3)":"BridgeWalker-v0","(8,3)":"BridgeWalker-v0","(9,3)":"BridgeWalker-v0","(10,3)":"BridgeWalker-v0","(11,3)":"BridgeWalker-v0","(12,3)":"BridgeWalker-v0","(13,3)":"BridgeWalker-v0","(14,3)":"BridgeWalker-v0","(15,3)":"BridgeWalker-v0","(16,3)":"BridgeWalker-v0","(17,3)":"BridgeWalker-v0","(18,3)":"BridgeWalker-v0","(19,3)":"BridgeWalker-v0","(0,4)":"BridgeWalker-v0","(1,4)":"BridgeWalker-v0","(2,4)":"BridgeWalker-v0","(3,4)":"BridgeWalker-v0","(4,4)":"BridgeWalker-v0","(5,4)":"BridgeWalker-v0","(6,4)":"BridgeWalker-v0","(7,4)":"BridgeWalker-v0","(8,4)":"BridgeWalker-v0","(9,4)":"BridgeWalker-v0","(10,4)":"BridgeWalker-v0","(11,4)":"BridgeWalker-v0","(12,4)":"BridgeWalker-v0","(13,4)":"BridgeWalker-v0","(14,4)":"BridgeWalker-v0","(15,4)":"BridgeWalker-v0","(16,4)":"BridgeWalker-v0","(17,4)":"BridgeWalker-v0","(18,4)":"BridgeWalker-v0","(19,4)":"BridgeWalker-v0"},
    "multi":{"(0,0)":"BridgeWalker-v0","(1,0)":"BridgeWalker-v0","(2,0)":"BridgeWalker-v0","(3,0)":"BridgeWalker-v0","(4,0)":"BridgeWalker-v0","(5,0)":"BridgeWalker-v0","(6,0)":"BridgeWalker-v0","(7,0)":"BridgeWalker-v0","(8,0)":"BridgeWalker-v0","(9,0)":"BridgeWalker-v0","(10,0)":"BridgeWalker-v0","(11,0)":"BridgeWalker-v0","(12,0)":"BridgeWalker-v0","(13,0)":"BridgeWalker-v0","(14,0)":"BridgeWalker-v0","(15,0)":"BridgeWalker-v0","(16,0)":"BridgeWalker-v0","(17,0)":"BridgeWalker-v0","(18,0)":"BridgeWalker-v0","(19,0)":"BridgeWalker-v0","(0,1)":"BridgeWalker-v0","(1,1)":"BridgeWalker-v0","(2,1)":"BridgeWalker-v0","(3,1)":"BridgeWalker-v0","(4,1)":"BridgeWalker-v0","(5,1)":"BridgeWalker-v0","(6,1)":"BridgeWalker-v0","(7,1)":"BridgeWalker-v0","(8,1)":"BridgeWalker-v0","(9,1)":"BridgeWalker-v0","(10,1)":"BridgeWalker-v0","(11,1)":"BridgeWalker-v0","(12,1)":"BridgeWalker-v0","(13,1)":"BridgeWalker-v0","(14,1)":"BridgeWalker-v0","(15,1)":"BridgeWalker-v0","(16,1)":"BridgeWalker-v0","(17,1)":"BridgeWalker-v0","(18,1)":"BridgeWalker-v0","(19,1)":"BridgeWalker-v0","(0,2)":"Walker-v0","(1,2)":"BridgeWalker-v0","(2,2)":"Walker-v0","(3,2)":"BridgeWalker-v0","(4,2)":"Walker-v0","(5,2)":"BridgeWalker-v0","(6,2)":"Walker-v0","(7,2)":"BridgeWalker-v0","(8,2)":"Walker-v0","(9,2)":"BridgeWalker-v0","(10,2)":"Walker-v0","(11,2)":"BridgeWalker-v0","(12,2)":"Walker-v0","(13,2)":"BridgeWalker-v0","(14,2)":"Walker-v0","(15,2)":"BridgeWalker-v0","(16,2)":"Walker-v0","(17,2)":"BridgeWalker-v0","(18,2)":"Walker-v0","(19,2)":"BridgeWalker-v0","(0,3)":"Walker-v0","(1,3)":"Walker-v0","(2,3)":"Walker-v0","(3,3)":"Walker-v0","(4,3)":"Walker-v0","(5,3)":"Walker-v0","(6,3)":"Walker-v0","(7,3)":"Walker-v0","(8,3)":"Walker-v0","(9,3)":"Walker-v0","(10,3)":"Walker-v0","(11,3)":"Walker-v0","(12,3)":"Walker-v0","(13,3)":"Walker-v0","(14,3)":"Walker-v0","(15,3)":"Walker-v0","(16,3)":"Walker-v0","(17,3)":"Walker-v0","(18,3)":"Walker-v0","(19,3)":"Walker-v0","(0,4)":"Walker-v0","(1,4)":"Walker-v0","(2,4)":"Walker-v0","(3,4)":"Walker-v0","(4,4)":"Walker-v0","(5,4)":"Walker-v0","(6,4)":"Walker-v0","(7,4)":"Walker-v0","(8,4)":"Walker-v0","(9,4)":"Walker-v0","(10,4)":"Walker-v0","(11,4)":"Walker-v0","(12,4)":"Walker-v0","(13,4)":"Walker-v0","(14,4)":"Walker-v0","(15,4)":"Walker-v0","(16,4)":"Walker-v0","(17,4)":"Walker-v0","(18,4)":"Walker-v0","(19,4)":"Walker-v0"}
}

################################################################
# builders.py
################################################################
# methods that read data from loaders and adjust to vizualizers
################################################################
FITNESS_COLUMN_PREFIX = "fit_world."

def build_directional_hamming_map(df: pd.DataFrame, rows:int, cols: int, toroid:bool=False):
    """Reads the results of a experiment and builds a matrix with directional hamming distances for each generation.

    Output:
    matrix -> a matrix for each generation containing directional hamming distances 
    generations -> a list with each generation in experiment data
    """
    #grabs all present generations
    generations = sorted(df["gen"].unique())
    n_gens = len(generations)
    #starts matrix with -1 to indicate stuff that was not found in dataframe
    matrix = np.full((n_gens, rows, cols), fill_value=-1, dtype=dict)

    #iterates throught generations and builds matrix
    for gen in generations:
        #get robots of this gen
        genBots = df[df["gen"]==gen]
        #build a dict to quick access (faster than filtering dataframe each line)
        shapeMap = {row["pos"]: row["shape"] for _, row in genBots.iterrows()}
        neighborDistMap = {row["pos"]: {} for _, row in genBots.iterrows()}

        for _, row in genBots.iterrows():
            #for each bot in the generation
            pos = row["pos"]
            matrix[gen, pos[1], pos[0]] = tools.get_directional_hamming_distances(pos, shapeMap, rows, cols, toroid)
         
    missing = np.sum(matrix == -1)
    if missing > 0: print("Missing values in matrix!")
    return matrix, generations

def build_fitness_map(df: pd.DataFrame, taskMap:dict, rows:int, cols: int):
    """
    Builds a matrix of (gens, rows, cols) where each cell has the fitness of the robot in its own task"""
    #grabs all present generations
    generations = sorted(df["gen"].unique())
    n_gens = len(generations)

    #starts matrix with -1 to indicate stuff that was not found in dataframe
    matrix = np.full((n_gens, rows, cols), fill_value=-1, dtype=float)
    uniqueTasks = set(taskMap.values())
    minmaxDict = {task: {"min": 7777777, "max": -7777777} for task in uniqueTasks}

    #gets the min and max value of fitness in each task
    for task in uniqueTasks:
        fitColumn = f"fit_{task}"
        all_fits = df[fitColumn].dropna()
        minmaxDict[task] = {
            "min": all_fits.min(),
            "max": all_fits.max()
        }

    #iterates throught generations and builds matrix
    for gen in generations:
        #get robots of this gen
        genBots = df[df["gen"]==gen]

        #write each bot of this gen
        for _, row in genBots.iterrows():
            x, y = row["pos"]
            pos = f"({x},{y})"
            taskName = taskMap[pos]
            fitValue = row[f"fit_{taskName}"]
            # normFit = (fitValue - minmaxDict[taskName]["min"]) / (minmaxDict[taskName]["max"] - minmaxDict[taskName]["min"])
            matrix[gen, y, x] = fitValue

    missing = np.sum(matrix == -1)
    if missing > 0: print("Missing values in matrix!")
    return matrix, generations, minmaxDict

def build_best_fitness_average_data(df: pd.DataFrame, taskMaps: dict) -> dict:
    """
    Reads the robots log dataframe and builds one dict entry per task.
    For each experiment, in each seed, gets the MAXIMUM fitness (best individual) of the population for each generation.
    Then, calculates the average of the best fitness across all seeds + std deviation per generation.

    Returns a dict with format:
        {
            "experiment-task": {
                "label": str,        # task name (short)
                "x": list[int],      # generation indices
                "y": list[float],    # mean of BEST fitness for each generation (across seeds)
                "std": list[float],  # std deviation of best fitness across seeds
            }
        }
    """
    generations = sorted(df["gen"].unique())
    seeds = sorted(df["seed"].unique())
    experiments = sorted(df["experiment"].unique())
    output = {}

    for experiment in experiments:
        expDf = df.loc[df["experiment"] == experiment].copy()
        uniqueTasks = list(set(taskMaps[experiment].values()))

        if experiment.startswith("walker"):
            label = "Walker Map"
        elif experiment.startswith("bridge"):
            label = "Bridge Map"
        elif experiment.startswith("mixed"):
            label = "Multi Map"

        # Prepare output structure
        for task in uniqueTasks:
            output[f"{experiment}-{task}"] = {
                "label": f"{label}-{task}",
                "x": [],
                "y": [],
                "std": []
            }

        seedFits = {task: {gen: [] for gen in generations} for task in uniqueTasks}

        for seed in seeds:
            seedDf = expDf.loc[expDf["seed"] == seed]

            for gen in generations:
                genBots = seedDf.loc[seedDf["gen"] == gen]
                genFits = {task: [] for task in uniqueTasks}

                # Get fitness values of each robot for the task of its position
                for _, row in genBots.iterrows():
                    x, y = row["pos"]
                    pos = f"({x},{y})"
                    taskName = taskMaps[experiment][pos]
                    fitValue = row[f"fit_{taskName}"]
                    genFits[taskName].append(fitValue)

                # Gets the MAXIMUM (best) fitness of the generation for each task
                for task in uniqueTasks:
                    if genFits[task]:
                        seedFits[task][gen].append(np.max(genFits[task]))

        # Aggregate across seeds
        for gen in generations:
            for task in uniqueTasks:
                values = seedFits[task][gen]
                output[f"{experiment}-{task}"]["x"].append(gen)
                output[f"{experiment}-{task}"]["y"].append(np.mean(values))
                output[f"{experiment}-{task}"]["std"].append(np.std(values))

    return output

def build_fitness_average_data(df: pd.DataFrame, taskMaps:dict) -> dict:
    """
    Reads the robots log dataframe and builds one dict entry per task.
    For each experiment, in each seed the average fitness of the population for each generation.
    Then, the average of each generation in all seeds + std deviation

    Returns a list of dicts with format:
        {
            "label": str,        # task name (short)
            "x": list[int],      # generation indices
            "y": list[float],    # mean fitness for each generation (across seed)
            "std": list[float],  # std deviation across seeds (or within itself if only seed)
        }
    """
    generations = sorted(df["gen"].unique())
    seeds = sorted(df["seed"].unique())
    experiments = sorted(df["experiment"].unique())
    output = {}
    #for each experiment
    for experiment in experiments:
        #   for each unique task in that experiment, a label.
        expDf = df.loc[df["experiment"] == experiment].copy()
        uniqueTasks = list(set(taskMaps[experiment].values()))

        if experiment.startswith("walker"):
            label = "Walker Map"
        elif experiment.startswith("bridge"):
            label = "Bridge Map"
        elif experiment.startswith("mixed"):
            label = "Multi Map"

        # prepare output structure
        for i,task in enumerate(uniqueTasks):
            output[f"{experiment}-{task}"] = {}
            output[f"{experiment}-{task}"]["label"] = f"{label}-{task}"
            output[f"{experiment}-{task}"]["x"] = []
            output[f"{experiment}-{task}"]["y"] = []
            output[f"{experiment}-{task}"]["std"] = []

        seedFits = {task: {gen: [] for gen in generations} for task in uniqueTasks}

        # for each seed of each experiment
        for seed in seeds:
            seedDf = expDf.loc[expDf["seed"] == seed]

            # for each generation in each seed of each experiment
            for gen in generations:
                genBots = seedDf.loc[seedDf["gen"]==gen].copy()
                genFits  = {task: [] for task in uniqueTasks}

                #get population fitValues of each robot for the task of its position
                for _, row in genBots.iterrows():
                    x, y = row["pos"]
                    pos = f"({x},{y})"
                    taskName = taskMaps[experiment][pos]
                    fitValue = row[f"fit_{taskName}"]
                    genFits[taskName].append(fitValue)

                #gets the average git of generation
                for task in uniqueTasks:
                    if genFits[task]:
                        seedFits[task][gen].append(np.mean(genFits[task]))

        # aggregate across seeds
        for gen in generations:
            for task in uniqueTasks:
                values = seedFits[task][gen]
                output[f"{experiment}-{task}"]["x"].append(gen)
                output[f"{experiment}-{task}"]["y"].append(np.mean(values))
                output[f"{experiment}-{task}"]["std"].append(np.std(values))

    return output   

def build_fit_specialization_data (df: pd.DataFrame, taskMap: dict) -> dict:
    """
    df: Corresponds to the df of ONE seed.
    Uses the final generation data (which must include cross-task evaluations) 
    to analyze specialist vs generalist robots.

    Returns:
    {
        "tasks": [str(unique-tasks-in-map)]
        "bots": [
                {
                    "pos":       (x, y),
                    "localTask": str,        # task assigned to this cell
                    "taskNameA":      float,      # normalized fitness on taskA
                    "taskNameB":      float,      # normalized fitness on taskB
                    "delta":     float,      # |fitA - fitB| specialization score
                },
                ...
            ]
    }
    """
    uniqueTasks = sorted(set(taskMap.values()))
    lastGen     = df["gen"].max()
    lastBots    = df[df["gen"] == lastGen]
    print(df.columns)

    bots = []
    for _, row in lastBots.iterrows():
        x, y = row["pos"]
        pos = f"({x},{y})"
        localTask = taskMap[pos]
        bot = {
            "pos": row["pos"],
            "localTask": localTask,
        }

        for taskName in uniqueTasks:
            bot[taskName] = row[f"fit_{taskName}"]
        
        bot["delta"] = abs(bot[uniqueTasks[0]] - bot[uniqueTasks[1]])
        bots.append(bot)
    return {"tasks": uniqueTasks, "bots":bots}    

def build_global_hamming_data (df: pd.DataFrame) -> dict:
    """
    Reads the robots log dataframe and builds a dataset of global average hamming distance per gen

    output:
    {
        experiment: {
            "avgPerSeed": {
                0: [...],
                1: [...],
                2: [...]
            },

            "mean": [...],
            "std": [...],
            "x": [...]
        }
    }
    """
    generations = sorted(df["gen"].unique())
    seeds = sorted(df["seed"].unique())
    experiments = sorted(df["experiment"].unique())

    output = {}
    for experiment in experiments:
        if experiment not in output: output[experiment] = {"avgPerSeed":{}, "mean":[], "std":[], "x":generations}
        for seed in seeds: 
            if seed not in output[experiment]["avgPerSeed"]: output[experiment]["avgPerSeed"][seed] = []
    
    for experiment in experiments:
        experimentBots = df[df["experiment"]==experiment]
        
        #build avg per generation in each seed of each experiment
        for seed in seeds:
            seedBots = experimentBots[experimentBots["seed"]==seed]
            for genIdx, gen in enumerate(generations):
                genBots = seedBots[seedBots["gen"]==gen]
                shapeMap = {tuple(row["pos"]): np.concatenate(row["shape"]).ravel() for _, row in genBots.iterrows()}

                #global hamming
                allPositions = list(shapeMap.keys())
                globalDistances = []
                for i in range(len(allPositions)):
                    for j in range(i + 1, len(allPositions)):
                        bot1 = shapeMap[allPositions[i]]
                        bot2 = shapeMap[allPositions[j]]
                        dist = tools.hamming_distance(bot1,bot2)    
                        globalDistances.append(dist)

                output[experiment]["avgPerSeed"][seed].append(np.mean(globalDistances))
        
        #build global avg of all seeds per generation
        for genIdx, gen in enumerate(generations):
            values = []

            for seed in seeds: #get each avg value for each gen and get the avg of seeds
                value = output[experiment]["avgPerSeed"][seed][genIdx]
                values.append(value)

            output[experiment]["mean"].append(np.mean(values))
            output[experiment]["std"].append(np.std(values))
    return output

def build_hamming_intra_inter_task(df: pd.DataFrame, taskMap:dict):
    """
    {
        experiment: {
            "avgPerSeed": {
                seed: {
                    "inter": [...],
                    "intra": {
                        "task1": [...],
                        "task2": [...]
                    }
                }
            },
            "x":[]
        }
    }"""

    experiments = sorted(df["experiment"].unique())
    experiments = [item for item in experiments if "baseline" not in item]
    generations = sorted(df["gen"].unique())
    seeds = sorted(df["seed"].unique())


    uniqueTasks = []
    for exp, values in taskMap.items():
        for position, task in values.items():
            if task not in uniqueTasks:
                uniqueTasks.append(task)

    output = {}
    for exp in experiments:
        if exp not in output: output[exp] = {"avgPerSeed":{}, "x":generations}
        for seed in seeds: 
            if seed not in output[exp]["avgPerSeed"]: 
                output[exp]["avgPerSeed"][seed] = {"inter":[], "intra":{}}
 
    for experiment in experiments:
        experimentBots = df[df["experiment"]==experiment]
        for seed in seeds:
            seedBots = experimentBots[experimentBots["seed"]==seed]
            for genIdx, gen in enumerate(generations):
                genBots = seedBots[seedBots["gen"]==gen]
                shapeMap = {tuple(row["pos"]): np.concatenate(row["shape"]).ravel() for _, row in genBots.iterrows()}
                taskGroups = {task: [] for task in uniqueTasks}

                #separate bots by task
                for _, row in genBots.iterrows():
                    x, y   = row["pos"]
                    posKey = f"({x},{y})"
                    taskGroups[taskMap[experiment][posKey]].append(shapeMap[(x,y)])

                #loop through tasks
                for task, bots in taskGroups.items():
                    distances = []
                    if task not in output[experiment]["avgPerSeed"][seed]["intra"]: 
                        output[experiment]["avgPerSeed"][seed]["intra"][task] = []
                    #intra-task 
                    for i in range(len(bots)):
                        for j in range(i+1, len(bots)): #compare inside same task, where i!=j, and only once (ij=ji)
                            bot1 = taskGroups[task][i]
                            bot2 = taskGroups[task][j]
                            dist = tools.hamming_distance(bot1,bot2)
                            distances.append(dist)
                    output[experiment]["avgPerSeed"][seed]["intra"][task].append(np.mean(distances))
                
                #inter-task
                tasks = list(taskGroups.keys())
                interDistances = []
                for i in range(len(tasks)):
                    for j in range(i+1, len(tasks)):
                        botsA = taskGroups[tasks[i]]
                        botsB = taskGroups[tasks[j]]
                        for bot1 in botsA:
                            for bot2 in botsB:
                                dist = tools.hamming_distance(bot1, bot2)
                                interDistances.append(dist)

                output[experiment]["avgPerSeed"][seed]["inter"].append(np.mean(interDistances))
                        
    return output

def build_hamming_inter_experiments(df: pd.DataFrame, taskMap:dict, task2becompared:str, exp2compare:tuple[str,str]):
    # assumes baseline experiment contains only task2becompared
    """
    output[seed] = {
        experiments[0]: {
            "mean": [],
            "std": []
        },
        experiments[1]: {
            "mean": [],
            "std": []
        },
        "interExp": {
            "mean": [],
            "std": []
        }
    }"""     
    generations = sorted(df["gen"].unique())
    seeds = sorted(df["seed"].unique())     
    #get all involved tasks
    uniqueTasks = []
    for exp, values in taskMap.items():
        for position, task in values.items():
            if task not in uniqueTasks:
                uniqueTasks.append(task)

    df = df[(df["experiment"]==exp2compare[0])|(df["experiment"]==exp2compare[1])]
    experiments = sorted(df["experiment"].unique())
    if len(experiments) != 2:
        raise ValueError(f"Expected 2 experiments, not {len(experiments)}: {experiments}")
    for expIdx, experiment in enumerate(experiments):
        if "baseline" in experiment:
            baselineIdx=expIdx
        else:
            otherIdx=expIdx

    output = {"x":generations}
    for seed in seeds:
        output[seed] = {experiments[0]:{"mean":[],"std":[]}, 
                        experiments[1]:{"mean":[],"std":[]}, 
                        "interExp":{"mean":[],"std":[]}}   

    baselineDf = df[df["experiment"]==experiments[baselineIdx]]
    otherDf = df[df["experiment"]==experiments[otherIdx]]

    for seed in seeds:
        baselineSeed = baselineDf[baselineDf["seed"]==seed]
        otherSeed = otherDf[otherDf["seed"]==seed]

        for gen in generations:
            baselineGen = baselineSeed[baselineSeed["gen"]==gen]
            otherGen = otherSeed[otherSeed["gen"]==gen]

            baselineShapeMap = {tuple(row["pos"]): np.concatenate(row["shape"]).ravel() for _, row in baselineGen.iterrows()}
            
            #get the bots from the specific task
            otherShapeMap = {}
            for _, row in otherGen.iterrows():
                x, y   = row["pos"]
                posKey = f"({x},{y})"
                posTask = taskMap[experiments[otherIdx]][posKey]
                posTaskName = posTask.split(".")[1]
                posTaskName = posTaskName.split("_")[0]
                if task2becompared.lower() == posTaskName.lower():
                    otherShapeMap[tuple(row["pos"])] = np.concatenate(row["shape"]).ravel()

            baselinePositions = list(baselineShapeMap.keys())
            otherPositions = list(otherShapeMap.keys())

            interExpDist = []
            intraBaselineDist = []
            intraOtherDist = []

            for i in range(len(baselinePositions)):
                #comparing robots of different experiments
                for j in range(len(otherPositions)):
                    baselineBot = baselineShapeMap[baselinePositions[i]]
                    otherBot = otherShapeMap[otherPositions[j]]
                    dist = tools.hamming_distance(baselineBot, otherBot)
                    interExpDist.append(dist)
                
                #comparing robots of baselines
                for k in range(i + 1, len(baselinePositions)):
                    baselineBot1 = baselineShapeMap[baselinePositions[i]]
                    baselineBot2 = baselineShapeMap[baselinePositions[k]]
                    dist = tools.hamming_distance(baselineBot1, baselineBot2)
                    intraBaselineDist.append(dist)
            
            output[seed][experiments[baselineIdx]]["mean"].append(np.mean(intraBaselineDist))
            output[seed][experiments[baselineIdx]]["std"].append(np.std(intraBaselineDist))

            output[seed]["interExp"]["mean"].append(np.mean(interExpDist))
            output[seed]["interExp"]["std"].append(np.std(interExpDist))


            for l in range(len(otherPositions)):
                for m in range(l + 1, len(otherPositions)):
                    otherBot1 = otherShapeMap[otherPositions[l]]
                    otherBot2 = otherShapeMap[otherPositions[m]]
                    dist = tools.hamming_distance(otherBot1, otherBot2)
                    intraOtherDist.append(dist)
            
            output[seed][experiments[otherIdx]]["mean"].append(np.mean(intraOtherDist))
            output[seed][experiments[otherIdx]]["std"].append(np.std(intraOtherDist))


    return output
            
def build_best_bots_per_task(df: pd.DataFrame, taskMaps: dict, botQtd: int = 5):
    """
    Gets the top N robots of each task, for each seed, in each experiment.
    Returns:
    {
        experiment: {
            task: {
                seed: [
                    {
                        "fitness": float,
                        "gen": int,
                        "pos": (x,y),
                        "shape": ndarray/list
                    },
                    ...
                ]
            }
        }
    }
    """

    experiments = sorted(df["experiment"].unique())
    seeds = sorted(df["seed"].unique())

    output = {}

    for experiment in experiments:
        expDf = df[df["experiment"] == experiment].copy()
        uniqueTasks = sorted(set(taskMaps[experiment].values()))

        # build localTask column
        expDf["localTask"] = expDf["pos"].apply(lambda pos: taskMaps[experiment][f"({pos[0]},{pos[1]})"])
        output[experiment] = {}

        for task in uniqueTasks:
            fitColumn = f"fit_{task}"
            output[experiment][task] = {}
            taskDf = expDf[expDf["localTask"] == task]

            for seed in seeds:
                seedDf = taskDf[taskDf["seed"] == seed]
                if seedDf.empty:
                    output[experiment][task][seed] = []
                    continue
                # get top N rows according to task fitness
                topBots = seedDf.nlargest(botQtd, fitColumn)
                output[experiment][task][seed] = []

                for _, row in topBots.iterrows():
                    output[experiment][task][seed].append({
                        "fitness": row[fitColumn],
                        "gen": row["gen"],
                        "pos": tuple(row["pos"]),
                        "shape": row["shape"]
                    })

    return output            

def build_genotype_clusters(df: pd.DataFrame, experiment: str, seed, eps: float = 0.2, min_samples: int = 4):
    seedDf = df[(df["experiment"] == experiment) & (df["seed"] == seed)].copy()
    X = np.stack(seedDf["shape"].apply(lambda s: s.flatten()).values)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="hamming", algorithm="ball_tree")
    seedDf["cluster"] = clustering.fit(X).labels_   
    return seedDf

def build_density_table():  
    
    multi_taskmap = TASKMAPS["multi"]
    
    df, _, _ = loaders.load_parquet_log("log/completeData.parquet") #loads archive with all executions
    df['seed'] = df['seed'].astype(int) #seed was string, has to be int
    df_unique = df.drop_duplicates(subset=['id', 'experiment', 'seed'], keep='first') #erases duplicates. bots with same id in the same exp and seed.
    
    species = pd.read_csv("log/genotypic_speciation_all.csv") #loads species archive
    species['seed'] = species['experiment'].apply(tools.get_seed) #gets seed from experiment name
    species['scenario_experiment'] = species['experiment'].str.split('_seed').str[0] #leave only whats before seed in exp name
    
    merged = species.merge(
        df_unique[['id', 'experiment', 'seed', 'pos']],
        left_on=['representative_id', 'seed', 'scenario_experiment'],
        right_on=['id', 'seed', 'experiment'],
        how='left'
    ) #merge both dataframes, so its possible to grab the task of each representative
    merged['pos'] = merged['pos'].apply(tools.parse_pos) #makes sure pos is a tuple
    
    #single task scenarios
    single_task = merged[merged['scenario_experiment'] != 'mixed-randomSelectAge50-20x5'] #do dont use multi map, this is to count map with single task
    single_task_counts = (
        single_task
        .groupby(['scenario_experiment', 'seed'])
        .size()
        .reset_index(name='n_species')
    ) #counts the number of species in the scenario_experiment and seed! (per execution) -> putz in n_species
    single_task_counts['population'] = 100 #adds population column with population size (constant)
    single_task_counts['task'] = single_task_counts['scenario_experiment'].map({ 
        'walker-randomSelectAge50-20x5': 'Walker-v0',
        'bridge-randomSelectAge50-20x5': 'BridgeWalker-v0',
    }) #creates task columns and changes ugly name to readable one
    
    # multi map scenarios
    mixed = merged[merged['scenario_experiment'] == 'mixed-randomSelectAge50-20x5'].copy() #grab just the multi map now
    mixed['task'] = mixed['pos'].apply(lambda p: tools.pos_to_task(p, multi_taskmap)) #creates a new column with the task of that position
    
    mixed_counts = (
        mixed
        .groupby(['seed', 'task'])
        .size()
        .reset_index(name='n_species')
    ) #counts the number of species in the scenario_experiment and seed! (per execution) -> putz in n_species
    mixed_counts['population'] = 50 #adds population column with population size (constant)
    mixed_counts['scenario_experiment'] = 'mixed-randomSelectAge50-20x5' #creates task columns and changes ugly name to readable one
    
    
    all_counts = pd.concat([single_task_counts, mixed_counts], ignore_index=True) #puts single tasks and multi together
    all_counts['density'] = all_counts['n_species'] / all_counts['population'] #calcs density and puts in new column
    
    print(all_counts)
    
    density_summary = (
        all_counts
        .groupby(['scenario_experiment', 'task'])['density']
        .agg(['mean', 'std'])
    ) #summarizes calculating the avg and std of the values
    
    print(density_summary)

def build_age_average_data(df: pd.DataFrame, taskMaps: dict) -> dict:
    """Reads the robots log dataframe and builds one dict entry per task.

    Calculates the average age (from parent2 < 0) per generation across seeds
    + std deviation. Keeps only the highest age for duplicated IDs within the
    same experiment and seed.

    Returns a dict with format:
        {
            "experiment-task": {
                "label": str,        # task name (formatted)
                "x": list[int],      # generation indices
                "y": list[float],    # mean age across seeds
                "std": list[float]   # std deviation across seeds
            }
        }
    """
    dfNeg = df[df["parent2"] < 0].copy()
    dfNeg["age"] = dfNeg["parent2"].abs()
    print(df["category"].value_counts())

    generations = sorted(dfNeg["gen"].unique())
    seeds = sorted(dfNeg["seed"].unique())
    experiments = sorted(dfNeg["experiment"].unique())

    output = {}

    # for each experiment
    for experiment in experiments:
        expDf = dfNeg.loc[dfNeg["experiment"] == experiment].copy()
        uniqueTasks = list(set(taskMaps[experiment].values()))

        if experiment.startswith("walker"):
            label = "Walker Map"
        elif experiment.startswith("bridge"):
            label = "Bridge Map"
        elif experiment.startswith("mixed"):
            label = "Multi Map"

        # prepare output structure
        for task in uniqueTasks:
            key = f"{experiment}-{task}"
            output[key] = {
                "label": f"{label}-{task}",
                "x": [],
                "y": [],
                "std": [],
            }

        seedAges = {task: {gen: [] for gen in generations} for task in uniqueTasks}

        # for each seed of each experiment
        for seed in seeds:
            seedDf = expDf.loc[expDf["seed"] == seed]

            # for each generation in each seed of each experiment
            for gen in generations:
                genBots = seedDf.loc[seedDf["gen"] == gen]
                genAges = {task: [] for task in uniqueTasks}

                # get age values of each robot for the task of its position
                for _, row in genBots.iterrows():
                    x, y = row["pos"]
                    pos = f"({x},{y})"
                    taskName = taskMaps[experiment][pos]
                    ageValue = row["age"]
                    genAges[taskName].append(ageValue)

                # gets the average age of generation for this seed
                for task in uniqueTasks:
                    if genAges[task]:
                        seedAges[task][gen].append(np.mean(genAges[task]))

        # aggregate across seeds
        for gen in generations:
            for task in uniqueTasks:
                values = seedAges[task][gen]
                key = f"{experiment}-{task}"
                output[key]["x"].append(gen)

                if values:
                    output[key]["y"].append(np.mean(values))
                    # std dev  0 if 1 seed
                    output[key]["std"].append(
                        np.std(values) if len(values) > 1 else 0.0
                    )
                else:
                    output[key]["y"].append(0.0)
                    output[key]["std"].append(0.0)
    return output

if __name__=="__main__":
    

    build_density_table()



        # Chamar a sua função de plotagem para a Task atual
            

    
    # matrix, gen = build_directional_hamming_map(experimentFolder="log/v1/quadrantv1_seed7_CGA_04302108", toroid=False)
    # df, _, _ = loaders.load_parquet_log("log/v1/completeData.parquet") 
    # with open("log/v1/taskMaps.json", "r") as f:
    #     taskmaps = json.load(f)
    # output = build_best_bots_per_task(df, taskmaps, 5)
    # a=2
    # output = build_hamming_inter_experiments(df=df, taskMap=taskmaps, task2becompared="walker", exp2compare=("quadrantv1","baseline-walkerv1"))
    # df2 = build_fitness_convergence_data(df, taskMap)
    # df, taskMap, (rows, cols) = loaders.load_log(logdir="log/v1/quadrantv1_seed7_CGA_04302108")
    # output = build_fit_specialization_data(df, taskMap)