import matplotlib, imageio, os, json, importlib, sys, re, ast
from Analysis.data import builders, loaders, tools
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from pathlib import Path
from pygifsicle import optimize
from robot.matteobot import evaluate_both_last

TASKMAPS = {
    "walker":{"(0,0)":"Walker-v0","(1,0)":"Walker-v0","(2,0)":"Walker-v0","(3,0)":"Walker-v0","(4,0)":"Walker-v0","(5,0)":"Walker-v0","(6,0)":"Walker-v0","(7,0)":"Walker-v0","(8,0)":"Walker-v0","(9,0)":"Walker-v0","(10,0)":"Walker-v0","(11,0)":"Walker-v0","(12,0)":"Walker-v0","(13,0)":"Walker-v0","(14,0)":"Walker-v0","(15,0)":"Walker-v0","(16,0)":"Walker-v0","(17,0)":"Walker-v0","(18,0)":"Walker-v0","(19,0)":"Walker-v0","(0,1)":"Walker-v0","(1,1)":"Walker-v0","(2,1)":"Walker-v0","(3,1)":"Walker-v0","(4,1)":"Walker-v0","(5,1)":"Walker-v0","(6,1)":"Walker-v0","(7,1)":"Walker-v0","(8,1)":"Walker-v0","(9,1)":"Walker-v0","(10,1)":"Walker-v0","(11,1)":"Walker-v0","(12,1)":"Walker-v0","(13,1)":"Walker-v0","(14,1)":"Walker-v0","(15,1)":"Walker-v0","(16,1)":"Walker-v0","(17,1)":"Walker-v0","(18,1)":"Walker-v0","(19,1)":"Walker-v0","(0,2)":"Walker-v0","(1,2)":"Walker-v0","(2,2)":"Walker-v0","(3,2)":"Walker-v0","(4,2)":"Walker-v0","(5,2)":"Walker-v0","(6,2)":"Walker-v0","(7,2)":"Walker-v0","(8,2)":"Walker-v0","(9,2)":"Walker-v0","(10,2)":"Walker-v0","(11,2)":"Walker-v0","(12,2)":"Walker-v0","(13,2)":"Walker-v0","(14,2)":"Walker-v0","(15,2)":"Walker-v0","(16,2)":"Walker-v0","(17,2)":"Walker-v0","(18,2)":"Walker-v0","(19,2)":"Walker-v0","(0,3)":"Walker-v0","(1,3)":"Walker-v0","(2,3)":"Walker-v0","(3,3)":"Walker-v0","(4,3)":"Walker-v0","(5,3)":"Walker-v0","(6,3)":"Walker-v0","(7,3)":"Walker-v0","(8,3)":"Walker-v0","(9,3)":"Walker-v0","(10,3)":"Walker-v0","(11,3)":"Walker-v0","(12,3)":"Walker-v0","(13,3)":"Walker-v0","(14,3)":"Walker-v0","(15,3)":"Walker-v0","(16,3)":"Walker-v0","(17,3)":"Walker-v0","(18,3)":"Walker-v0","(19,3)":"Walker-v0","(0,4)":"Walker-v0","(1,4)":"Walker-v0","(2,4)":"Walker-v0","(3,4)":"Walker-v0","(4,4)":"Walker-v0","(5,4)":"Walker-v0","(6,4)":"Walker-v0","(7,4)":"Walker-v0","(8,4)":"Walker-v0","(9,4)":"Walker-v0","(10,4)":"Walker-v0","(11,4)":"Walker-v0","(12,4)":"Walker-v0","(13,4)":"Walker-v0","(14,4)":"Walker-v0","(15,4)":"Walker-v0","(16,4)":"Walker-v0","(17,4)":"Walker-v0","(18,4)":"Walker-v0","(19,4)":"Walker-v0"},
    "bridge":{"(0,0)":"BridgeWalker-v0","(1,0)":"BridgeWalker-v0","(2,0)":"BridgeWalker-v0","(3,0)":"BridgeWalker-v0","(4,0)":"BridgeWalker-v0","(5,0)":"BridgeWalker-v0","(6,0)":"BridgeWalker-v0","(7,0)":"BridgeWalker-v0","(8,0)":"BridgeWalker-v0","(9,0)":"BridgeWalker-v0","(10,0)":"BridgeWalker-v0","(11,0)":"BridgeWalker-v0","(12,0)":"BridgeWalker-v0","(13,0)":"BridgeWalker-v0","(14,0)":"BridgeWalker-v0","(15,0)":"BridgeWalker-v0","(16,0)":"BridgeWalker-v0","(17,0)":"BridgeWalker-v0","(18,0)":"BridgeWalker-v0","(19,0)":"BridgeWalker-v0","(0,1)":"BridgeWalker-v0","(1,1)":"BridgeWalker-v0","(2,1)":"BridgeWalker-v0","(3,1)":"BridgeWalker-v0","(4,1)":"BridgeWalker-v0","(5,1)":"BridgeWalker-v0","(6,1)":"BridgeWalker-v0","(7,1)":"BridgeWalker-v0","(8,1)":"BridgeWalker-v0","(9,1)":"BridgeWalker-v0","(10,1)":"BridgeWalker-v0","(11,1)":"BridgeWalker-v0","(12,1)":"BridgeWalker-v0","(13,1)":"BridgeWalker-v0","(14,1)":"BridgeWalker-v0","(15,1)":"BridgeWalker-v0","(16,1)":"BridgeWalker-v0","(17,1)":"BridgeWalker-v0","(18,1)":"BridgeWalker-v0","(19,1)":"BridgeWalker-v0","(0,2)":"BridgeWalker-v0","(1,2)":"BridgeWalker-v0","(2,2)":"BridgeWalker-v0","(3,2)":"BridgeWalker-v0","(4,2)":"BridgeWalker-v0","(5,2)":"BridgeWalker-v0","(6,2)":"BridgeWalker-v0","(7,2)":"BridgeWalker-v0","(8,2)":"BridgeWalker-v0","(9,2)":"BridgeWalker-v0","(10,2)":"BridgeWalker-v0","(11,2)":"BridgeWalker-v0","(12,2)":"BridgeWalker-v0","(13,2)":"BridgeWalker-v0","(14,2)":"BridgeWalker-v0","(15,2)":"BridgeWalker-v0","(16,2)":"BridgeWalker-v0","(17,2)":"BridgeWalker-v0","(18,2)":"BridgeWalker-v0","(19,2)":"BridgeWalker-v0","(0,3)":"BridgeWalker-v0","(1,3)":"BridgeWalker-v0","(2,3)":"BridgeWalker-v0","(3,3)":"BridgeWalker-v0","(4,3)":"BridgeWalker-v0","(5,3)":"BridgeWalker-v0","(6,3)":"BridgeWalker-v0","(7,3)":"BridgeWalker-v0","(8,3)":"BridgeWalker-v0","(9,3)":"BridgeWalker-v0","(10,3)":"BridgeWalker-v0","(11,3)":"BridgeWalker-v0","(12,3)":"BridgeWalker-v0","(13,3)":"BridgeWalker-v0","(14,3)":"BridgeWalker-v0","(15,3)":"BridgeWalker-v0","(16,3)":"BridgeWalker-v0","(17,3)":"BridgeWalker-v0","(18,3)":"BridgeWalker-v0","(19,3)":"BridgeWalker-v0","(0,4)":"BridgeWalker-v0","(1,4)":"BridgeWalker-v0","(2,4)":"BridgeWalker-v0","(3,4)":"BridgeWalker-v0","(4,4)":"BridgeWalker-v0","(5,4)":"BridgeWalker-v0","(6,4)":"BridgeWalker-v0","(7,4)":"BridgeWalker-v0","(8,4)":"BridgeWalker-v0","(9,4)":"BridgeWalker-v0","(10,4)":"BridgeWalker-v0","(11,4)":"BridgeWalker-v0","(12,4)":"BridgeWalker-v0","(13,4)":"BridgeWalker-v0","(14,4)":"BridgeWalker-v0","(15,4)":"BridgeWalker-v0","(16,4)":"BridgeWalker-v0","(17,4)":"BridgeWalker-v0","(18,4)":"BridgeWalker-v0","(19,4)":"BridgeWalker-v0"},
    "multi":{"(0,0)":"BridgeWalker-v0","(1,0)":"BridgeWalker-v0","(2,0)":"BridgeWalker-v0","(3,0)":"BridgeWalker-v0","(4,0)":"BridgeWalker-v0","(5,0)":"BridgeWalker-v0","(6,0)":"BridgeWalker-v0","(7,0)":"BridgeWalker-v0","(8,0)":"BridgeWalker-v0","(9,0)":"BridgeWalker-v0","(10,0)":"BridgeWalker-v0","(11,0)":"BridgeWalker-v0","(12,0)":"BridgeWalker-v0","(13,0)":"BridgeWalker-v0","(14,0)":"BridgeWalker-v0","(15,0)":"BridgeWalker-v0","(16,0)":"BridgeWalker-v0","(17,0)":"BridgeWalker-v0","(18,0)":"BridgeWalker-v0","(19,0)":"BridgeWalker-v0","(0,1)":"BridgeWalker-v0","(1,1)":"BridgeWalker-v0","(2,1)":"BridgeWalker-v0","(3,1)":"BridgeWalker-v0","(4,1)":"BridgeWalker-v0","(5,1)":"BridgeWalker-v0","(6,1)":"BridgeWalker-v0","(7,1)":"BridgeWalker-v0","(8,1)":"BridgeWalker-v0","(9,1)":"BridgeWalker-v0","(10,1)":"BridgeWalker-v0","(11,1)":"BridgeWalker-v0","(12,1)":"BridgeWalker-v0","(13,1)":"BridgeWalker-v0","(14,1)":"BridgeWalker-v0","(15,1)":"BridgeWalker-v0","(16,1)":"BridgeWalker-v0","(17,1)":"BridgeWalker-v0","(18,1)":"BridgeWalker-v0","(19,1)":"BridgeWalker-v0","(0,2)":"Walker-v0","(1,2)":"BridgeWalker-v0","(2,2)":"Walker-v0","(3,2)":"BridgeWalker-v0","(4,2)":"Walker-v0","(5,2)":"BridgeWalker-v0","(6,2)":"Walker-v0","(7,2)":"BridgeWalker-v0","(8,2)":"Walker-v0","(9,2)":"BridgeWalker-v0","(10,2)":"Walker-v0","(11,2)":"BridgeWalker-v0","(12,2)":"Walker-v0","(13,2)":"BridgeWalker-v0","(14,2)":"Walker-v0","(15,2)":"BridgeWalker-v0","(16,2)":"Walker-v0","(17,2)":"BridgeWalker-v0","(18,2)":"Walker-v0","(19,2)":"BridgeWalker-v0","(0,3)":"Walker-v0","(1,3)":"Walker-v0","(2,3)":"Walker-v0","(3,3)":"Walker-v0","(4,3)":"Walker-v0","(5,3)":"Walker-v0","(6,3)":"Walker-v0","(7,3)":"Walker-v0","(8,3)":"Walker-v0","(9,3)":"Walker-v0","(10,3)":"Walker-v0","(11,3)":"Walker-v0","(12,3)":"Walker-v0","(13,3)":"Walker-v0","(14,3)":"Walker-v0","(15,3)":"Walker-v0","(16,3)":"Walker-v0","(17,3)":"Walker-v0","(18,3)":"Walker-v0","(19,3)":"Walker-v0","(0,4)":"Walker-v0","(1,4)":"Walker-v0","(2,4)":"Walker-v0","(3,4)":"Walker-v0","(4,4)":"Walker-v0","(5,4)":"Walker-v0","(6,4)":"Walker-v0","(7,4)":"Walker-v0","(8,4)":"Walker-v0","(9,4)":"Walker-v0","(10,4)":"Walker-v0","(11,4)":"Walker-v0","(12,4)":"Walker-v0","(13,4)":"Walker-v0","(14,4)":"Walker-v0","(15,4)":"Walker-v0","(16,4)":"Walker-v0","(17,4)":"Walker-v0","(18,4)":"Walker-v0","(19,4)":"Walker-v0"}
}


HATCHING_PATTERNS = {
    0: '',     
    1: '/////',      
    2: '|||',    
    3: '***',    
    4: '---',
}

VOXEL_COLORS = {
    0: '#FFFFFF',  # vazio
    1: "#000000",  # rígido
    2: "#9a9a9a",  # soft
    3: '#ED832F',  # atuador horizontal
    4: "#529BD3",  # atuador vertical
}

def print_matteo_bot(pickleFile:str, pos:tuple[int,int]):
    from robot.matteobot import render
    picklePath = Path(pickleFile)
    paramsPath = f"{picklePath.parent.parent}{os.sep}parameters.json"
    imgRootPath = f"{picklePath.parent.parent}{os.sep}BotImages"
    os.makedirs(imgRootPath, exist_ok=True)
    with open(paramsPath, "r") as f:
        params = json.load(f)
    params["botparams"]["env_name"] = params["world_types"]
    params["botparams"]["sim_step"] = params["sim_step"]

    type_env = params["grid_worlds"][pos[1]][pos[0]]
    render(filename= pickleFile,
           out_dir= imgRootPath,
           pos= pos,
           type_env= type_env,
           params= params["botparams"])

def print_bot(logdir:str, gen:int, pos:tuple[int,int]):
    """Saves a GIF of a bot (generation+position in grid) from an experiment (logdir)
    """
    df, _, _ = loaders.load_log(logdir)
    outputPath = os.path.join(logdir, "printedBots")
    os.makedirs(outputPath, exist_ok=True)

    #load parameters
    with open(os.path.join(logdir, 'parameters.json'), 'r') as f:
        params = json.load(f)  
    
    botType = params["robot_type"]
    worldTypes = params["world_types"]
    gridWorlds = params["grid_worlds"]
    simSteps = params["sim_step"]

    #get bot from df
    mask = (df["gen"]==gen) & (df["pos"]==pos)
    if not mask.any():
        print("Didn't find that bot!")
        return
    
    botRow = df[mask].iloc[0]
    botShape = np.array(botRow["shape"])
    
    for worldType in worldTypes:
        #load modules
        robotModule = importlib.import_module(f"robot.{botType}")
        # worldIndex = gridWorlds[pos[1]][pos[0]]
        # worldType = worldTypes[worldIndex]
        worldModule = importlib.import_module(f"world.{worldType}")

        #make bot and world
        bot = robotModule.SinRobot()
        bot.shape = botShape
        world = worldModule.get_world()

        #simulate and render
        world.set_robot(bot)
        world.reset()
        viewer = world.get_viewer()
        frames = []
        for step in range(simSteps):
            world.step()
            frames.append(viewer.render(mode="img"))

        outputfile = os.path.join(outputPath, 
                                f"{str(worldType)}_gen{gen}_pos{pos[0]}-{pos[1]}.gif")
        imageio.mimsave(outputfile, frames, duration=20)
        optimize(outputfile)
        
        score = world.get_score()
        print(f"Score: {score}")
        print(f"GIF saved in: {outputfile}")

def render_task_map(taskMatrix: np.ndarray, taskNames: list, figSize: tuple = (15,15), title: str = None) -> np.ndarray:
    """Renders an image of task assignment across the grid, with legend."""
    plt.close("all")
    rows, cols = taskMatrix.shape
    fig, ax = plt.subplots(figsize=figSize)

    # draw grid cells, hatched by task
    for y in range(rows):
        for x in range(cols):
            taskIndex = taskMatrix[y, x]
            pattern = HATCHING_PATTERNS.get(taskIndex, '')
            rect = patches.Rectangle(
                (x - 0.5, y - 0.5),
                0.99, 0.99,
                linewidth=1,
                edgecolor='black',
                facecolor='white',
                hatch=pattern,
            )
            ax.add_patch(rect)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.axis("off")

    legendElements = [
        patches.Patch(hatch=HATCHING_PATTERNS.get(i, ''),
                       facecolor="white",
                       edgecolor="black", linewidth=1,
                       label=taskNames[i].split(".")[-1])
        for i in range(len(taskNames))
    ]
    ax.legend(handles=legendElements, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=20)

    if title:
        ax.set_title(title, fontsize=20)

    fig.subplots_adjust(
    left=0.01,
    right=0.70,
    top=0.85,
    bottom=0.05
    )
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())
    frame = frame[:, :, :3] 
    plt.close(fig)
    return frame

def render_hamming_direction_map(
    fitnessMatrix: np.ndarray, #output of build_fitness_map
    directionalMatrix: np.ndarray, #output of build_directional_hamming_map
    taskMatrix: np.ndarray,   #matrix that indicates which task is in which cell
    taskNames: list,          #task names in the correct order - indexes here indicate tasks in taskMatrix
    gen: int,                 #generation of this map
    taskColors: list[str],    #colors of each task in taskNames (for rectangle) 
    legendText: str,          #text that appears beside legends
    minMaxDict:dict,          #dictionary with minMax of all tasks
    figSize: tuple = (10,10) ) -> np.ndarray:
    """Renders a frame of the hamming direction map function"""

    plt.close("all")
    rows, cols = directionalMatrix.shape
    fig, ax = plt.subplots(figsize = figSize)

    # get global max
    globalMinMax = {
    'min': min(d['min'] for d in minMaxDict.values()),
    'max': max(d['max'] for d in minMaxDict.values())
    }

    #set heatmap with max and min values
    im = ax.imshow(
        fitnessMatrix,
        cmap='bwr',
        vmin=0, #globalMinMax['min'],
        vmax=globalMinMax['max'],
        aspect="equal"
    )

    #set limits
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect('equal')

    #draw hamming lines
    for y in range(rows):
        for x in range(cols):
            neighDict = directionalMatrix[y, x]
            # if neighDict is None or isinstance(neighDict, float): continue
            #to each cell neighbor
            for (neighX, neighY), hamming in neighDict.items():
                # if hamming is None or hamming < 0: continue
                if hamming < 0.15:
                    alpha = 1
                    linewidth = 5
                elif hamming < 0.5:
                    alpha = 0.4
                    linewidth = 3
                else:
                    alpha = 0
                    linewidth = 0
                                
                x_end = x + (neighX - x) * 0.4
                y_end = y + (neighY - y) * 0.4

                ax.plot([x, x_end], [y, y_end],
                        color='black', alpha=alpha, linewidth=linewidth,
                        solid_capstyle='round')
    
    #draw task borders
    for y in range(rows):
        for x in range(cols):
            taskIndex = taskMatrix[y, x]
            pattern = HATCHING_PATTERNS.get(taskIndex, '')
            rect = patches.Rectangle(
            (x - 0.5, y - 0.5), 
            0.99, 0.99,                  
            linewidth=1,
            edgecolor='black',
            facecolor='none',
            hatch=pattern,
            alpha=0.4)
            ax.add_patch(rect)

    #color bar and title
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(legendText, fontsize=10)

    #task subtitles
    legendElements = [
        patches.Patch(hatch=HATCHING_PATTERNS.get(i, ''),
        facecolor="white",
        edgecolor="black", linewidth=1,
        label=taskNames[i].split(".")[-1])
        for i in range(len(taskNames))]
    ax.legend(handles=legendElements, loc="upper left",
              bbox_to_anchor=(1.25, 1.0), fontsize=9)

    #write minMax
    if minMaxDict is not None:
        yCoord = -0.10
        for taskName in minMaxDict.keys():
            taskMax = minMaxDict[taskName]["max"]
            text = f"{taskName} Max: {taskMax:.2f}"

            plt.text(0.5, yCoord, text, 
                     transform=plt.gca().transAxes, 
                     fontsize=9, 
                     verticalalignment='top')
            yCoord = yCoord - 0.1

    #render
    ax.set_title(f"Generation {gen}", fontsize=13)
    ax.axis("off")
    
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())
    frame = frame[:, :, :3]  # RGB apenas
    plt.close(fig)
    return frame

def print_task_map(logdir: str, filename: str = "task_map.png", figSize: tuple = (8, 4), title: str = None):
    """Loads a task map from an experiment's log and saves a static PNG showing
    task assignment across the grid, with legend.
    """
    _, taskMap, (rows, cols) = loaders.load_log(logdir)
    taskMatrix, taskNames = tools.build_task_overlay(taskMap, rows, cols)

    frame = render_task_map(taskMatrix=taskMatrix, taskNames=taskNames, figSize=figSize, title=title)

    outputPath = os.path.join(Path(logdir).parent, "Graphs")
    os.makedirs(outputPath, exist_ok=True)
    outputFile = os.path.join(outputPath, filename)

    plt.imsave(outputFile, frame)
    print(f"Task map saved in: {outputFile}")

def print_directional_hammming_map_gif(logdir:str, taskColors:str, frameInterval:int=5, frameDuration:float=300, toroidal:bool=False):
    """
    Generates gif of all generation's heatmap considering the hammming distance of a cell to its neighbors, indicating to which cell theyre the most similar to
    Returns gif to the folder given by logdir."""
    df, taskMap, (rows, cols) = loaders.load_log(logdir)
    dirHammMatrix, dirHammGenerations = builders.build_directional_hamming_map(df, rows, cols, toroidal)
    fitnessMatrix, generations, minmaxDict = builders.build_fitness_map(df, taskMap, rows, cols)
    overlayMatrix, taskNames = tools.build_task_overlay(taskMap,rows,cols)
    plt.close("all")
    frames = []

    #starts to build gif
    print(f"Working on {logdir}...")
    for g_idx, gen in enumerate(dirHammGenerations):
        isLastGen = (g_idx == len(dirHammGenerations) - 1)
        if (gen % frameInterval == 0) or isLastGen:
            frame = render_hamming_direction_map(
                fitnessMatrix=fitnessMatrix[g_idx], minMaxDict=minmaxDict,
                directionalMatrix=dirHammMatrix[g_idx], 
                taskMatrix=overlayMatrix, taskNames=taskNames,
                gen=gen, taskColors=taskColors, legendText="Distance traveled to the right.",
                figSize=(8,8))
            frames.append(frame)
            # if g_idx % 10 == 0:
            #     print(f"Frame {g_idx}/{len(hammGenerations)} gerado...")

    # Salva o GIF
    output_path = os.path.join(logdir, "directionalHammingDistance_fromNeighbors.mp4")
    imageio.mimsave(output_path, frames, fps=frameDuration)#duration=frameDuration)  # frameDuration ms por frame
    print(f"GIF salved in: {output_path}")

def _print_line_graph(data:dict,outputPath:str, title:str="Title", xLabel:str="x", yLabel:str="y",
                                     figSize:tuple=(15,15), colors:list[str]=["red","blue","purple","green","pink","gray","black"]):
    
    plt.close("all")
    fig, ax = plt.subplots(figsize=figSize)

    for idx, (key, content) in enumerate(data.items()):
        x      = content["x"]
        y      = content["y"]
        std    = content["std"]
        label  = content.get("label", f"missingLabel{idx}")

        ax.plot(x, y, label=label, linewidth=2)

        yArr   = np.array(y)
        stdArr = np.array(std)
        ax.fill_between(x, yArr - stdArr, yArr + stdArr,
                        alpha=0.2, linewidth=0)

        ax.set_title(title, fontsize=23)
        ax.set_xlabel(xLabel, fontsize=25)
        ax.set_ylabel(yLabel, fontsize=25)
        ax.legend(fontsize=23)
        ax.tick_params(axis='both', which='major', labelsize=23)
        ax.grid(True, linestyle="--", alpha=1)
        ax.set_xlim(0, max(content["x"][-1] for content in data.values()))
        
    fig.tight_layout()
    fig.savefig(f"{outputPath}/{title}.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {outputPath}/{title}.png")

def print_population_average_fitness(logdir:str, title:str="Title", xLabel:str="x", yLabel:str="y",
                                     figSize:tuple=(10,10), colors:list[str]=["red","blue","purple","green","pink","gray","black"]):
    """
    Reads a parquet file.
    Prints a graph of the avg fit of population in each generation across seeds.
    """
    taskMap = {}
    parentDir = os.path.dirname(logdir)
    taskMapJsonPath = os.path.join(parentDir,"taskMaps.json")
    with open(taskMapJsonPath, "r") as f:
        taskMap = json.load(f)
    df, _, _ = loaders.load_parquet_log(logdir) 
    outputPath = os.path.join(Path(logdir).parent, "Graphs")
    os.makedirs(outputPath, exist_ok=True)
    data = builders.build_fitness_average_data(df, taskMap)
    _print_line_graph(data, outputPath, title, xLabel, yLabel, figSize, colors)

def print_population_best_fitness(logdir:str, title:str="Title", xLabel:str="x", yLabel:str="y",
                                figSize:tuple=(10,10), colors:list[str]=["red","blue","purple","green","pink","gray","black"]):
    """
    Reads a parquet file.
    Prints a graph of the avg of best fit of population in each generation across seeds.
    """
    taskMap = {}
    parentDir = os.path.dirname(logdir)
    taskMapJsonPath = os.path.join(parentDir,"taskMaps.json")
    with open(taskMapJsonPath, "r") as f:
        taskMap = json.load(f)
    df, _, _ = loaders.load_parquet_log(logdir) 
    outputPath = os.path.join(Path(logdir).parent, "Graphs")
    os.makedirs(outputPath, exist_ok=True)
    data = builders.build_best_fitness_average_data(df, taskMap)
    _print_line_graph(data, outputPath, title, xLabel, yLabel, figSize, colors)

def _print_scatter_graph(data: dict, logdir: str,
                        title:str="Title", 
                        xLabel:str="X", 
                        yLabel:str="Y", 
                        figsize:tuple=(8,8),
                        colors:list[str]=["red","blue","purple","pink"]):
    
    outputPath = os.path.join(logdir, "Graphs")
    os.makedirs(outputPath, exist_ok=True)
    #save data
    savedDictPath = os.path.join(outputPath, "scatterData.jsonl")
    with open(savedDictPath, "w") as file:
        for bot in data["bots"]:
            record = {**bot, "pos": list(bot["pos"])}  # tuple → list for JSON
            json.dump(record, file)
            file.write("\n")

    minMax = {}
    for task in data["tasks"]:
        if task not in minMax: minMax[task] = {"min":0, "max":-1.0}      
        minMax[task]["max"] = max(bot[task] for bot in data["bots"])

    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize)

    xTask = data["tasks"][0]
    yTask = data["tasks"][1]

    for i, task in enumerate(data["tasks"]):
        botsFromTask = [bot for bot in data["bots"] if bot["localTask"]==task]
        ax.scatter(
            [bot[xTask] for bot in botsFromTask],
            [bot[yTask] for bot in botsFromTask],
            color = colors[i],
            label = task.split(".")[-1],
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
            s=80,
            zorder=3,
        )
    
    # Diagonal: perfect generalist sits here
    ax.plot([0, minMax[xTask]["max"]], [0, minMax[yTask]["max"]], linestyle="--", color="gray",
            linewidth=1, label="Same fitness line", zorder=2)

    ax.set_xlabel(f"fitness — {data['tasks'][0]}", fontsize=11)
    ax.set_ylabel(f"fitness — {data['tasks'][1]}", fontsize=11)
    ax.set_title("Robots' fitness for both tasks (last generation)", fontsize=13)
    ax.set_xlim(0, minMax[xTask]["max"])
    ax.set_ylim(0, minMax[yTask]["max"])
    # ax.set_xticks(np.arange(0, 1.0, 0.1))
    # ax.set_yticks(np.arange(0, 1.0, 0.1))
    # ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=1)

    fig.tight_layout()
    fig.savefig(os.path.join(outputPath, "specialization_scatter.png"), dpi=150)
    plt.close(fig)
    print("Scatter saved.")

def print_last_gen_specialization_graph(logdir:str, title:str="Title", xLabel:str="x", yLabel:str="y",
                                     figSize:tuple=(10,10), colors:list[str]=["red","blue","purple","green","pink","gray","black"]):
    """
    Reads data from ONE experiments and plots a graph where X axis is fitness in one task and Y axis
    is fitness in another task. Also plots a diagonal line. 
    """

    df, taskMap, _ = loaders.load_log(logdir)
    inputs = builders.build_fit_specialization_data(df=df, taskMap=taskMap)
    _print_scatter_graph(data=inputs, logdir=logdir, title="Specialization x Generalization")

def print_global_hamming_distance_per_seed(logdir:str, title:str="Title", xLabel:str="Generation", yLabel:str="Hamming distance",
                                     figSize:tuple=(10,10), colors:list[str]=["red","blue","purple","green","pink","gray","black"]):
    """Prints a graph of global (pair-wise) average hamming distance"""

    df, _, _ = loaders.load_parquet_log(logdir) 
    outputPath = os.path.join(Path(logdir).parent, "Graphs")
    os.makedirs(outputPath, exist_ok=True)
    data = builders.build_global_hamming_data(df=df)

    for experiment, expData in data.items():
        plotData = {}
        #one line for each seed of the same exp
        for seed, values in expData["avgPerSeed"].items():
            plotData[f"seed{seed}"] = {
                "x": expData["x"],
                "y": values,
                "std": [0] * len(values),
                "label": f"seed{seed}"
            }
        _print_line_graph(plotData, outputPath, f"{experiment}-hammingDistance", xLabel, yLabel, figSize, colors)

def print_hammig_inter_intra_task(logdir:str, title:str="Title", xLabel:str="Generation", yLabel:str="Hamming distance",
                                     figSize:tuple=(10,10), colors:list[str]=["red","blue","purple","green","pink","gray","black"]):
    taskMap = {}
    parentDir = os.path.dirname(logdir)
    taskMapJsonPath = os.path.join(parentDir,"taskMaps.json")
    with open(taskMapJsonPath, "r") as f:
        taskMap = json.load(f)
    df, _, _ = loaders.load_parquet_log(logdir) 
    outputPath = os.path.join(Path(logdir).parent, "Graphs")
    os.makedirs(outputPath, exist_ok=True)
    data = builders.build_hamming_intra_inter_task(df, taskMap)

    for experiment, expData in data.items():
        for seed, compType in expData["avgPerSeed"].items():
            plotData = {}
            interValues = compType["inter"]

            plotData[f"inter-seed{seed}"] = {
                    "x": expData["x"],
                    "y": interValues,
                    "std": [0] * len(interValues),
                    "label": f"inter-seed{seed}"
                }

            for task, values in compType["intra"].items():
                plotData[f"intra-{task}-seed{seed}"] = {
                    "x": expData["x"],
                    "y": values,
                    "std": [0] * len(values),
                    "label": f"intra-{task}-seed{seed}"
                }
            _print_line_graph(plotData, outputPath, f"exp{experiment}-seed{seed}-InterIntraHammingDistance", xLabel, yLabel, figSize, colors)

def print_hamming_inter_exp(logdir:str, task2becompared:str, exp2compare:tuple[str,str], title:str="Title",
        xLabel:str="Generation", yLabel:str="Hamming distance", figSize:tuple=(10,10),
        colors:list[str]=["red","blue","purple","green","pink","gray","black"]):

    parentDir = os.path.dirname(logdir)
    taskMapJsonPath = os.path.join(parentDir, "taskMaps.json")
    with open(taskMapJsonPath, "r") as f:
        taskMap = json.load(f)
    df, _, _ = loaders.load_parquet_log(logdir)
    outputPath = os.path.join(Path(logdir).parent, "Graphs")
    os.makedirs(outputPath, exist_ok=True)

    data = builders.build_hamming_inter_experiments(df=df,taskMap=taskMap,task2becompared=task2becompared,exp2compare=exp2compare)

    generations = data["x"]

    for seed, seedData in data.items():
        if seed == "x":  continue
        plotData = {}

        for seriesName, values in seedData.items():
            plotData[seriesName] = {
                "x": generations,
                "y": values["mean"],
                "std": values["std"],
                "label": seriesName}

        _print_line_graph(
            plotData,
            outputPath,
            f"seed{seed}-InterExperimentHamming-{task2becompared}",
            xLabel,
            yLabel,
            figSize,
            colors)

def print_best_bots(logdir:str, botQtd:int=5):
    df, taskMap, _ = loaders.load_log(logdir)
    df["localTask"] = df["pos"].apply(lambda pos: taskMap[f"({pos[0]},{pos[1]})"])

    for task in sorted(set(taskMap.values())):
        fitColumn = f"fit_{task}"
        topBots = (df[df["localTask"] == task].nlargest(botQtd, fitColumn))

        print(f"\nTask: {task}")

        for rank, (_, bot) in enumerate(topBots.iterrows(), start=1):
            print(f"#{rank} "
                f"fit={bot[fitColumn]:.3f} "
                f"gen={bot['gen']} "
                f"pos={bot['pos']}"
            )

            print_bot(
                logdir=logdir,
                gen=bot["gen"],
                pos=tuple(bot["pos"])
            )

def print_matteo_specialization_graph(logFolder:str):
    results = evaluate_both_last(logFolder, gen=500, n_steps=500, tasks=["Walker-v0", "BridgeWalker-v0"])
    print(results)
    log_folder = logFolder
    json_path = os.path.join(log_folder, "last_gen.json")

    with open(json_path, "r") as file : 
        robots = json.load(file)

    task_x = "Walker-v0" 
    task_y = "BridgeWalker-v0"  

    blue_x, blue_y = [], [] # blue is walker
    red_x, red_y = [], [] # red is bridge

    for result in robots.values():
        fitness_on_x = result[task_x]
        fitness_on_y = result[task_y]

        if result["original"] == task_x:
            blue_x.append(fitness_on_x)
            blue_y.append(fitness_on_y)
        else:
            red_x.append(fitness_on_x)
            red_y.append(fitness_on_y)

    all_fitness = blue_x + blue_y + red_x + red_y
    lim = 1.05 * max(all_fitness)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1, color="grey", zorder=0)
    ax.scatter(blue_x, blue_y, color="blue", s=28, alpha=0.6,
            label=f"tache d'origine : {task_x} ({len(blue_x)} robots)")
    ax.scatter(red_x, red_y, color="red", s=28, alpha=0.6,
            label=f"tache d'origine : {task_y} ({len(red_x)} robots)")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")

    ax.set_xlabel(f"fitness sur {task_x}")
    ax.set_ylabel(f"fitness sur {task_y}")
    ax.set_title("Derniere generation : fitness sur les deux taches")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend()

    plt.tight_layout()
    fig.savefig(f"{logFolder}{os.sep}specializationGraph.png")   

def print_genotypic_counts(logdir:str):
    outputPath = os.path.join(Path(logdir), "Graphs")
    os.makedirs(outputPath, exist_ok=True)

    speciationCsv = f"{logdir}{os.sep}genotypic_speciation_all.csv"
    df = pd.read_csv(speciationCsv)
    df['scenario'] = df['experiment'].str.split('-').str[0]
    speciesPerSeef = df.groupby(['scenario', 'experiment']).size()
    summary = speciesPerSeef.groupby('scenario').agg(['mean', 'std'])

    order = ['walker', 'bridge', 'mixed']
    labels = ['Walker Map', 'Bridge Map', 'Multi Map']
    colors = ['#d62728', '#1f77b4', '#ff7f0e']
    
    means = summary.loc[order, 'mean']
    stds = summary.loc[order, 'std']

    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors,
                alpha=0.85, edgecolor='black', linewidth=0.6)

    ax.set_ylabel('Number of species (Genotype)')
    ax.set_title('Avg. Number of Species per Scenario (Genotype)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 2, f'{m:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(f"{outputPath}/Avg. Number of Genotyphic Species per Scenario.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {outputPath}/Avg. Number of Genotyphic Species per Scenario.png")

def print_phenotypic_counts(logdir:str):
    outputPath = os.path.join(Path(logdir), "Graphs")
    os.makedirs(outputPath, exist_ok=True)

    speciationCsv = f"{logdir}{os.sep}phenotypic_speciation_all.csv"
    df = pd.read_csv(speciationCsv)
    df['scenario'] = df['experiment'].str.split('-').str[0]
    speciesPerSeef = df.groupby(['scenario', 'experiment']).size()
    summary = speciesPerSeef.groupby('scenario').agg(['mean', 'std'])

    order = ['walker', 'bridge', 'mixed']
    labels = ['Walker Map', 'Bridge Map', 'Multi Map']
    colors = ['#d62728', '#1f77b4', '#ff7f0e']
    
    means = summary.loc[order, 'mean']
    stds = summary.loc[order, 'std']

    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors,
                alpha=0.85, edgecolor='black', linewidth=0.6)

    ax.set_ylabel('Number of species (Phenotype)')
    ax.set_title('Avg. Number of Species per Scenario (Phenotype)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 2, f'{m:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(f"{outputPath}/Avg. Number of Penotyphic Species per Scenario.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {outputPath}/Avg. Number of Penotyphic Species per Scenario.png")

def print_phenotypic_counts_boxplot(logdir: str):
    outputPath = os.path.join(Path(logdir), "Graphs")
    os.makedirs(outputPath, exist_ok=True)

    speciationCsv = f"{logdir}{os.sep}phenotypic_speciation_all.csv"
    df = pd.read_csv(speciationCsv)
    df["scenario"] = df["experiment"].str.split("-").str[0]

    speciesPerSeed = df.groupby(["scenario", "experiment"]).size()

    order = ["walker", "bridge", "mixed"]
    labels = ["Walker Map", "Bridge Map", "Multi Map"]
    colors = ["#d62728", "#1f77b4", "#ff7f0e"]

    # Extract the distribution data for each scenario in order
    data_to_plot = [speciesPerSeed.loc[scenario].values for scenario in order]

    fig, ax = plt.subplots(figsize=(5, 4.5))

    # Create boxplot
    bp = ax.boxplot(
        data_to_plot,
        patch_artist=True,  # Enables custom fill colors
        tick_labels=labels,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
    )

    # Apply custom colors to boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_ylabel("Number of species (Phenotype)")
    ax.set_title("Distribution of Species per Scenario (Phenotype)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(
        f"{outputPath}/Boxplot Number of Penotyphic Species per Scenario.png",
        dpi=300,
    )
    plt.close(fig)
    print(
        f"Saved: {outputPath}/Boxplot Number of Penotyphic Species per Scenario.png"
    )

def print_genotypic_counts_boxplot(logdir: str):
    outputPath = os.path.join(Path(logdir), "Graphs")
    os.makedirs(outputPath, exist_ok=True)

    speciationCsv = f"{logdir}{os.sep}genotypic_speciation_all.csv"
    df = pd.read_csv(speciationCsv)
    df["scenario"] = df["experiment"].str.split("-").str[0]

    speciesPerSeed = df.groupby(["scenario", "experiment"]).size()

    order = ["walker", "bridge", "mixed"]
    labels = ["Walker Map", "Bridge Map", "Multi Map"]
    colors = ["#d62728", "#1f77b4", "#ff7f0e"]

    data_to_plot = [speciesPerSeed.loc[scenario].values for scenario in order]

    fig, ax = plt.subplots(figsize=(5, 4.5))

    bp = ax.boxplot(
        data_to_plot,
        patch_artist=True,  
        tick_labels=labels,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_ylabel("Number of species (Genotype)")
    ax.set_title("Distribution of Species per Scenario (Genotype)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(
        f"{outputPath}/Boxplot Number of Genotyphic Species per Scenario.png",
        dpi=300,
    )
    plt.close(fig)
    print(
        f"Saved: {outputPath}/Boxplot Number of Genotyphic Species per Scenario.png"
    )

def get_best_bots_from_seed(seed:int, topperc:float):
    CHOSEN_SEED = seed
    TOP_N = topperc

    #Grab unique ID bots from parquet, merge with representative df, filter only the asked seed
    multi_taskmap = TASKMAPS["multi"]
    df, _, _ = loaders.load_parquet_log("log/completeData.parquet")
    df['seed'] = df['seed'].astype(int)
    df_unique = df.drop_duplicates(subset=['id', 'experiment', 'seed'], keep='first')
    representatives = pd.read_csv("log/genotypic_representatives_all.csv")
    representatives['seed'] = representatives['experiment'].apply(tools.get_seed)
    representatives['scenario_experiment'] = representatives['experiment'].str.split('_seed').str[0]
    merged = representatives.merge(
        df_unique[['id', 'experiment', 'seed', 'pos', 'shape', 'fit_Walker-v0', 'fit_BridgeWalker-v0']],
        left_on=['robot_id', 'seed', 'scenario_experiment'],
        right_on=['id', 'seed', 'experiment'],
        how='left'
    )
    # missing = merged['pos'].isna().sum()
    # print(f"{missing} representantes não encontrados no parquet")
    merged['pos'] = merged['pos'].apply(tools.parse_pos)
    merged_seed = merged[merged['seed'] == CHOSEN_SEED].copy()
    #separate by scenario and task
    walkerMap = merged_seed[merged_seed['scenario_experiment'] == 'walker-randomSelectAge50-20x5']
    bridgeMap = merged_seed[merged_seed['scenario_experiment'] == 'bridge-randomSelectAge50-20x5']

    mixedMap = merged_seed[merged_seed['scenario_experiment'] == 'mixed-randomSelectAge50-20x5'].copy()
    mixedMap['task'] = mixedMap['pos'].apply(lambda p: tools.pos_to_task(p, multi_taskmap))

    mixedWalker = mixedMap[mixedMap['task'] == 'Walker-v0']
    mixedBridge = mixedMap[mixedMap['task'] == 'BridgeWalker-v0']
    #get the top best
    def topNByTitness(group: pd.DataFrame, fitness_col: str, n: int = TOP_N) -> pd.DataFrame:
        return group.nlargest(n, fitness_col)

    walkerTop = topNByTitness(walkerMap, 'fit_Walker-v0')
    bridgeTop = topNByTitness(bridgeMap, 'fit_BridgeWalker-v0')
    mixedWalkerTop = topNByTitness(mixedWalker, 'fit_Walker-v0')
    mixedBridgeTop = topNByTitness(mixedBridge, 'fit_BridgeWalker-v0')
    #see results
    # print("\n--- Walker Map (top", TOP_N, ") ---")
    # print(walkerTop[['id', 'shape', 'fit_Walker-v0']])

    # print("\n--- Bridge Map (top", TOP_N, ") ---")
    # print(bridgeTop[['id', 'shape', 'fit_BridgeWalker-v0']])

    # print("\n--- Multi Map / Walker-v0 half (top", TOP_N, ") ---")
    # print(mixedWalkerTop[['id', 'shape', 'fit_Walker-v0']])

    # print("\n--- Multi Map / BridgeWalker-v0 half (top", TOP_N, ") ---")
    # print(mixedBridgeTop[['id', 'shape', 'fit_BridgeWalker-v0']])
    return walkerTop, bridgeTop, mixedWalkerTop, mixedBridgeTop

def shape_to_rgb(shape) -> np.ndarray:
    """builds and img of robot body"""
    if isinstance(shape, str):
        shape = ast.literal_eval(shape)
    shape = np.array(shape)
    h, w = shape.shape
    img = np.zeros((h, w, 3))
    for voxel_type, hex_color in VOXEL_COLORS.items():
        mask = shape == voxel_type
        rgb = np.array([int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5)])
        img[mask] = rgb
    return img

def build_column(top_df, fitness_col, task_label=None):
    """Function used to build morphology figure. it builds a column of bots given a df"""
    entries = []
    for _, row in top_df.iterrows():
        subtitle = f"{row[fitness_col]:.2f}"
        if task_label:
            subtitle = f"{task_label}\n{subtitle}"
        entries.append((row['shape'], row[fitness_col], subtitle))
    return entries

def print_gallery_by_seed(seed:int, nBots:int):
    """given a seed and nbots, prints an img where columns are tasks and maps. each line has a bot from the nbots best of that task and that seed"""
    seed=7
    nBots = 5
    walkerTop, bridgeTop, mixedWalkerTop, mixedBridgeTop = get_best_bots_from_seed(seed,nBots)
    walkerColumn = build_column(walkerTop, 'fit_Walker-v0', task_label='Walker-v0')
    bridgeColumn = build_column(bridgeTop, 'fit_BridgeWalker-v0', task_label='BridgeWalker-v0')
    mixedColumnWalker = build_column(mixedWalkerTop, 'fit_Walker-v0', task_label='Walker-v0')
    mixedColumnBridge = build_column(mixedBridgeTop, 'fit_BridgeWalker-v0', task_label='BridgeWalker-v0')
    columns = {
    'Walker Map': walkerColumn,
    'Multi Map (Walker)': mixedColumnWalker,
    'Bridge Map': bridgeColumn,
    'Multi Map (Bridge)': mixedColumnBridge
    }

    n_rows = nBots
    n_cols = len(columns)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    for colIdx, (map_name, entries) in enumerate(columns.items()):
        axes[0, colIdx].set_title(map_name, fontsize=14, fontweight='bold', pad=15)
        for row_idx in range(n_rows):
            ax = axes[row_idx, colIdx]
            if row_idx < len(entries):
                shape, fitness, subtitle = entries[row_idx]
                ax.imshow(shape_to_rgb(shape))
                ax.set_xlabel(subtitle, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f'log/bot-imgs/seed{seed}_robot_gallery.png', dpi=200, bbox_inches='tight')

def plot_multi_seed_gallery(seeds, save_path="log/bot-imgs/multi_seed_gallery.png"):
    """
    Generates a grid where each line is a seed and column a map/task
    """
    n_rows = len(seeds)
    
    col_configs = [
        ("Walker Map", "fit_Walker-v0", "Walker-v0"),
        ("Multi Map (Walker)", "fit_Walker-v0", "Walker-v0"),
        ("Bridge Map", "fit_BridgeWalker-v0", "BridgeWalker-v0"),
        ("Multi Map (Bridge)", "fit_BridgeWalker-v0", "BridgeWalker-v0")
    ]
    n_cols = len(col_configs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)

    for row_idx, seed in enumerate(seeds):
        walkerTop, bridgeTop, mixedWalkerTop, mixedBridgeTop = get_best_bots_from_seed(seed, topperc=1)
        dfs = [walkerTop, mixedWalkerTop, bridgeTop, mixedBridgeTop]
        for colIdx, ((mapName, fitCol, taskLabel), df) in enumerate(zip(col_configs, dfs)):
            ax = axes[row_idx, colIdx]
            if row_idx == 0:
                ax.set_title(mapName, fontsize=14, fontweight='bold', pad=15)
            if colIdx == 0:
                ax.set_ylabel(f"Seed {seed}", fontsize=12, fontweight='bold')
            if not df.empty:
                row = df.iloc[0]
                subtitle = f"{taskLabel}\n{row[fitCol]:.2f}"
                ax.imshow(shape_to_rgb(row['shape']))
                ax.set_xlabel(subtitle, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

def print_age_graph(logdir:str, title:str="Title", xLabel:str="x", yLabel:str="y",
                    figSize:tuple=(10,10), colors:list[str]=["red","blue","purple","green","pink","gray","black"]):
    """
    Reads a parquet file.
    Prints a graph of the avg fit of population in each generation across seeds.
    """
    taskMap = {}
    parentDir = os.path.dirname(logdir)
    taskMapJsonPath = os.path.join(parentDir,"taskMaps.json")
    with open(taskMapJsonPath, "r") as f:
        taskMap = json.load(f)
    df, _, _ = loaders.load_parquet_log(logdir) 
    outputPath = os.path.join(Path(logdir).parent, "Graphs")
    os.makedirs(outputPath, exist_ok=True)
    data = builders.build_age_average_data(df, taskMap)
    _print_line_graph(data, outputPath, title, xLabel, yLabel, figSize, colors)

if __name__=="__main__":
    # print_age_graph("log/completeData.parquet","Average Age",xLabel="Generations",yLabel="Average Age")
    # import matplotlib.pyplot as plt
    # seeds = [7, 49, 343, 2401, 16807]  # Passe a lista de seeds desejada

    # print_phenotypic_counts("log")
    print_genotypic_counts_boxplot("log")
    # plot_multi_seed_gallery(seeds)

    # print_population_best_fitness(logdir="log/completeData.parquet", title="Avg. Best-Individual Fitness Progression\nacross Scenarios and Tasks", xLabel="Generation", yLabel="Avg. of Max Fitness")
#     representativesID = [9094, 12284, 18988, 26496, 48341]
#     lastIndividuals = [24571, 32767, 20247, 49100, 48341]
# # [9094, 12284, 18988, 26496, 48341]
#     for rep in representativesID:
#         print_matteo_bot_from_id("log/mixed-randomSelectAge50-20x5_seed7_CGA_08271513", rep)
    # print_task_map("log/mixed-randomSelectAge50-20x5_seed49_CGA_08271831","MixedMap.png",figSize=(12,3),title="Task organization in Mixed Map")
    # print_matteo_bot("log/tester_seed7_CGA_08250033/bots/generation_1.pkl",(1,0))

    # rootLog = "log"
    # for folder in os.listdir(rootLog):
    #     if not(folder.startswith("walker") or folder.startswith("bridge") or folder.startswith("mixed")): continue
    #     expFolder = f"{rootLog}{os.sep}{folder}"
    #     print_directional_hammming_map_gif(logdir=expFolder,taskColors=["green"], frameInterval=1, frameDuration=30)
    # print_directional_hammming_map_gif(logdir="log/mixed-randomSelectAge50-20x5_seed343_CGA_08281513",taskColors=["green"], frameInterval=100)
    # print_matteo_specialization_graph(logFolder="log/bridge-randomSelectAge50-20x5_seed7_CGA_08271609")
    # print_directional_hammming_map_gif("log/EXP1-300gen/mixed_10x5_seed7_CGA_08101954", ["red","blue"], 5, 300, False)
    # print_bot("log/EXP1-300gen/mixed_10x5_seed7_CGA_08101954",500,(0,0))
    # print_bot("log/EXP1-300gen/mixed_10x5_seed7_CGA_08101954",500,(0,1))



    # print_directional_hammming_map_gif(logdir="log/v3-thin-grid/multiTaskv3_seed7_CGA_06062117",
    #                                     taskColors=["green","purple"])

    # print_hammig_hinter_intra_task(logdir="log/v3-thin-grid/completeData.parquet")
    # print_global_hamming_distance_per_seed(logdir="log/v3-thin-grid/completeData.parquet")
    # print_hamming_inter_exp(logdir="log/v1/completeData.parquet", task2becompared="bridgewalker", exp2compare=("baseline-BridgeWalkerv1","quadrantv1"))
    # print_best_bots(logdir="log/v1/completeData.parquet")
    # tools.get_robot_with_fitness(logdir="log/v1/quadrantv1_seed7_CGA_04302108", minFit=60, maxFit=61)
    # print_bot(logdir="log/v1/quadrantv1_seed7_CGA_04302108", gen=931, pos=(4,7))

    # print_last_gen_specialization_graph(logdir="log/v2/baseline-BridgeWalkerv2_seed7_CGA_05221444", title="test",
    #                                             xLabel="x", yLabel="y", figSize=(10,10))


