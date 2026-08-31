import numpy as np, time, json, time, os
import Search as Search
import pickle
from typing import Optional
from multiprocessing import Pool
from robot.basicrobot import SinRobot as Robot, get_random, get_fromfile

class CGA():
    def __init__(self, logdir, prefix, save_interval, numprocs, robotModule, worldModules, sim_step:int, gridWorlds:list[list[int]], maxGeneration:int, toroidal:bool=False, mutationChance:float = 0.05, rng: Optional[np.random.Generator]=None, botsparam: Optional[dict]=None):
        self._random = rng if rng is not None else np.random.default_rng()
        self._robot = robotModule
        self._robotType = str(robotModule.__name__).split("robot.")[1]
        self._botparam = botsparam
        self._worlds = worldModules
        self._gridWorlds = gridWorlds
        self.rows = len(self._gridWorlds) #assumes rectangle, always.
        self.cols = len(self._gridWorlds[0])
        self._sim_step = sim_step
        self._numprocs = numprocs
        self._prefix = prefix
        self._logdir = logdir
        self._save_interval = save_interval
        self.toroidal = toroidal
        self.grid:dict[tuple[int,int],'Robot'] = {}
        self.lastGen = maxGeneration
        self.mutationChance = mutationChance
                        
        self.currentGen:int = 0
        self.robotCounter:int = 0
        self._botsLog = f"{self._prefix}{os.sep}robots_log.jsonl"
        self.meanTime = []

        #Best Dict
        self._bestDict = {"pos":(-1,-1), "fit":-1, "robot":None}  
        self.buildTaskMap()

    def buildTaskMap(self):
        self._taskMap = {}
        for y in range(self.rows):
            for x in range(self.cols):
                index = self._gridWorlds[y][x]
                self._taskMap[(x,y)] = self._worlds[index]

    def get_moore_neighbors(self, pos: tuple[int,int]) -> list[tuple[int,int]]:
        """Returns moore neighbors depending of [self.toroidal] value"""
        x = pos[0]
        y = pos[1]
        output:list[tuple[int,int]] = []
            #Consider toroidal neighbors
        if self.toroidal:
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i == 0 and j == 0: continue
                    # Apply Toroidal effect
                    neighPosX = (x + i) % self.cols
                    neighPosY = (y + j) % self.rows
                    output.append((neighPosX,neighPosY))
            return output
        else:
            #Not consider toroidal neighbors
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    if i==0 and j==0: continue
                    neighPosX, neighPosY = x+i, y+j
                    if (neighPosX < 0 or neighPosY < 0): continue
                    elif (neighPosX > self.cols-1 or neighPosY > self.rows-1): continue
                    output.append((neighPosX,neighPosY))  
            return output
    
    def hamming_distance(self, shape1: list, shape2: list) -> float:
        A = np.array(shape1).flatten()
        B = np.array(shape2).flatten()
        
        maxDist = max(A.size, B.size) #A and B MUST have the same size!
        dist = np.sum(A != B)
        return dist/maxDist

    def reset(self) -> None:
        evalpars = []
        botsList = []
        open(self._botsLog, "w").close()         # start bots log; all bots will be written here

        # os.makedirs(self._prefix, exist_ok=True)

        #random bots as start population
        for y in range(self.rows):
            for x in range(self.cols):
                if self._botparam==None:
                    robot = self._robot.get_random(rng=self._random)
                else:
                    pos = (x,y)
                    taskName = self._taskMap[pos]
                    type_env = self._botparam["env_name"].index(taskName)
                    robot = self._robot.get_random(params=self._botparam,
                                                   type_env=type_env,
                                                   w=5,
                                                   h=5,
                                                   rng=self._random)
                
                self.robotCounter += 1
                robot.id = self.robotCounter
                botsList.append(((x,y), robot))
                state = [self.currentGen, self._prefix]
                evalpars.append((self._robotType, robot, self._taskMap[(x,y)], self._sim_step, state))
        #Evaluate bots
        with Pool(self._numprocs) as p:
            scoresResult = p.starmap_async(Search.evaluate, evalpars)
            scores = scoresResult.get(timeout=600)
            # scores = p.starmap(Search.evaluate, evalpars)
        
        for s in scores:
            self.meanTime.append(s[1])

        #fill grid
        gridSnapShot = {}
        for i, (pos, bot) in enumerate(botsList):
            taskName = self._taskMap[pos]
            bot.fit[taskName] = scores[i][0]
            # self.check_best(bot, pos, "None")
            self.grid[pos] = bot 
            bot.pos = pos
            self.log_robot(robot=self.grid[pos], gen=self.currentGen, pos=pos)

        self.save_gen_pickle(self.grid, 0)

        # log task map
        jsonTaskMap = {}
        for (x,y) in self._taskMap:
            jsonTaskMap[f"({x},{y})"] = self._taskMap[(x,y)]

        with open(f"{self._prefix}{os.sep}grid_taskMap.json", "w") as out_f2:
            json.dump(jsonTaskMap, out_f2, separators=(',', ':'))
            

    def log_robot(self, robot:"SinRobot", gen:int, pos:tuple[int,int], parent2id:int=-1):
        entry = {
            "id":      robot.id,
            "gen":     gen,
            "pos":     list(pos),
            "parent2": parent2id,   # parent1 is the bot that occupied this position in the last gen
            "fit":     robot.fit,
            "shape":   robot.shape.tolist()
        }
        with open(self._botsLog, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def select(self, currentPos:tuple[int,int], neighbors: list[tuple[int,int]]) -> tuple[int,int]:
        picked = self._random.choice(len(neighbors))
        return neighbors[picked]
        # taskName = self._taskMap[currentPos] #type(self._taskMap[currentPos]).__module__
        # toEvaluate = []
        # evalpars = []

        # #check if any neighbors need evaluation in current cell task
        # for pos in neighbors:
        #     if taskName in self.grid[pos].fit: continue
        #     else:
        #         robot = self.grid[pos]
        #         world = self._taskMap[currentPos]

        #         toEvaluate.append((pos,robot))
        #         evalpars.append((self._robotType, robot, world, self._sim_step))


        # #if any neighbors need evaluation in current cell task
        # if evalpars: 
        #     with Pool(self._numprocs) as p:
                        # scoresResult = p.starmap_async(Search.evaluate, evalpars)
                        # scores = scoresResult.get(timeout=600)
        #         scores = p.starmap(Search.evaluate, evalpars)
        #     for s in scores:
        #         self.meanTime.append(s[1])
            
        #     for i, (pos, robot) in enumerate(toEvaluate):
        #         robot.fit[taskName] = scores[i][0]

        # #rank selection
        # sortedNeighbors = sorted(neighbors, key=lambda pos: self.grid[pos].fit[taskName], reverse=True)

        # n = len(neighbors)
        # weights = [1 / (2**i) for i in range(n)]
        # weights[-1] = weights[-2] / 2
        # total = sum(weights)
        # weights = [w / total for w in weights]

        # picked = self._random.choice(len(neighbors), p=weights)
        # return sortedNeighbors[picked]
    
    def update(self):        
        self.currentGen += 1
        # if self.currentGen % (self.lastGen*(0.1))==0: print(f"Gen: {self.currentGen}")
        childrenList = []
        evalpars = []

        #generate children
        for y in range(self.rows):
            for x in range(self.cols):
                localWorld = self._taskMap[(x,y)]
                parent1 = self.grid[(x,y)]
                p1Neighbors = self.get_moore_neighbors(pos=(x,y))
                parent2pos = self.select((x,y),neighbors=p1Neighbors)
                parent2 = self.grid[parent2pos]
                child = parent1.crossover(mate=parent2)
                if self._random.random() <= self.mutationChance:
                    child = child.mutate()
                
                self.robotCounter += 1
                child.id = self.robotCounter
                
                childrenList.append(((x,y), child, parent2.id))
                state = [self.currentGen, self._prefix]
                evalpars.append((self._robotType, child, localWorld, self._sim_step, state))

        #evaluate children that need evaluation
        if evalpars:
            with Pool(self._numprocs) as p:
                scoresResult = p.starmap_async(Search.evaluate, evalpars)
                scores = scoresResult.get(timeout=600)
                # scores = p.starmap(Search.evaluate, evalpars)
                p.close()
                p.join()
            for s in scores:
                self.meanTime.append(s[1])
            
            #fill fit of evaluated children
            scoreIdx = 0
            for i, (pos, child, parent2Id) in enumerate(childrenList):
                taskName = self._taskMap[pos]
                child.fit[taskName] = scores[scoreIdx][0]
                scoreIdx += 1

        #fill new grid
        newGrid = {}
        gridSnapShot = {}
        for i, (pos, child, parent2Id) in enumerate(childrenList):
            taskName = self._taskMap[pos]
            parent1 = self.grid[pos]
            parent2id = parent2Id
            #set to new grid (child goes if fit is same or better than parent)
            if (child.fit[taskName] >= parent1.fit[taskName]) or (parent1.age >= 50):
                newGrid[pos] = child
            else:
                newGrid[pos] = parent1
                newGrid[pos].age += 1
                parent2id = parent1.age * -1
            #save a bot in full log
            self.log_robot(robot=newGrid[pos], gen=self.currentGen, pos=pos, parent2id=parent2id)

        #update current grid for next gen
        self.grid = newGrid
        self.save_gen_pickle(self.grid, self.currentGen)

    def evaluate_on_all_tasks(self):
        """This function goes through all cells in [self.grid] and evaluates all robots in all tasks 
        (only the tasks the robot are missing)"""

        #get world instances of all tasks with their names
        taskNames = sorted(set(self._taskMap.values())) 

        worldInstances = {}
        for taskName in taskNames:
            for pos, world in self._taskMap.items():
                if world == taskName: #type(world).__module__ 
                    worldInstances[taskName] = world
                    break

        #group bots that need to be evaluated by task
        toEvaluate = {}
        for taskName in taskNames:
            toEvaluate[taskName] = []

        for pos, bot in self.grid.items():
            for taskName in taskNames:
                if taskName not in bot.fit:
                    toEvaluate[taskName].append((bot, pos))

        #evaluate bots 
        for taskName in taskNames:
            bots2Evaluate = toEvaluate[taskName]
            if len(bots2Evaluate)==0: continue

            evalpars = []
            for bot, pos in bots2Evaluate:
                world = worldInstances[taskName]
                state = [self.currentGen, self._prefix]
                evalpars.append((self._robotType, bot, world, self._sim_step, state))

            if evalpars:
                with Pool(self._numprocs) as p:
                    scoresResult = p.starmap_async(Search.evaluate, evalpars)
                    scores = scoresResult.get(timeout=600)
                    # scores = p.starmap(Search.evaluate, evalpars)
                for s in scores:
                    self.meanTime.append(s[1])

            #save scores in bots
            for i,(bot, pos) in enumerate(toEvaluate[taskName]):
                bot.fit[taskName] = scores[i][0]
                self.log_robot(robot=bot, gen=99999, pos=pos)

    def save_gen_pickle(self, grid:dict, gen:int):
        picklePath = f"{self._prefix}{os.sep}bots{os.sep}generation_{gen}.pkl"
        with open(picklePath, "wb") as file:
            pickle.dump(grid, file)