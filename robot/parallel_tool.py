import math 
from pathos.multiprocessing import ProcessPool
import dill
import multiprocess 
import errno 
import time

class ParallelTool :

    def __init__(self, bot, cpus) :
        self.cpus = cpus 
        self.timeout = bot.params.get("timeout", 25)


    def run(self, function, chunk) : 
        return self._process_parallel(function, chunk)
            


    # def _cut_chunk(self, chunk) : 
        
    #     chunks = []
    #     length = len(chunk)
    #     size_of_chunk = math.ceil(length / self.cpus)
    #     for i in range(0, length, size_of_chunk) :
    #         chunks.append(chunk[i:i+size_of_chunk])
    #     return chunks

    # def _worker(self, function, chunk) : 
    #     results = []
    #     for arguments in chunk : 
    #         results.append(function(*arguments))
    #     return results
    
    # def _process_parallel(self, function, chunk) : 

    #     chunks = self._cut_chunk(chunk)
    #     process_worker = lambda chunk : self._worker(function, chunk)

    #     pool = ProcessPool(nodes = self.cpus)
    #     results = pool.map(
    #             process_worker,
    #             chunks
    #         )
    #     pool.close()
    #     pool.join()
    #     pool.clear()
        
    #     results_formatted = []
    #     for list_of_result in results : 
    #         for result in list_of_result : 
    #             results_formatted.append(result)
        
    #     return results_formatted 


    def _process_parallel(self, function, chunk) :

        pool = multiprocess.Pool(self.cpus, maxtasksperchild=1)
        try :
            ars = [(args[0], pool.apply_async(function, args)) for args in chunk]

            # budget global : nb de vagues x timeout, au lieu de 25 s par item
            batches = math.ceil(len(chunk) / self.cpus)
            deadline = time.monotonic() + self.timeout * batches

            results = []

            for entity_id, ar in ars :
                # une fois la deadline passee on n'attend plus, mais timeout=0
                # ramasse quand meme les resultats deja prets
                remaining = max(0, deadline - time.monotonic())

                try :
                    results.append(ar.get(timeout=remaining))
                except Exception :
                    results.append(-1000)

            return results

        finally :
            pool.terminate()
            pool.join()