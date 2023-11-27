# implementacion del kkn para range search y knn usando heap.
import numpy as np
from maxHeap import MaxHeap
import time

class KNNSequential:
    def __init__(self, features, query_features):

        self.features = features
        self.query_features = query_features

    def range_search(self, radius):
        start_time = time.time()

        results = []
        # Calcular la distancia euclidiana entre el query y cada una de las características
        for path, feature in self.features:
            dist = np.linalg.norm(feature - self.query_features)

            if(dist < radius):
                results.append((dist, path))

        sorted_results = sorted(results, key=lambda x: x[0])

        end_time = time.time()
        query_time = end_time - start_time
        return sorted_results, query_time

    
    def heap_search_maxHeap(self, k):
        start_time = time.time()

        heap = MaxHeap()

        for path, feature in self.features:
            dist = np.linalg.norm(feature - self.query_features)
            
            if(len(heap) < k):
                heap.push([dist, path])
            else:
                if(heap.peek()[0] > dist):
                    heap.pop()
                    heap.push([dist, path])
        
        heapList = heap.heap
        
        end_time = time.time()
        query_time = end_time - start_time

        return heapList, query_time


