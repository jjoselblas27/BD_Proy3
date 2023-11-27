import numpy as np
import faiss
import time

# features es losfeatures ya leidos

class LSHIndex:
    def __init__(self, features, query_feature):
        start_time = time.time()
        self.features = features
        self.query_features = query_feature

        dimention = 512
        num_bits = 32
        self.index = faiss.IndexLSH(dimention, num_bits)
        self.index.add(np.ascontiguousarray(np.asarray([i[1] for i in self.features], dtype="float32")))
        end_time = time.time()
        construction_time = end_time - start_time
        print(f"Tiempo de construcción: {construction_time} segundos")


    def knn_query(self, k):
        start_time = time.time()
        resultados = []
        distances, id_array = self.index.search(x=np.asarray([self.query_features], dtype="float32"), k=k)

        for pos, id in enumerate(id_array[0]):
            #print(pos, id)            
            resultados.append((distances[0][pos], self.features[id][0])) # (rank, path)
        
        end_time = time.time()
        query_time = end_time - start_time
        
        return resultados, query_time