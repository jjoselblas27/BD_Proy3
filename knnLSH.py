import numpy as np
import faiss

# features es losfeatures ya leidos

class LSHIndex:
    def __init__(self, num_bits, features, query_feature):
        self.features = features
        self.query_features = query_feature

        dimention = 512
        self.index = faiss.IndexLSH(dimention, num_bits)
        self.index.add(np.ascontiguousarray(np.asarray([i[1] for i in self.features], dtype="float32")))


    def knn_query(self, k):

        resultados = []
        distances, id_array = self.index.search(x=np.asarray([self.query_features], dtype="float32"), k=k)

        print(distances)
        print(id_array)

        for pos, id in enumerate(id_array[0]):
            #print(pos, id)            
            resultados.append((distances[0][pos], self.features[id][0])) # (rank, path)
        

        return resultados