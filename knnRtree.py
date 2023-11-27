
from rtree import index
import numpy as np
import time
import os

class RtreeIndex:
    def __init__(self, features, query_feature) -> None:
        start_time = time.time()

        self.features = features
        self.query_features = query_feature
        prop = index.Property()
        prop.dimension = 512  
        prop.buffering_capacity = 20
        self.idx = index.Index(properties=prop)
            
        for i in range(len(self.features)):
            path, feature = self.features[i]# [0,1.1,0.5,1.2,.....]
                    # transformarndolo a bounding box falsas
            coordenadas = feature + feature
            self.idx.insert(id=i, coordinates=coordenadas)

        end_time = time.time()
        construction_time = end_time - start_time
        print(f"Tiempo de construcción: {construction_time} segundos")

    
    def knnSearch(self, k):
        start_time = time.time()

        results = []   
        # transformandolo a bounding box falsas
        coord_query = self.query_features + self.query_features

        nearest = list(self.idx.nearest(coordinates=coord_query, num_results=k))

        for ids in nearest:
            path,feature = self.features[ids]
            dist = np.linalg.norm(feature - self.query_features)
            results.append((dist, path))

        end_time = time.time()
        query_time = end_time - start_time

        return results, query_time
    