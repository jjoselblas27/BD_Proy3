
from rtree import index
import numpy as np


class RtreeIndex:
    def __init__(self, capacity, features, query_feature) -> None:

        self.features = features
        self.query_features = query_feature

        p = index.Property()
        p.dimension = 512  
        p.buffering_capacity = capacity  
        self.idx = index.Index(properties=p)
        
        for i in range(len(self.features)):
            path, feature = self.features[i]# [0,1.1,0.5,1.2,.....]
            # transformarndolo a bounding box falsas
            coordenadas = feature + feature
            self.idx.insert(id=i, coordinates=coordenadas)

    
    def knnSearch(self, k):
        results = []   
        # transformandolo a bounding box falsas
        coord_query = self.query_features + self.query_features

        nearest = list(self.idx.nearest(coordinates=coord_query, num_results=k))

        for ids in nearest:
            path,feature = self.features[ids]
            dist = np.linalg.norm(feature - self.query_features)
            results.append((dist, path))

        return results
