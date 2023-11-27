from knnLSH import *
from knnSequential import *
from knnRtree import *
from knnSequential import KNNSequential
from transformers import AutoFeatureExtractor, ClapModel
from feature_v2 import extraction_feature
import pickle
import time

def prueba_range(features, query_features):

    knn = KNNSequential(features, query_features)

    # Prueba de range search
    radius = 0.5
    results, query_time = knn.range_search(radius)

    print("Tiempo de query: ", query_time)
    print("Resultados de range search: ")
    print(len(results))
    for dist, path in results:
        print("Distancia: ", dist, " Path: ", path)

def prueba_heap(features, query_features):

    knn = KNNSequential(features, query_features)

    # Prueba de range search
    k = 8
    results, query_time = knn.heap_search_maxHeap(k)

    print("Tiempo de query: ", query_time)
    print("Resultados de range search: ")
    print(len(results))
    for dist, path in results:
        print("Distancia: ", dist, " Path: ", path)


def prueba_lsh(features, query_features):

    k = 8
    knn = LSHIndex(features, query_features)
    results, query_time = knn.knn_query(k)

    print("Tiempo de query: ", query_time)
    print("Resultados de LST: ")
    print(len(results))
    for dist, path in results:
        print("Distancia: ", dist, " Path: ", path)

def prueba_rtree(features, query_features):

    
    k = 8
    rtree = RtreeIndex(features, query_features)
    results, query_time = rtree.knnSearch(k)

    print("Tiempo de query: ", query_time)
    print("Resultados de rtree: ")
    print(len(results))
    for dist, path in results:
        print("Distancia: ", dist, " Path: ", path)


def main():
    
    # Cargar el modelo y el extractor de características
    query_path = './query.mp3'
    features_file = './datos/audio_features_v2.pkl'

    infile = open(features_file, mode="rb")
    features = pickle.load(infile)
    infile.close()

    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")

    query_features = extraction_feature(query_path, model, feature_extractor)


    features_red = features
    print(8000)
    prueba_heap(features_red, query_features)
    prueba_lsh(features_red, query_features)
    prueba_rtree(features_red, query_features)
    """
    prueba = [4, 8, 16, 32, 64]
    for i in prueba:
        print("num_bits: ", i)
        prueba_lsh(i, features, query_features)
        print("----------------------------------------------------")
    """

if __name__ == "__main__":
    main()

