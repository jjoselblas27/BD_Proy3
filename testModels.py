from knnLSH import *
from knnSequential import *
from knnRtree import *
from knnSequential import KNNSequential
from transformers import AutoFeatureExtractor, ClapModel
from feature_v2 import extraction_feature
import pickle

def prueba_range(features, query_features):

    knn = KNNSequential(features, query_features)

    # Prueba de range search
    radius = 1
    results = knn.range_search(radius)

    print("Resultados de range search: ")
    print(len(results))
    for dist, path in results:
        print("Distancia: ", dist, " Path: ", path)


def prueba_lsh(features, query_features):
    num_bits = 4

    knn = LSHIndex(num_bits, features, query_features)
    results = knn.knn_query(3)

    print("Resultados de LST: ")
    print(len(results))
    for dist, path in results:
        print("Distancia: ", dist, " Path: ", path)

def prueba_rtree(features, query_features):

    capacity = 10
    rtree = RtreeIndex(capacity, features, query_features)
    results = rtree.knnSearch(3)

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

    #prueba_range(features, query_features)

    prueba_lsh(features, query_features)

    #prueba_rtree(features, query_features)

if __name__ == "__main__":
    main()

