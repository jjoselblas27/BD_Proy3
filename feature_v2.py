import os
import numpy as np
import pickle
from transformers import AutoFeatureExtractor, ClapModel
import librosa
import torch

# fuente : https://huggingface.co/docs/transformers/main/en/model_doc/clap


def extraction_feature(audio_path, model, feature_extractor):
    # Cargar el archivo de audio usando librosa
    audio, sr = librosa.load(audio_path)
    audio_tensor = torch.tensor(audio)
    inputs = feature_extractor(audio_tensor, return_tensors="pt")
    # Obtener las características del audio usando el modelo
    audio_features = model.get_audio_features(**inputs)

    audio_features_numpy = audio_features.detach().numpy()[0]

    return audio_features_numpy


def main():
    # Cargar el modelo y el extractor de características
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")

    path_features = "./datos/audio_features_v2.pkl"

    outfile = open(path_features , mode="wb")
    output = []

    path = os.path.join(os.getcwd(), "fma")

    for subdir, dirs, files in os.walk(path):
        for file in files:
            print("Procesando: ", file)
            if file.endswith(".mp3"):
                audio_file = os.path.join(subdir, file)
                audio_features = extraction_feature(audio_file, model, feature_extractor)
                output.append((audio_file, audio_features))
    
    pickle.dump(output, outfile)
    outfile.close()


if __name__ == "__main__":
    main()
    print("Ejecutando features_v2.py")
