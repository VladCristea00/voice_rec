import functions
import numpy as np
import pickle

# PROJECT_PATH = "/home/ubadmin/voice_rec/"
PROJECT_PATH = "C:/Users/Vlad/Desktop/voice_rec/"

DATASET_PATH = PROJECT_PATH + "mydataset/"

mapare = {
    'calendar': 0,
    'dieta': 1,
    'meniu': 2,
    'profil': 3,
    'setari': 4
}

fs = 16000

audio_paths = functions.load_paths(DATASET_PATH)
audio_keys = list(audio_paths.keys())
np.random.shuffle(audio_keys)

n_files = len(audio_keys)
n_train = round(n_files * 0.7)
n_valid = n_files - n_train
train_files = audio_keys[:n_train]
valid_files = audio_keys[n_train:]

valid_labels = [mapare[audio_paths[file]] for file in valid_files]
valid_specs = [functions.mel_spec(functions.audio_convert2(path), fs) for path in valid_files]

train_labels = [mapare[audio_paths[file]] for file in train_files]
train_specs = [functions.mel_spec(functions.audio_convert2(path), fs) for path in train_files]

with open(PROJECT_PATH + "datasets/ds1_no_aug", 'wb') as f:
    pickle.dump((train_specs, train_labels, valid_specs, valid_labels), f)
    f.close()

print("done")
