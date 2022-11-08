import tensorflow as tf
import os
import functions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# PROJECT_PATH = "/home/ubadmin/voice_rec/"
PROJECT_PATH = "C:/Users/Vlad/Desktop/voice_rec/"

TESTFILES_PATH = PROJECT_PATH + "testfiles/"
model_name = "model1"

mapare = {
    'calendar': 0,
    'dieta': 1,
    'meniu': 2,
    'profil': 3,
    'setari': 4
}

fs = 16000
model = tf.keras.models.load_model(f"{PROJECT_PATH}models/{model_name}")

test_files = os.listdir(TESTFILES_PATH)

test_audios = [functions.audio_convert2(TESTFILES_PATH + file) for file in test_files]
test_labels = [mapare[file.split("_")[0]] for file in test_files]
test_spects = [functions.mel_spec(audio, fs).numpy() for audio in test_audios]

test_spects = np.array(test_spects)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_spects), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f"Test set accuracy = {test_acc:.3%}")

conf_mat = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(conf_mat,
            xticklabels=list(mapare.keys()),
            yticklabels=list(mapare.keys()),
            annot=True, fmt='g'
            )
plt.xlabel("Prediction")
plt.ylabel("Label")
plt.title(f"{model_name}")
plt.savefig(f"{PROJECT_PATH}models/{model_name}/conf_mat.png")

print("done")
