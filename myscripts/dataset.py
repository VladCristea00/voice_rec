import os
import functions
import soundfile

# PROJECT_PATH = "/home/ubadmin/voice_rec/"
PROJECT_PATH = "C:/Users/Vlad/Desktop/voice_rec/"

comenzi_path = PROJECT_PATH + "comenzi-test"
dataset_path = PROJECT_PATH + "testfiles"

comenzi = os.listdir(comenzi_path)

for comanda in comenzi:
    audio_files = os.listdir(comenzi_path + "/" + comanda)

    index = 0
    for audio in audio_files:
        audio_rec = functions.audio_convert(comenzi_path + "/" + comanda + "/" + audio)
        write_path = dataset_path + "/" + comanda.split("-")[0].lower() + "_" + str(index) + ".wav"
        soundfile.write(write_path, audio_rec, 16000)
        index += 1


print("Done")