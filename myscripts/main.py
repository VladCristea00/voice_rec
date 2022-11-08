import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import functions
import tensorflow as tf
import numpy as np
import sounddevice as sd

import webbrowser
import pyttsx3




PROJECT_PATH = "C:/Users/Vlad/Desktop/voice_rec/"
open_sound = PROJECT_PATH + "open_sound.wav"
close_sound = PROJECT_PATH + "close_sound.wav"

fs = 16000
model_name = "model1"
model = tf.keras.models.load_model(f"{PROJECT_PATH}models/{model_name}")

mapare = {
    0: 'calendar',
    1: 'dieta',
    2: 'meniu',
    3: 'profil',
    4: 'setari'
}


engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)



def speak(audio):
    engine.say(audio)
    engine.runAndWait()
    engine.setProperty('rate', 140)

    if audio == "setari":
        speak("Vă rog selectați ce doriți să modificați: 1. Notificări, 2. Limbă, 3.Locație, 4.Politică de confidențialitate, 5.Deconectare")
    elif audio == "dieta":
        speak("1.Adaugă mâncare, 2.Adaugă apă, 3.Adaugă suplimente alimentare, 4.Efort sau Exerciții fizice")
    elif audio == "calendar":
        speak("Din această pagină vă puteți verifica istoricul meselor apăsând pe calendarul afișat")
    elif audio == "meniu":
        speak("1. Mic dejun, 2. Masa de prînz, 3. Cină, 4.Gustări între mese, 5. Recomandarea zilei, 6. Desert, 7. Sucuri de fructe ")
    elif audio == "profil":
        speak("1.Adaugă poză de profil, 2. Adaugă înălțimea, 3.Modifică greutatea actuală, 4.Greutate dorită, 5.Exerciții favorite, 6.Alimente preferate")


def audio_frame(audio, fs):
    max_value = int(np.argmax(audio, axis=0))
    audio = audio[..., 0]

    if max_value < fs/2:
        rec = audio[:max_value + round(fs/2)]
        rec = functions.audio_cut(rec, fs)
    elif max_value > (2*rec_duration-1)*fs/2:
        rec = audio[max_value - round(fs/2):]
        rec = functions.audio_cut(rec, fs)
    else:
        rec = audio[max_value - round(fs/2):max_value + round(fs/2)]

    return rec

rec_duration = 2

speak("Vorbiți")
print("Rostește comanda")

myrec = sd.rec(fs * rec_duration, samplerate=fs, channels=1, dtype='float32')
sd.wait()

print(f"Recunoaștere încheiată")

framed = audio_frame(myrec, fs)

spect = functions.mel_spec(framed, fs)
spect = spect[None, ...]

prediction = model.predict(spect)
idx = int(np.argmax(prediction))

print(f"Predicted command: '{mapare[idx]}' Confidence: {float(prediction[:, idx]):5f}")

webbrowser.open(f"file:///C:/Users/Vlad/Desktop/Site/{mapare[idx]}.html")



speak(mapare[idx])
