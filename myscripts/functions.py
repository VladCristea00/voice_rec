import os
import tensorflow as tf
import tensorflow_io as tfio
import librosa
import numpy as np

def audio_convert(audio_path):
    audio_binary, fs = librosa.load(audio_path, sr=16000, mono=False)
    mono = librosa.to_mono(audio_binary)

    binary = mono[4800:20800]

    return binary

def audio_cut(audio, fs):
    audio_len = audio.shape[0]

    if audio_len > fs:
        audio = audio[:fs]
    elif audio_len < fs:
        start_pad_len = int(np.floor(fs - audio_len)/2)
        end_pad_len = fs - audio_len - start_pad_len

        start_pad = np.random.uniform(-0.001, 0.001, start_pad_len)
        end_pad = np.random.uniform(-0.001, 0.001, end_pad_len)


        audio = np.concatenate([start_pad, audio, end_pad], dtype='float32')

    return audio

def audio_convert2(audio_path):
    audio_binary, fs = librosa.load(audio_path, sr=16000)

    audio_binary = audio_cut(audio_binary, fs)

    return audio_binary

def load_paths(ds_path):
    folders = os.listdir(ds_path)
    paths = {}

    for command in folders:
        wavs = os.listdir(ds_path + command)
        for wav in wavs:
            path = ds_path + command + "/" + wav
            paths[path] = command

    return paths

def mel_spec(audio, fs, n_mels=64, n_fft=255, hop_len=125):
    audio_t = tf.convert_to_tensor(audio, dtype='float32')
    spectrogram = tfio.audio.spectrogram(audio_t, nfft=n_fft, window=n_fft, stride=hop_len)
    mel_spect = tfio.audio.melscale(spectrogram, fs, mels=n_mels, fmin=0, fmax=8000)
    mel_db = tfio.audio.dbscale(mel_spect, top_db=80)

    result = tf.reverse(tf.transpose(mel_db), [0])

    result = result[..., tf.newaxis]
    return result

