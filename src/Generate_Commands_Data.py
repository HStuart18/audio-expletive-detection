import soundfile as sf
import numpy as np
import os
import shutil
import librosa 
import librosa.display as display
import xml.etree.ElementTree as ET
from pydub import AudioSegment
from pydub.playback import play
from pydub.silence import split_on_silence
from pydub.generators import WhiteNoise
import random
import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import sklearn
import math

Fs = 16000
audio_length_samples = 16000
window_length_seconds = 0.02
window_length_samples = int(window_length_seconds * Fs)
NFFT = 2 ** math.ceil(math.log(window_length_samples, 2))
HOP_LENGTH = int(window_length_samples / 2)
c = 0

target_words = ["bed", "bird", "happy", "left", "up"]
folders = next(os.walk(r"D:\ML_Datasets\commands"))[1]

for file in os.listdir(r"D:\ML_Datasets\commands\marvin"):
    with open(r"D:\ML_Datasets\commands\marvin" + fr"\{file}", "rb") as f:
        audio, _ = librosa.load(f, sr=Fs)
        if len(audio) < audio_length_samples:
            continue
        samples = audio[:audio_length_samples]
        stft = librosa.stft(samples, n_fft=512, hop_length=256, win_length=window_length_samples)
        stft = stft[:-1, :]
        stft = np.append(stft, np.zeros(shape=(256, 1), dtype=np.complex128), axis=1)
        v = np.stack((stft.real,stft.imag),-1)
        stft_old = v.reshape(stft.shape + (2,))
        np.save(rf"C:\Users\Harry\Desktop\stft\marvin\{file[:-4]}.npy", stft_old)
        #stft = np.load(rf"C:\Users\Harry\Desktop\stft\{file[:-4]}.npy")
        #stft_new = stft.astype(float).view(np.complex128)
        #stft_new = np.squeeze(stft_new, axis=2)
        #y_hat = librosa.istft(stft_new, hop_length=HOP_LENGTH, win_length=window_length_samples)
        #librosa.output.write_wav("lol.wav", y_hat, Fs)

amendments = {word:0 for word in folders}

fig = plt.Figure()
canvas = FigureCanvas(fig)

for folder in folders:
    output_dir = r"C:\Users\Harry\Desktop\Data\other"
    if folder == "_background_noise_":
        pass
        #output_dir = fr"C:\Users\Harry\Desktop\Data\silence"
        #if not os.path.exists(output_dir):
            #os.makedirs(output_dir)

        #for file in os.listdir(r"D:\ML_Datasets\commands" + fr"\{folder}"):
            #with open(r"D:\ML_Datasets\commands" + fr"\{folder}" + fr"\{file}", "rb") as f:
                #audio, _ = librosa.load(f, sr=Fs)

            #for i in range(300):
                #ind = random.randint(0, len(audio) - 1 - audio_length_samples)
                #samples = audio[ind:ind + audio_length_samples]
                #melspec = librosa.feature.melspectrogram(samples, sr=Fs, n_fft=NFFT, hop_length=HOP_LENGTH, win_length=window_length_samples, n_mels=128)
                #melspec = librosa.power_to_db(melspec, ref=np.max)
                #ax = fig.add_subplot(111)
                #p = librosa.display.specshow(melspec, fmax=8000, sr=Fs, hop_length=HOP_LENGTH, ax=ax)
                #fig.savefig(output_dir + rf"\{c}.png")
                #ax.remove()
                #c += 1

    elif folder in target_words:
        pass
        #output_dir = fr"C:\Users\Harry\Desktop\Data\{folder}"
        #if not os.path.exists(output_dir):
            #os.makedirs(output_dir)

        #for file in os.listdir(r"D:\ML_Datasets\commands" + fr"\{folder}"):
            #with open(r"D:\ML_Datasets\commands" + fr"\{folder}" + fr"\{file}", "rb") as f:
                #samples, _ = librosa.load(f, sr=Fs)

            #if len(samples) < audio_length_samples:
                #amendments[folder] += 1
            #samples = librosa.util.fix_length(samples, audio_length_samples, mode='mean')
            #melspec = librosa.feature.melspectrogram(samples, sr=Fs, n_fft=NFFT, hop_length=HOP_LENGTH, win_length=window_length_samples, n_mels=128)
            #melspec = librosa.power_to_db(melspec, ref=np.max)
            #ax = fig.add_subplot(111)
            #p = librosa.display.specshow(melspec, fmax=8000, sr=Fs, hop_length=HOP_LENGTH, ax=ax)
            #fig.savefig(output_dir + rf"\{file[:-4]}.png")
            #ax.remove()
            #c += 1
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for file in os.listdir(r"D:\ML_Datasets\commands" + fr"\{folder}"):
            with open(r"D:\ML_Datasets\commands" + fr"\{folder}" + fr"\{file}", "rb") as f:
                samples, _ = librosa.load(f, sr=Fs)

            if len(samples) < audio_length_samples:
                amendments[folder] += 1
            samples = librosa.util.fix_length(samples, audio_length_samples, mode='mean')
            melspec = librosa.feature.melspectrogram(samples, sr=Fs, n_fft=NFFT, hop_length=HOP_LENGTH, win_length=window_length_samples, n_mels=128)
            melspec = librosa.power_to_db(melspec, ref=np.max)
            ax = fig.add_subplot(111)
            p = librosa.display.specshow(melspec, fmax=8000, sr=Fs, hop_length=HOP_LENGTH, ax=ax)
            fig.savefig(output_dir + rf"\{file[:-4]}.png")
            ax.remove()
            c += 1
    print(c)
    print(folder)
    print(amendments)

