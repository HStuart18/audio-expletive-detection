import soundfile as sf
import numpy as np
import os
import shutil
import librosa 
import xml.etree.ElementTree as ET
from pydub import AudioSegment
from pydub.playback import play
from pydub.silence import split_on_silence
import random

MAX_EPOCHS = 100_000
fs = 16000

def random_ints_with_sum(n, num_elements):
    """
    Generate num_elements non-negative random integers summing to `n`.
    """
    if num_elements == 0:
        return None
    count = 0
    while n > 0:
        if count == num_elements - 1:
            break
        r = random.randint(0, int(n / (num_elements - count)))
        yield r
        n -= r
        count += 1
    yield n

def choose_background_file():
    """
    Returns a random background audio file.
    """
    folders = next(os.walk(r"D:\ML_Datasets\demand"))[1]
    n = len(folders)
    folder = folders[random.randint(0, n - 1)]
    file = random.choice(os.listdir(rf"D:\ML_Datasets\demand\{folder}"))
    return r"D:\ML_Datasets\demand" + rf"\{folder}\{file}"

def choose_positive_files(max_samples):
    """
    Returns up to max_samples positive samples.
    """
    num_samples = random.randint(0, max_samples)
    duration = 0
    files = []
    for i in range(num_samples):
        file = random.choice(os.listdir(r"D:\ML_Datasets\commands\happy"))
        audio = AudioSegment.from_file(r"D:\ML_Datasets\commands\happy" + rf"\{file}")
        files.append([audio, len(audio), 1])
        duration += len(audio)
    # Ensure that the positive samples will not exceed the length of the background audio
    while duration >= 10000:
        files = files[:-1]
    return files

def choose_negative_files(max_samples, positive_duration):
    """
    Returns up to max_samples negative samples.
    """
    num_samples = random.randint(0, max_samples)
    duration = 0
    files = []
    for i in range(num_samples):
        audio = None
        # Handle failed mp3->wav conversion
        while audio is None:
            file = random.choice([f for f in os.listdir(r"D:\ML_Datasets\clips") if f.endswith(".wav")])
            try:
                with open(r"D:\ML_Datasets\clips" + rf"\{file}", "rb") as f:
                    audio = AudioSegment.from_file(f)
            except:
                os.remove(r"D:\ML_Datasets\clips" + rf"\{file}")
        files.append([audio, len(audio), -1])
        duration += len(audio)
    # Ensure that positive and negative samples do not exceed length of background audio
    while duration >= 10000 - positive_duration:
        duration -= files[-1][1]
        files = files[:-1]
    return files

background_audio = AudioSegment.from_file(choose_background_file()) 
ind = random.randint(0, len(background_audio) - 11000)
background_audio = background_audio[ind:ind + 10000]
for e in range(MAX_EPOCHS):
    # Generate 20 samples using the same random background audio clip
    if (e % 20 == 0):
        background_audio = AudioSegment.from_file(choose_background_file()) 
        ind = random.randint(0, len(background_audio) - 11000)
        background_audio = background_audio[ind:ind + 10000]
    
    # Congregate positive and negative samples
    positive_samples = choose_positive_files(3)
    negative_samples = choose_negative_files(4, sum([e[1] for e in positive_samples]))

    samples = positive_samples + negative_samples
    random.shuffle(samples)

    total_overlay_dur = sum([e[1] for e in samples])

    # Create random left paddings for each sample
    overlay_offsets = list(random_ints_with_sum(10000 - total_overlay_dur, len(samples)))

    duration = 0

    combined_audio = background_audio
    timestamps = []

    # Overlay samples with respective paddings onto background audio
    for i in range(len(samples)):
        combined_audio = combined_audio.overlay(samples[i][0], position=overlay_offsets[i] + duration)
        if samples[i][2] == 1:
            timestamps.append((overlay_offsets[i] + duration, overlay_offsets[i] + duration + samples[i][1]))
        duration = duration + samples[i][1] + overlay_offsets[i]

    # Export generated audio clip and associated positive sample alignment file
    combined_audio.export(rf"data\processed\audio\{e}.wav", format="wav")
    with open(rf"data\processed\timestamps\{e}.txt", "w+") as f:
        for i in range(len(timestamps)):
            f.write(fr"({timestamps[i][0]},{timestamps[i][1]})")
            f.write("\n")
    print(e)