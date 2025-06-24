# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 14:52:14 2025

@author: Mateus
"""

import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import os
from PIL import Image
import pandas as pd


TARGET_SIZE = (224, 224)  
OUTPUT_FOLDER = "./preprocessed_audio_samples" #path pra salvar os dados

audio_folder = "./audio_samples" # path dos arquivos
audio_files = [os.path.join(audio_folder, file) for file in os.listdir(audio_folder) if file.endswith('.wav')]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

data_dict = {}

def preprocess_and_save_spectrogram(file_path, sr=44100, duration=5):
    sample_rate, audio = wav.read(file_path)

    # Converte para mono se precisar
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Gera espectrograma
    frequencies, times, Sxx = signal.spectrogram(audio, fs=sr, nperseg=1024)
    Sxx_log = np.log10(Sxx + 1e-10)  # Escala logarítmica

    # Normaliza o espectrograma para 0-255 (escala de imagem)
    Sxx_min, Sxx_max = Sxx_log.min(), Sxx_log.max()
    Sxx_scaled = 255 * (Sxx_log - Sxx_min) / (Sxx_max - Sxx_min)
    Sxx_scaled = Sxx_scaled.astype(np.uint8)

    # Converte para imagem
    image = Image.fromarray(Sxx_scaled)
    
    # Canal único (grayscale)
    image = image.convert("L")  

    # Redimensiona para 224x224
    image = image.resize(TARGET_SIZE)

    
    filename = os.path.splitext(os.path.basename(file_path))[0]

    image_path = os.path.join(OUTPUT_FOLDER, f'{filename}.png')
    image.save(image_path)

    # Converte imagem para array numpy
    image_array = np.array(image)

    return filename, image_array


# Loop para processar todos os arquivos
for file in audio_files:
    file_id, array = preprocess_and_save_spectrogram(file)

    data_dict[file_id] = array