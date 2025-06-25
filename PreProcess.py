# Script para pré-processar o conjunto de teste: carrega cada amostra de áudio via VideoDataset e salva em .pt
import os
import torch
from dataloader_multimodal import VideoDataset
from torch.utils.data import DataLoader

# Crie o dataset e o loader originais
original_dataset = VideoDataset(mode='test')
# Use batch_size=1 para processar uma amostra por vez
original_loader = DataLoader(original_dataset, batch_size=1, shuffle=False)

# Loop para salvar tudo
for i, batch_data in enumerate(original_loader):
    audio_tensor = batch_data['audio']
    label = batch_data['player_lvl']

    # Crie um dicionário para salvar
    data_to_save = {'audio': audio_tensor.squeeze(0), 'label': label.item()}

    # Salve o dicionário em um arquivo .pt
    torch.save(data_to_save, f"./processed_data/test/sample_{i}.pt")

    print(f"Salvando amostra {i+1}/{len(original_loader)}")