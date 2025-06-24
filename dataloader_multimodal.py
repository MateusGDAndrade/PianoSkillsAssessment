# Author: Paritosh Parmar (https://github.com/ParitoshParmar)
'''
Code used in the following. Also if you find our work useful, please consider citing the following:

@article{parmar2021piano,
  title={Piano Skills Assessment},
  author={Parmar, Paritosh and Reddy, Jaiden and Morris, Brendan},
  journal={arXiv preprint arXiv:2101.04884},
  year={2021}
}
'''

# --- Importação das Bibliotecas Necessárias ---
import os  # Para interagir com o sistema operacional (lidar com arquivos e pastas)
import numpy as np  # Para computação numérica eficiente (arrays, matrizes)
import random  # Para gerar números aleatórios (usado na data augmentation)
import torch  # A biblioteca principal de deep learning (PyTorch)
from torch.utils.data import Dataset  # A classe base para criar datasets customizados no PyTorch
from torchvision import transforms  # Funções para transformar imagens (converter para tensor, normalizar, etc.)
import glob  # Para encontrar arquivos que correspondem a um padrão (ex: todos os .jpg em uma pasta)
from PIL import Image  # Para abrir, manipular e salvar imagens (Python Imaging Library)
from opts import * # Importa todas as variáveis de um arquivo de configuração chamado 'opts.py'
import pickle as pkl  # Para carregar arquivos .pkl (usados para salvar objetos Python, como dicionários)
import librosa  # Biblioteca especializada em análise de áudio e música


# --- Configuração de Sementes para Reprodutibilidade ---
# Fixar as sementes garante que os resultados "aleatórios" (como a inicialização de pesos)
# sejam os mesmos toda vez que o código rodar, o que é crucial para depuração e comparação de experimentos.
torch.manual_seed(random_seed); torch.cuda.manual_seed_all(random_seed); random.seed(random_seed); np.random.seed(random_seed)
torch.backends.cudnn.deterministic=True


# --- Funções Auxiliares ---

# Função para carregar e transformar uma única imagem de frame de vídeo
# def load_image(image_path, hori_flip, transform=None):
#     image = Image.open(image_path)  # Abre o arquivo de imagem
#     image = image.resize(c3d_input_resize, Image.BILINEAR)  # Redimensiona a imagem para o tamanho esperado pelo modelo
#     if hori_flip:  # Se a data augmentation de espelhamento horizontal estiver ativa
#         image.transpose(Image.FLIP_LEFT_RIGHT)  # Espelha a imagem
#     if transform is not None:  # Se uma sequência de transformações for fornecida
#         image = transform(image).unsqueeze(0)  # Aplica as transformações (ex: ToTensor, Normalize) e adiciona uma dimensão de batch
#     return image


# Função para converter um espectrograma (array numpy) em uma imagem (tensor do PyTorch)
def spec_to_image(spec, transform, eps=1e-6):
    mean = spec.mean()  # Calcula a média dos valores do espectrograma
    std = spec.std()  # Calcula o desvio padrão
    spec_norm = (spec - mean) / (std + eps)  # Normaliza o espectrograma (z-score), subtraindo a média e dividindo pelo desvio padrão
    spec_min, spec_max = spec_norm.min(), spec_norm.max()  # Encontra os valores mínimo e máximo do espectrograma normalizado
    
    if (spec_max - spec_min) == 0: # Evita divisão por zero se o espectrograma for "plano" (ex: silêncio)
        spec_scaled = 255 * (spec_norm - spec_min) / 5
    else:
        # Escala os valores normalizados para o intervalo de 0 a 255, para que possam ser representados como uma imagem em escala de cinza
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    
    spec_scaled = spec_scaled.astype(np.uint8)  # Converte os valores para inteiros de 8 bits (o formato de uma imagem)
    
    # Garante que a largura da imagem do espectrograma seja a esperada, cortando se necessário
    if spec_scaled.shape[1] != audio_img_W:
        spec_scaled = spec_scaled[:,:audio_img_W]
    
    true_image = Image.fromarray(spec_scaled)  # Converte o array numpy em um objeto de imagem PIL
    true_image = true_image.resize((224, 224), Image.BILINEAR)  # Redimensiona a imagem para 224x224, o tamanho esperado pela ResNet
    true_image = transform(true_image).unsqueeze(0)  # Aplica a transformação final (geralmente ToTensor) e adiciona a dimensão de batch
    return true_image


# Função para extrair um melspectrograma em escala de decibéis (dB) de um trecho de áudio
def get_melspectrogram_db(wav, sr, start_fr, end_fr, framerate, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    # Calcula o ponto de início e fim no áudio (em amostras) com base nos frames de vídeo
    start_time = int((start_fr/framerate)*sr)
    end_time = int((end_fr/framerate)*sr)
    wav = wav[start_time:end_time]  # Corta o array de áudio para pegar apenas o segmento desejado

    # Gera o melspectrograma usando librosa. `n_mels` define a resolução na frequência.
    spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
    # Converte a escala de potência do espectrograma para decibéis (dB), que é mais próxima da percepção humana de volume
    spec_db = librosa.power_to_db(spec,top_db=top_db)
    return spec_db


# --- Classe Principal do Dataset ---
# Esta classe herda de `torch.utils.data.Dataset` e define como carregar e pré-processar os dados
class VideoDataset(Dataset):
    # O método de inicialização, executado uma vez quando a classe é criada
    def __init__(self, mode):
        super(VideoDataset, self).__init__()
        self.sampling_scheme = sampling_scheme  # Define o esquema de amostragem (ex: 'uniform')
        self.mode = mode  # Define o modo ('training' ou 'test')

        # Abre o arquivo de anotações .pkl, que contém um dicionário com todos os metadados das amostras
        with open(anno_r2u_dir + '/annotations_' + self.sampling_scheme + '_' + self.mode + '.pkl', 'rb') as fp:
            self.set = pkl.load(fp)

        # Guarda as chaves do dicionário, que serão usadas para acessar cada amostra
        self.keys = list(self.set.keys())

    # O método principal, que define como obter UMA amostra do dataset dado um índice (ix)
    def __getitem__(self, ix):
        # Define as transformações para as imagens de vídeo (não será usado no seu caso)
        # if backbone == 'R18-3D':
        #     transform = transforms.Compose([...])
        # elif backbone == 'C3D' or backbone == 'Dilated_C3D':
        #     transform = transforms.Compose([...])
        # else:
        #     input('Error: Unknown Backbone! What you want to do?')

        # Define a transformação para a imagem do espectrograma (apenas converte para tensor)
        transform_audio_img = transforms.Compose([transforms.ToTensor()])

        # Pega a amostra específica do dicionário de anotações usando a chave correspondente ao índice
        sample = self.set[self.keys[ix]]
        sample_video = self.keys[ix][0]  # O ID do vídeo
        sample_frames = sample['frames']  # A lista de frames a serem usados para esta amostra
        sample_player_lvl = sample['player_level']  # A etiqueta (nível de habilidade do músico)
        sample_song_lvl = sample['song_level'] # A etiqueta de dificuldade da música
        sample_framerate = sample['framerate']  # A taxa de frames do vídeo original
        
        # Lógica para o vídeo (pode ser ignorada para o seu caso de áudio apenas)
        # if with_modality_video:
            # ... (código para carregar e processar frames de vídeo)

        # Lógica para o áudio (a parte mais importante para você)
        if with_modality_audio:
            # Reorganiza a lista de frames em clipes
            sample_frames_reshaped = sample_frames.reshape((nclips, clip_len))

            # Constrói o caminho para o arquivo de áudio .wav
            audio_file = dataset_audio_dir + str(sample_video) + '.wav'
            # Carrega o arquivo de áudio COMPLETO uma única vez usando librosa
            wav, sr = librosa.load(audio_file, sr=None)
            if sr != 44100:
                print('Different sr!')

            # Inicializa um tensor vazio para guardar as imagens de espectrograma de todos os clipes
            audio_imgs = torch.zeros(nclips, audio_img_C, audio_img_H, audio_img_W)

            # Loop para processar cada clipe separadamente
            for clip in range(nclips):
                start_fr = sample_frames_reshaped[clip][0]  # Pega o frame inicial do clipe
                end_fr = sample_frames_reshaped[clip][-1]  # Pega o frame final do clipe
                
                # Gera o melspectrograma apenas para este trecho do áudio
                mel_spec = get_melspectrogram_db(wav, sr=sr, start_fr=start_fr,
                                                 end_fr=end_fr, framerate=sample_framerate)
                
                # Converte o espectrograma em um tensor de imagem pronto para o modelo
                audio_imgs[clip] = spec_to_image(mel_spec, transform_audio_img)
        
        # Monta o dicionário de dados que será retornado
        data = {}
        # if with_modality_video:
        #     data['video'] = video
        if with_modality_audio:
            data['audio'] = audio_imgs  # Adiciona os tensores de espectrograma
        data['player_lvl'] = sample_player_lvl  # Adiciona a etiqueta
        return data

    # Método que retorna o número total de amostras no dataset
    def __len__(self):
        return len(self.set)