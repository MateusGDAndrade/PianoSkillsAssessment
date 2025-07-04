# Variáveis de configuração

dataset_audio_dir = "./audio_samples/"
anno_r2u_dir = "./annotations/" # Diretório de anotações em .pkl
with_modality_audio = True # Na primeira fase do projeto apenas apenas o áudio é analisado
with_modality_video = False

# Define a semente para as bibliotecas de aleatoriedade (torch, numpy, random)
# garantindo que os resultados do treinamento sejam reprodutíveis.
# Pode ser definido com um valor arbitrário.
random_seed = 20

sampling_scheme = "unidist"  # unidist para 'uniformly distributed'
nclips = 10 # Número de clipes por amostra
clip_len = 16 # Frames por clipe

audio_img_C = 1   # Canais (1 para escala de cinza)
audio_img_H = 224 # Altura
audio_img_W = 224 # Largura