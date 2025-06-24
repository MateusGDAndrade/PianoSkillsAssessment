# --- Configurações de Caminho e Modalidade ---
dataset_audio_dir = "./audio_samples/"
anno_r2u_dir = "./annotations/"
with_modality_audio = True
with_modality_video = False

# --- Configurações de Reprodutibilidade ---
random_seed = 20

# --- Configurações de Amostragem de Dados ---
sampling_scheme = "unidist"  # unidist para 'uniformly distributed'
nclips = 10
clip_len = 16

# --- Configurações de Dimensão da Imagem de Áudio ---
audio_img_C = 1   # Canais (1 para escala de cinza)
audio_img_H = 224 # Altura
audio_img_W = 224 # Largura