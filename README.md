# PianoSkillsAssessment


Este repositório contém a implementação de um pipeline multimodal para avaliação de habilidades de pianistas, baseado no *Multimodal PISA Dataset* (Papers with Code). Para esta primeira etapa do projeto, o objetivo é processar sinais de áudio, extrair espectrogramas, e treinar modelos de rede neural unimodais para classificar o nível de habilidade do pianista. Para a próxima entrega, será feito o treinamento de modelos de rede neural multimodais incluindo agora os frames dos vídeos para a classificação o nível de habilidade do pianista. Por fim, o desempenho de ambas as abordagens serão comparados por meio das anoteções das amostras. 

---
## Integrantes
Lucas Vizzotto de Castro 
Diego Cabral Morales 
Vitor
Jean Michel Furtado M'Peko 
Gabriela Barros 
Mateus Gentil Dantas de Andrade 
Pedro Henrique Gonçalez Moracci 

## 🔍 Visão Geral do Projeto

1. **Dataset**: Multimodal PISA Dataset (áudio + vídeo) disponível em [https://paperswithcode.com/dataset/multimodal-pisa](https://paperswithcode.com/dataset/multimodal-pisa)
2. **Modalidade Principal**: Áudio (melspectrogramas)
3. **Fluxo**:

   * Pré-processamento bruto de áudio em arquivos `.pt`
   * Geração de espectrogramas em `.png` para análise visual
   * DataLoader multimodal para conversão em tensores prontos para treino
   * Dataset simplificado que carrega diretamente arquivos pré-processados
   * Treinamento e avaliação de rede neural 

---

## 📦 Estrutura de Diretórios

```
PianoSkillsAssessment/
├── audio_samples/             # Arquivos .wav originais
├── annotations/               # Anotações multimodais em .pkl
├── processed_data/            # Dados pré-processados em .pt (test)
├── preprocessed_audio_samples/# Imagens .png dos espectrogramas
├── PreProcess.py              # Extrai áudio e salva .pt
├── PreProcess2.py             # Gera e salva espectrogramas .png
├── dataloader_multimodal.py   # DataLoader multimodal
├── preprocessed_dataset.py    # Dataset rápido de arquivos .pt
├── opts.py                    # Configurações globais (caminhos, seeds, dims)
└── README.md                  # Documentação do projeto
```


## ⚙️ Configuração

1. **Defina os caminhos** em `opts.py`:

   * `dataset_audio_dir`: local dos arquivos `.wav`
   * `anno_r2u_dir`: diretório com arquivos `.pkl` de anotações
2. **Escolha a semente** (`random_seed`) e parâmetros de amostragem:

   * `nclips`, `clip_len`, dimensões de imagem (`audio_img_C/H/W`)

---

## 🔄 Pré-processamento

### 1. PreProcess.py

* Carrega o dataset em modo `test` via `VideoDataset`
* Itera com `DataLoader(batch_size=1)`
* Salva cada amostra (áudio + label) em `processed_data/test/sample_{i}.pt`

### 2. PreProcess2.py

* Lê arquivos `.wav` de `audio_samples/`
* Gera espectrogramas via `scipy.signal.spectrogram`
* Converte para escala logarítmica, normaliza, salva como imagem `.png`
* Armazena em `preprocessed_audio_samples/`


---

## 📚 Data Loaders

### dataloader\_multimodal.py

* Classe `VideoDataset(Dataset)`:

  * Lê anotações em `.pkl` (áudio + frames de vídeo)
  * Converte segmentos de áudio em melspectrogramas (`librosa`)
  * Normaliza e converte em tensor 1×224×224

### preprocessed\_dataset.py

* Classe `PreprocessedDataset(Dataset)`:

  * Carrega arquivos `.pt` gerados por `PreProcess.py`

# PianoSkillsAssessment

Este repositório contém a implementação de um pipeline multimodal para avaliação de habilidades pianísticas, baseado no *Multimodal PISA Dataset* (Papers with Code). O objetivo é processar sinais de áudio (e potencialmente vídeo), extrair espectrogramas, e treinar modelos de rede neural para classificar o nível de habilidade do pianista.

---

## 🔍 Visão Geral do Projeto

1. **Dataset**: Multimodal PISA Dataset (áudio + vídeo) disponível em [https://paperswithcode.com/dataset/multimodal-pisa](https://paperswithcode.com/dataset/multimodal-pisa)
2. **Modalidade Principal**: Áudio (melspectrogramas)
3. **Fluxo**:

   * Pré-processamento bruto de áudio em arquivos `.pt`
   * Geração de espectrogramas em `.png` para análise visual
   * DataLoader multimodal para conversão em tensores prontos para treino
   * Dataset simplificado que carrega diretamente arquivos pré-processados
   * Treinamento e avaliação de rede neural via Jupyter Notebook (`t1_redes_neurais.ipynb`) e script `train.py`

---

## 📦 Estrutura de Diretórios

```
PianoSkillsAssessment/
├── audio_samples/               # Arquivos .wav originais
├── annotations/                 # Anotações multimodais em .pkl
├── processed_data/              # Dados pré-processados em .pt (test)
├── preprocessed_audio_samples/  # Imagens .png dos espectrogramas
├── notebooks/                   # Notebooks de análise e experimentos
│   └── t1_redes_neurais.ipynb   # Experimentos de treino e avaliação
├── PreProcess.py                # Extrai áudio e salva .pt
├── PreProcess2.py               # Gera e salva espectrogramas .png
├── dataloader_multimodal.py     # DataLoader customizado multimodal
├── preprocessed_dataset.py      # Dataset rápido de arquivos .pt
├── opts.py                      # Configurações globais (caminhos, seeds, dims)
├── train.py                     # Pipeline de treino e avaliação (script)
├── requirements.txt             # Dependências do projeto
└── README.md                    # Documentação do projeto
```

---

## 🛠️ Requisitos

* Python 3.8 ou superior
* Bibliotecas:

  * `torch`, `torchvision`
  * `numpy`, `scipy`, `librosa`
  * `Pillow`, `pandas`
  * `jupyter`, `matplotlib` para visualização

Instale com:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuração

1. **Defina os caminhos** em `opts.py`:

   * `dataset_audio_dir`: local dos arquivos `.wav`
   * `anno_r2u_dir`: diretório com arquivos `.pkl` de anotações
2. **Escolha a semente** (`random_seed`) e parâmetros de amostragem:

   * `nclips`, `clip_len`, dimensões de imagem (`audio_img_C/H/W`)

---

## 🔄 Pré-processamento

### 1. PreProcess.py

* Carrega o dataset em modo `test` via `VideoDataset`
* Itera com `DataLoader(batch_size=1)`
* Salva cada amostra (áudio + label) em `processed_data/test/sample_{i}.pt`

Comando:

```bash
python PreProcess.py
```

### 2. PreProcess2.py

* Lê arquivos `.wav` de `audio_samples/`
* Gera espectrogramas via `scipy.signal.spectrogram`
* Converte para escala logarítmica, normaliza, salva como imagem `.png`
* Armazena em `preprocessed_audio_samples/`

Comando:

```bash
python PreProcess2.py
```

---

## 📚 Data Loaders

### dataloader\_multimodal.py

* Classe `VideoDataset(Dataset)`:

  * Lê anotações em `.pkl` (áudio + frames de vídeo)
  * Converte segmentos de áudio em melspectrogramas (`librosa`)
  * Normaliza e converte em tensor 1×224×224

### preprocessed\_dataset.py

* Classe `PreprocessedDataset(Dataset)`:

  * Carrega arquivos `.pt` gerados por `PreProcess.py`
  * Rápido carregamento para treino/validação

Uso no script ou notebook:

```python
from preprocessed_dataset import PreprocessedDataset
from torch.utils.data import DataLoader
train_dataset = PreprocessedDataset("./processed_data/train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

---

## 📓 Notebook `t1_redes_neurais.ipynb`

Este notebook reúne:

1. **Exploração de Dados**:

   * Visualização de espectrogramas
   * Análise de distribuição de labels (níveis de habilidade)
2. **Definição de Modelo**:

   * Arquitetura CNN simples aplicada a espectrogramas
   * Critério de perda e otimizador configurados via `opts.py`
3. **Loop de Treino e Validação**:

   * Funções para treino por época e avaliação em conjunto de validação
   * Registro de métricas (loss, acurácia) e geração de gráficos
4. **Teste e Resultados**:

   * Predições finais sobre conjunto de teste
   * Matriz de confusão e relatório de classificação
5. **Salvamento de Checkpoints**:

   * Arquivos `.pt` com pesos do modelo em `./checkpoints/`

