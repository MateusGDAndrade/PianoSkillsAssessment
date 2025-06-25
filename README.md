# PianoSkillsAssessment


Este repositório contém a implementação de um pipeline multimodal para avaliação de habilidades de pianistas, baseado no *Multimodal PISA Dataset* (Papers with Code). Para esta primeira etapa do projeto, o objetivo é processar sinais de áudio, extrair espectrogramas, e treinar modelos de rede neural unimodais para classificar o nível de habilidade do pianista. Para a próxima entrega, será feito o treinamento de modelos de rede neural multimodais incluindo, dessa vez, os frames dos vídeos para a classificação o nível de habilidade do pianista. Por fim, o desempenho de ambas as abordagens serão comparados por meio das anotações das amostras. 

---
## Integrantes
Lucas Vizzotto de Castro 
Diego Cabral Morales 
Vitor Okubo
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
├── processed_data/            # Dados pré-processados em .pt (treinamento e teste)
├── preprocess.py              # Extrai áudio e salva .pt
├── dataloader_multimodal.py   # DataLoader multimodal, disponibilizado pelos autores do artigo
├── preprocessed_dataset.py    # Classe do Dataset com dados pré-processados
├── opts.py                    # Configurações globais (caminhos, seeds, dims)
├── t1_redes_neurais.ipynb     # Experimentos de treino e avaliação
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

### preprocess.py

* Carrega o dataset em modo `test` e `train` via `VideoDataset` (classe do Data Loader multimodal)
* Itera com `DataLoader(batch_size=1)`
* Salva cada amostra (áudio + label) em `processed_data/test/sample_{i}.pt`

---

## 📚 Data Loaders

### dataloader\_multimodal.py

* Classe `VideoDataset(Dataset)`:

  * Lê anotações em `.pkl` (áudio + frames de vídeo)
  * Converte segmentos de áudio em melspectrogramas (`librosa`)
  * Normaliza e converte em tensor 1×224×224

### preprocessed\_dataset.py

* Classe `PreprocessedDataset(Dataset)`:

  * Carrega arquivos `.pt` gerados por `preprocess.py`
  * Rápido carregamento para treino/validação

Uso no script ou notebook:

```python
from preprocessed_dataset import PreprocessedDataset
from torch.utils.data import DataLoader
train_dataset = PreprocessedDataset("./processed_data/train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```