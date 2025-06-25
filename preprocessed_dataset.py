import torch
from torch.utils.data import Dataset
import os
import glob

class PreprocessedDataset(Dataset):
    """
    Dataset customizado do PyTorch para carregar dados que já foram 
    pré-processados e salvos em arquivos .pt.
    
    Esta classe é muito mais rápida que o DataLoader original porque sua única
    tarefa é ler arquivos do disco, em vez de realizar cálculos pesados
    (como gerar espectrogramas) em tempo real, o que otimiza os laços realizados sobre os dados.
    """
    def __init__(self, data_dir):
        """
        Construtor da classe. Sua principal tarefa é encontrar e listar
        todos os arquivos de dados na pasta especificada.

        :param data_dir: O caminho para a pasta que contém os arquivos .pt.
                         Ex: '/processed_data/train/'
        """
        # cria um padrão para encontrar todos os arquivos que terminam com .pt
        # os.path.join garante que o caminho seja construído corretamente em qualquer sistema operacional.
        path_pattern = os.path.join(data_dir, '*.pt')
        
        # encontra todos os caminhos de arquivo que correspondem ao padrão e retorna uma lista com eles.
        self.file_paths = glob.glob(path_pattern)
        
        # verificação para garantir que os arquivos foram encontrados.
        if not self.file_paths:
            print(f"Aviso: Nenhum arquivo .pt foi encontrado no diretório: {data_dir}")
        else:
            print(f"Dataset encontrado. Número de amostras: {len(self.file_paths)}")

    def __len__(self):
        """
        Retorna o número total de amostras no dataset.
        O DataLoader usa este método para saber o tamanho do dataset.
        """
        # o tamanho do dataset é simplesmente o número de arquivos que encontramos.
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Carrega e retorna uma única amostra do dataset.
        O DataLoader chama este método para cada item que ele precisa pegar.

        :param idx: O índice da amostra a ser carregada (de 0 a len(dataset)-1).
        :return: Um dicionário contendo o tensor de áudio e sua label.
        """
        # 1. Pega o caminho do arquivo para o índice solicitado
        file_path = self.file_paths[idx]
        
        # 2. Carrega o arquivo .pt usando a função do PyTorch. Isso retorna o objeto que foi salvo, que no nosso caso é um dicionário.
        loaded_data = torch.load(file_path)
        
        # 3. Extrai o tensor de áudio e a label do dicionário carregado.
        audio_tensor = loaded_data['audio']
        label = loaded_data['label']
        
        # 4. Retorna um novo dicionário no formato esperado pelo loop de treinamento.
        return {'audio': audio_tensor, 'player_lvl': label}