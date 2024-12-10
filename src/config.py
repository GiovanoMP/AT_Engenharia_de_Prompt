import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurações da API
API_BASE_URL = 'https://dadosabertos.camara.leg.br/api/v2/'
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Configurações de data
DATA_INICIO = '2024-08-01'
DATA_FIM = '2024-08-30'

# Configurações de processamento
TEMAS_PROPOSICOES = [40, 46, 62]  # Economia, Educação, Ciência/Tecnologia
MAX_PROPOSICOES_POR_TEMA = 10

# Caminhos dos arquivos
DATA_DIR = 'data'
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')

# Garantir que os diretórios existam
for dir_path in [RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, IMAGES_DIR]:
    os.makedirs(dir_path, exist_ok=True)