"""
Gera embeddings apenas para o arquivo de sumarizações
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import json
from pathlib import Path
import numpy as np
import pickle
import faiss
from embeddings import EmbeddingManager
import logging
from tqdm import tqdm

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_json(file_path):
    """Carrega arquivo JSON"""
    logger.info(f"Carregando JSON: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_embeddings(name, data, embeddings, index):
    """Salva embeddings e dados relacionados"""
    base_path = Path("data/embeddings")
    base_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Salvando dados para: {name}")
    
    # Salva dados originais
    with open(base_path / f"{name}_data.pkl", "wb") as f:
        pickle.dump(data, f)
    
    # Salva embeddings
    np.save(str(base_path / f"{name}_embeddings.npy"), embeddings)
    
    # Salva índice FAISS
    faiss.write_index(index, str(base_path / f"{name}_index.faiss"))
    
    logger.info(f"Dados salvos com sucesso para: {name}")

def main():
    # Inicializa gerenciador
    logger.info("Inicializando EmbeddingManager com BERT português...")
    manager = EmbeddingManager()
    
    # Processa sumarizações
    logger.info("\nProcessando sumarizações...")
    sumarizacoes = load_json("data/processed/sumarizacao_proposicoes.json")
    
    textos_sumarizacoes = []
    for sum in tqdm(sumarizacoes['sumarizacoes_por_tema'], desc="Processando sumarizações"):
        texto = f"Sumarização do tema {sum['tema']}: {sum['sumarizacao']}"
        textos_sumarizacoes.append(texto)
    
    logger.info(f"Total de sumarizações: {len(textos_sumarizacoes)}")
    
    # Gera embeddings e índice
    logger.info("Gerando embeddings e índice FAISS...")
    manager.criar_index(textos_sumarizacoes)
    
    # Salva resultados
    save_embeddings("sumarizacoes", textos_sumarizacoes,
                   manager.embeddings, manager.index)
    
    logger.info("\nSumarizações processadas com sucesso!")
    logger.info("Usando modelo: neuralmind/bert-base-portuguese-cased")
    logger.info("Arquivos gerados em: data/embeddings/")

if __name__ == "__main__":
    main()
