"""
Gera embeddings para todos os dados usando BERT e FAISS
"""

import pandas as pd
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
    
    # Lista todos os arquivos que vamos processar
    logger.info("\nArquivos que serão processados:")
    logger.info("1. data/processed/deputados.parquet")
    logger.info("2. data/processed/proposicoes_deputados.parquet")
    logger.info("3. data/processed/serie_despesas_diarias_deputados.parquet")
    logger.info("4. data/processed/insights_despesas_deputados.json")
    logger.info("5. data/processed/insights_distribuicao_deputados.json")
    logger.info("6. data/processed/sumarizacao_proposicoes.json")
    
    # Processa deputados
    logger.info("\nProcessando deputados...")
    deputados = pd.read_parquet("data/processed/deputados.parquet")
    textos_deputados = [
        f"Deputado {row['nome']} do partido {row['siglaPartido']} " + 
        f"representando {row['siglaUf']}"
        for _, row in deputados.iterrows()
    ]
    logger.info(f"Total de deputados: {len(textos_deputados)}")
    manager.criar_index(textos_deputados)
    save_embeddings("deputados", textos_deputados, 
                   manager.embeddings, manager.index)
    
    # Processa proposições
    logger.info("\nProcessando proposições...")
    proposicoes = pd.read_parquet("data/processed/proposicoes_deputados.parquet")
    textos_proposicoes = [
        f"Proposição: {row['ementa']}"
        for _, row in proposicoes.iterrows()
    ]
    logger.info(f"Total de proposições: {len(textos_proposicoes)}")
    manager.criar_index(textos_proposicoes)
    save_embeddings("proposicoes", textos_proposicoes, 
                   manager.embeddings, manager.index)
    
    # Processa despesas (em batches)
    logger.info("\nProcessando despesas...")
    despesas = pd.read_parquet("data/processed/serie_despesas_diarias_deputados.parquet")
    logger.info("Colunas disponíveis:")
    logger.info(despesas.columns.tolist())
    
    textos_despesas = []
    batch_size = 1000
    total_batches = (len(despesas) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="Processando despesas em batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(despesas))
        batch_despesas = despesas.iloc[start_idx:end_idx]
        
        batch_textos = [
            f"Despesa de {row['nomeDeputado']}: {row['tipoDespesa']} " +
            f"no valor de R$ {row['valorDocumento']:.2f}"
            for _, row in batch_despesas.iterrows()
        ]
        textos_despesas.extend(batch_textos)
    
    logger.info(f"Total de despesas: {len(textos_despesas)}")
    manager.criar_index(textos_despesas)
    save_embeddings("despesas", textos_despesas, 
                   manager.embeddings, manager.index)
    
    # Processa insights de despesas
    logger.info("\nProcessando insights de despesas...")
    insights_despesas = load_json("data/processed/insights_despesas_deputados.json")
    textos_insights_despesas = [
        f"Insight despesa: {insight}" 
        for insight in insights_despesas
    ]
    logger.info(f"Total de insights de despesas: {len(textos_insights_despesas)}")
    manager.criar_index(textos_insights_despesas)
    save_embeddings("insights_despesas", textos_insights_despesas,
                   manager.embeddings, manager.index)
    
    # Processa insights de distribuição
    logger.info("\nProcessando insights de distribuição...")
    insights_dist = load_json("data/processed/insights_distribuicao_deputados.json")
    textos_insights_dist = [
        f"Insight distribuição: {insight}"
        for insight in insights_dist
    ]
    logger.info(f"Total de insights de distribuição: {len(textos_insights_dist)}")
    manager.criar_index(textos_insights_dist)
    save_embeddings("insights_distribuicao", textos_insights_dist,
                   manager.embeddings, manager.index)
    
    # Processa sumarizações
    logger.info("\nProcessando sumarizações...")
    sumarizacoes = load_json("data/processed/sumarizacao_proposicoes.json")
    textos_sumarizacoes = [
        f"Sumarização do tema {sum['tema']}: {sum['sumarizacao']}"
        for sum in sumarizacoes['sumarizacoes_por_tema']
    ]
    logger.info(f"Total de sumarizações: {len(textos_sumarizacoes)}")
    manager.criar_index(textos_sumarizacoes)
    save_embeddings("sumarizacoes", textos_sumarizacoes,
                   manager.embeddings, manager.index)
    
    logger.info("\nTodos os embeddings foram gerados com sucesso!")
    logger.info("Usando modelo: neuralmind/bert-base-portuguese-cased")
    logger.info("Arquivos gerados em: data/embeddings/")

if __name__ == "__main__":
    main()
