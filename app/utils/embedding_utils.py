"""
Módulo responsável pela geração e gerenciamento de embeddings usando BERT
"""

import logging
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Any
from .data_loader import DataLoader

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, model_name: str = "neuralmind/bert-base-portuguese-cased"):
        """
        Inicializa o gerenciador de embeddings
        
        Args:
            model_name: Nome do modelo BERT a ser utilizado
        """
        logger.info(f"Inicializando EmbeddingManager com modelo {model_name}")
        self.model = SentenceTransformer(model_name)
        self.data_loader = DataLoader()
        self.index = None
        self.text_mapping = {}  # Mapeia IDs para textos originais

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Gera embeddings para uma lista de textos
        
        Args:
            texts: Lista de textos para gerar embeddings
            
        Returns:
            np.ndarray: Matriz de embeddings
        """
        logger.info(f"Gerando embeddings para {len(texts)} textos")
        # Gera embeddings em lotes para economizar memória
        embeddings = self.model.encode(
            texts,
            batch_size=32,  # Ajuste conforme sua memória disponível
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray):
        """
        Constrói índice FAISS para busca rápida
        
        Args:
            embeddings: Matriz de embeddings numpy
        """
        logger.info("Construindo índice FAISS")
        dimension = embeddings.shape[1]  # Dimensão dos vetores
        
        # Inicializa índice L2
        self.index = faiss.IndexFlatL2(dimension)
        
        # Adiciona os vetores ao índice
        self.index.add(embeddings.astype(np.float32))
        logger.info(f"Índice construído com {self.index.ntotal} vetores")

    def save_index(self, filepath: str):
        """
        Salva índice FAISS e mapeamento de textos em disco
        
        Args:
            filepath: Caminho para salvar o índice
        """
        if self.index is None:
            raise ValueError("Nenhum índice para salvar")
            
        filepath = Path(filepath)
        logger.info(f"Salvando índice em {filepath}")
        
        # Salva o índice FAISS
        faiss.write_index(self.index, str(filepath))
        
        # Salva o mapeamento de textos
        mapping_path = filepath.parent / f"{filepath.stem}_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.text_mapping, f, ensure_ascii=False, indent=2)
            
    def load_index(self, filepath: str):
        """
        Carrega índice FAISS e mapeamento de textos do disco
        
        Args:
            filepath: Caminho para carregar o índice
        """
        filepath = Path(filepath)
        logger.info(f"Carregando índice de {filepath}")
        
        # Carrega o índice FAISS
        self.index = faiss.read_index(str(filepath))
        
        # Carrega o mapeamento de textos
        mapping_path = filepath.parent / f"{filepath.stem}_mapping.json"
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.text_mapping = json.load(f)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Busca documentos similares usando o índice FAISS
        
        Args:
            query: Texto de consulta
            k: Número de resultados a retornar
            
        Returns:
            Lista de dicionários com resultados e scores
        """
        if self.index is None:
            raise ValueError("Índice não inicializado")
            
        # Gera embedding para a query
        query_embedding = self.model.encode([query])[0].reshape(1, -1)
        
        # Realiza a busca
        distances, indices = self.index.search(
            query_embedding.astype(np.float32), 
            k
        )
        
        # Formata os resultados
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in self.text_mapping:
                results.append({
                    'text': self.text_mapping[str(idx)],
                    'score': float(dist)
                })
                
        return results

    def process_data(self):
        """
        Processa todos os dados necessários e cria índices
        """
        logger.info("Iniciando processamento de dados")
        
        # Carrega dados
        self.data_loader.load_all_data()
        
        # Lista para armazenar todos os textos
        texts = []
        
        # Processa deputados
        deputados_df = self.data_loader.load_deputados()
        if deputados_df is not None:
            texts.extend(deputados_df['nome'].tolist())
        
        # Processa proposições
        proposicoes_df = self.data_loader.load_proposicoes()
        if proposicoes_df is not None:
            texts.extend(proposicoes_df['ementa'].tolist())
        
        # Cria mapeamento de textos
        self.text_mapping = {str(i): text for i, text in enumerate(texts)}
        
        # Gera embeddings e constrói índice
        embeddings = self.generate_embeddings(texts)
        self.build_faiss_index(embeddings)
        
        # Salva o índice
        index_path = Path("data/embeddings/faiss_index")
        self.save_index(index_path)
        
        logger.info("Processamento concluído")
