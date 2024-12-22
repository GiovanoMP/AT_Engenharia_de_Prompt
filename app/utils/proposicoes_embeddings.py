"""
Módulo responsável pelo gerenciamento de embeddings das proposições
"""

import logging
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from .embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

class ProposicoesEmbeddings:
    """Gerencia os embeddings das proposições"""
    
    def __init__(self):
        """Inicializa o gerenciador de embeddings"""
        self.manager = EmbeddingManager("proposicoes")
        self.load_data()
        
    def load_data(self):
        """Carrega ou processa os dados das proposições"""
        try:
            # Tenta carregar dados existentes
            data = self.manager.load_saved_data()
            
            # Se não existir, processa novos dados
            if data is None:
                data_path = Path("data/processed/proposicoes_deputados.parquet")
                if not data_path.exists():
                    raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")
                    
                data = pd.read_parquet(data_path)
                
                # Campos relevantes para embeddings
                text_fields = ['ementa', 'keywords', 'tema']
                
                # Processa e salva embeddings
                self.manager.process_data(data, text_fields)
                
            logger.info("Dados das proposições carregados com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados das proposições: {str(e)}")
            raise
            
    def search_similar(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Busca proposições similares à query"""
        return self.manager.search_similar(query, k)
