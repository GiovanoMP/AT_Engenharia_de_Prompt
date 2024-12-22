"""
Módulo responsável pelo gerenciamento de embeddings dos deputados
"""

import logging
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from .embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

class DeputadosEmbeddings:
    """Gerencia os embeddings dos deputados"""
    
    def __init__(self):
        """Inicializa o gerenciador de embeddings"""
        self.manager = EmbeddingManager("deputados")
        self.load_data()
        
    def load_data(self):
        """Carrega ou processa os dados dos deputados"""
        try:
            # Tenta carregar dados existentes
            data = self.manager.load_saved_data()
            
            # Se não existir, processa novos dados
            if data is None:
                data_path = Path("data/processed/deputados.parquet")
                if not data_path.exists():
                    raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")
                    
                data = pd.read_parquet(data_path)
                
                # Usa todos os campos relevantes
                text_fields = data.columns.tolist()
                
                # Remove campos não textuais
                text_fields = [f for f in text_fields if data[f].dtype == 'object']
                
                # Processa e salva embeddings
                self.manager.process_data(data, text_fields)
                
            logger.info("Dados dos deputados carregados com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados dos deputados: {str(e)}")
            raise
            
    def search_similar(self, query: str, k: int = 100) -> List[Dict[str, Any]]:
        """Busca deputados similares à query"""
        return self.manager.search_similar(query, k)
