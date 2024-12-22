"""
Módulo responsável pelo gerenciamento e persistência de embeddings
"""

import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle
import os

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Gerencia a geração, armazenamento e recuperação de embeddings"""
    
    def __init__(self, name: str, model_name: str = "neuralmind/bert-base-portuguese-cased"):
        """
        Inicializa o gerenciador de embeddings
        
        Args:
            name: Nome identificador para este conjunto de embeddings
            model_name: Nome do modelo BERT a ser usado
        """
        self.name = name
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.index = None
        self.data = []
        
        # Diretórios para armazenamento
        self.base_dir = Path("data/embeddings")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.base_dir / f"{name}_index.faiss"
        self.data_path = self.base_dir / f"{name}_data.pkl"
        self.embeddings_path = self.base_dir / f"{name}_embeddings.npy"
        
    def initialize_bert(self):
        """Inicializa o modelo BERT"""
        if self.tokenizer is None or self.model is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
                self.model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
                self.model.eval()  # Modo de avaliação
                logger.info("Modelo BERT inicializado com sucesso")
            except Exception as e:
                logger.error(f"Erro ao inicializar BERT: {str(e)}")
                raise
                
    def generate_embedding(self, text: str) -> np.ndarray:
        """Gera embedding para um texto usando BERT"""
        try:
            self.initialize_bert()
            
            # Tokenização
            inputs = self.tokenizer(text, return_tensors="pt", 
                                  max_length=512, truncation=True, padding=True)
            
            # Gera embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Usa o embedding do token [CLS] como representação do texto
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
            
            return embedding[0]  # Retorna o embedding como array 1D
            
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {str(e)}")
            raise
            
    def load_or_create_index(self, dimension: int) -> faiss.Index:
        """Carrega índice existente ou cria um novo"""
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Índice carregado de {self.index_path}")
                return self.index
            except Exception as e:
                logger.warning(f"Erro ao carregar índice: {e}. Criando novo...")
                
        self.index = faiss.IndexFlatL2(dimension)
        return self.index
        
    def process_data(self, data: List[Dict[str, Any]], text_fields: List[str]):
        """Processa dados para criar embeddings"""
        try:
            # Concatena os campos de texto
            texts = []
            for item in data:
                text = " ".join([str(item[field]) for field in text_fields if field in item])
                texts.append(text)
            
            # Gera embeddings
            embeddings = []
            for text in texts:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
            
            # Converte para array numpy
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Cria/atualiza índice FAISS
            dimension = embeddings_array.shape[1]
            self.index = self.load_or_create_index(dimension)
            self.index.add(embeddings_array)
            
            # Salva índice
            faiss.write_index(self.index, str(self.index_path))
            logger.info(f"Processados {len(texts)} itens para {self.name}")
            
        except Exception as e:
            logger.error(f"Erro ao processar dados: {str(e)}")
            raise
            
    def load_saved_data(self) -> Optional[pd.DataFrame]:
        """Carrega dados salvos anteriormente"""
        try:
            if self.data_path.exists():
                with open(self.data_path, 'rb') as f:
                    self.data = pickle.load(f)
                logger.info(f"Dados carregados de {self.data_path}")
                return self.data
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
        return None
            
    def search_similar(self, query: str, k: int = 100) -> List[Dict[str, Any]]:
        """
        Busca itens similares à query
        
        Args:
            query: Texto para busca
            k: Número de resultados a retornar
        """
        try:
            # Carrega dados e índice se necessário
            if self.index is None:
                if not self.index_path.exists():
                    raise ValueError(f"Índice não encontrado em {self.index_path}")
                self.index = faiss.read_index(str(self.index_path))
                
            if len(self.data) == 0:
                self.load_saved_data()
                
            # Gera embedding para a query
            query_embedding = self.generate_embedding(query)
            
            # Busca os k vizinhos mais próximos
            D, I = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                k
            )
            
            # Prepara resultados
            results = []
            for i, (dist, idx) in enumerate(zip(D[0], I[0])):
                if idx < len(self.data):
                    item = self.data.iloc[idx].to_dict()
                    item['similarity_score'] = float(dist)
                    results.append(item)
                    
            return results
            
        except Exception as e:
            logger.error(f"Erro na busca: {str(e)}")
            return []
