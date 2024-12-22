"""
Módulo responsável pelo gerenciamento de embeddings das despesas
"""

import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class DespesasEmbeddings:
    """Gerencia os embeddings das despesas usando BERT"""
    
    def __init__(self):
        """Inicializa o gerenciador de embeddings"""
        self.model_name = "neuralmind/bert-base-portuguese-cased"
        self.tokenizer = None
        self.model = None
        self.index = None
        self.despesas_data = []
        
    def initialize_bert(self):
        """Inicializa o modelo BERT e tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            logger.info("Modelo BERT inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar BERT: {str(e)}")
            raise
            
    def generate_embedding(self, text: str) -> np.ndarray:
        """Gera embedding para um texto usando BERT"""
        try:
            # Tokeniza e prepara o input
            inputs = self.tokenizer(text, return_tensors="pt", 
                                  max_length=512, truncation=True, padding=True)
            
            # Gera o embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Usa a média do último hidden state como embedding
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            return embeddings[0]
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {str(e)}")
            raise
            
    def process_despesas(self, despesas: List[Dict[str, Any]]):
        """Processa as despesas e cria índice FAISS"""
        try:
            if not self.tokenizer or not self.model:
                self.initialize_bert()
                
            # Converte para DataFrame se não for
            if not isinstance(despesas, pd.DataFrame):
                despesas = pd.DataFrame(despesas)
                
            # Prepara os dados
            self.despesas_data = despesas
            embeddings = []
            
            # Gera embeddings para cada despesa
            for _, desp in despesas.iterrows():
                # Combina informações relevantes da despesa
                texto = f"{desp.get('tipoDespesa', '')} {desp.get('dataDocumento', '')} {desp.get('nomeFornecedor', '')} {desp.get('valorDocumento', '')}"
                embedding = self.generate_embedding(texto)
                embeddings.append(embedding)
                
            # Cria índice FAISS
            embeddings_array = np.array(embeddings).astype('float32')
            dimension = embeddings_array.shape[1]
            
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings_array)
            
            logger.info(f"Processadas {len(despesas)} despesas")
            
        except Exception as e:
            logger.error(f"Erro ao processar despesas: {str(e)}")
            raise
            
    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Busca despesas similares à query"""
        try:
            # Gera embedding para a query
            query_embedding = self.generate_embedding(query)
            
            # Busca os k vizinhos mais próximos
            D, I = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                k
            )
            
            # Retorna as despesas encontradas
            similar_despesas = []
            for i, idx in enumerate(I[0]):
                if idx < len(self.despesas_data):
                    despesa = self.despesas_data.iloc[idx].to_dict()
                    despesa['similarity_score'] = float(D[0][i])
                    similar_despesas.append(despesa)
                    
            return similar_despesas
            
        except Exception as e:
            logger.error(f"Erro ao buscar despesas similares: {str(e)}")
            return []
            
    def get_embedding(self, desp_id: str) -> np.ndarray:
        """Recupera o embedding de uma despesa específica"""
        try:
            if desp_id in self.despesas_data['id'].values:
                idx = self.despesas_data[self.despesas_data['id'] == desp_id].index[0]
                return np.array(self.index.reconstruct(int(idx)))
            return None
        except Exception as e:
            logger.error(f"Erro ao recuperar embedding: {str(e)}")
            return None
