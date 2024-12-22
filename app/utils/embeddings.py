"""
Gerenciamento de embeddings usando BERT e FAISS
"""

import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict
import logging
from tqdm import tqdm

class EmbeddingManager:
    """Gerencia a criação e busca de embeddings usando BERT e FAISS"""
    
    def __init__(self):
        """Inicializa o gerenciador de embeddings"""
        self.model_name = "neuralmind/bert-base-portuguese-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.index = None
        self.embeddings = None
        self.dados_originais = []
        
    def gerar_embeddings_batch(self, textos: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Gera embeddings para uma lista de textos em batches
        
        Args:
            textos: Lista de textos
            batch_size: Tamanho do batch
            
        Returns:
            Array numpy com embeddings
        """
        embeddings = []
        
        # Processa em batches
        for i in tqdm(range(0, len(textos), batch_size), desc="Gerando embeddings"):
            batch = textos[i:i + batch_size]
            
            # Tokeniza batch
            inputs = self.tokenizer(batch, return_tensors="pt", 
                                  max_length=512, truncation=True, padding=True)
            
            # Move para GPU se disponível
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Gera embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
        
    def criar_index(self, textos: List[str], batch_size: int = 32):
        """
        Cria índice FAISS com embeddings dos textos
        
        Args:
            textos: Lista de textos para indexar
            batch_size: Tamanho do batch para processamento
        """
        # Gera embeddings em batches
        embeddings = self.gerar_embeddings_batch(textos, batch_size)
        
        # Cria índice FAISS
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Adiciona em batches
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Adicionando ao FAISS"):
            batch = embeddings[i:i + batch_size]
            index.add(batch)
        
        # Salva no objeto
        self.index = index
        self.embeddings = embeddings
        self.dados_originais = textos
        
    def buscar_similares(self, texto: str, k: int = 5) -> List[Dict]:
        """
        Busca textos similares no índice
        
        Args:
            texto: Texto para buscar similares
            k: Número de resultados
            
        Returns:
            Lista com os k textos mais similares
        """
        if self.index is None:
            return []
            
        # Gera embedding da query
        query_emb = self.gerar_embeddings_batch([texto])[0].reshape(1, -1)
        
        # Busca similares
        D, I = self.index.search(query_emb, k)
        
        # Formata resultados
        resultados = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(self.dados_originais):
                resultados.append({
                    'texto': self.dados_originais[idx],
                    'distancia': float(dist)
                })
                
        return resultados
