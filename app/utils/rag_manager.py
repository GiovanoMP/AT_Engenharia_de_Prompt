"""
Módulo responsável pelo RAG (Retrieval Augmented Generation)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
from pathlib import Path
from .embedding_manager import EmbeddingManager
from .data_loader import DataLoader
from .search_enhancer import SearchEnhancer

logger = logging.getLogger(__name__)

class RAGManager:
    """Gerencia a recuperação de documentos usando BERT e FAISS"""
    
    def __init__(self, data_loader: DataLoader, embedding_manager: EmbeddingManager):
        """Inicializa o RAG Manager"""
        self.data_loader = data_loader
        self.embedding_manager = embedding_manager
        
        # Processa dados para FAISS
        self._process_data()
        
    def _process_data(self):
        """Processa os dados para o índice FAISS"""
        try:
            # Processa deputados
            deputados_data = self.data_loader.deputados_df.to_dict('records')
            self.embedding_manager.process_data(
                deputados_data,
                ['nome', 'siglaPartido', 'siglaUf', 'email', 'urlFoto']
            )
            
            # Processa proposições
            proposicoes_data = self.data_loader.proposicoes_df.to_dict('records')
            self.embedding_manager.process_data(
                proposicoes_data,
                ['siglaTipo', 'numero', 'ano', 'ementa', 'tema']
            )
            
            # Processa despesas
            despesas_data = self.data_loader.despesas_df.to_dict('records')
            self.embedding_manager.process_data(
                despesas_data,
                ['nomeDeputado', 'tipoDespesa', 'dataDocumento', 'valorDocumento']
            )
            
            logger.info("Dados processados e indexados com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao processar dados para FAISS: {str(e)}")
            raise
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Recupera documentos relevantes usando FAISS"""
        try:
            # Gera embedding para a query
            query_embedding = self.embedding_manager.generate_embedding(query)
            
            # Busca documentos similares
            D, I = self.embedding_manager.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                k
            )
            
            # Recupera documentos
            documents = []
            for idx in I[0]:
                if idx < len(self.data_loader.deputados_df):
                    doc = self.data_loader.deputados_df.iloc[idx]
                    documents.append(f"Deputado: {doc['nome']} ({doc['siglaPartido']}-{doc['siglaUf']})")
                elif idx < len(self.data_loader.proposicoes_df):
                    doc = self.data_loader.proposicoes_df.iloc[idx]
                    documents.append(f"Proposição: {doc['siglaTipo']} {doc['numero']}/{doc['ano']} - {doc['ementa']}")
                else:
                    doc = self.data_loader.despesas_df.iloc[idx]
                    documents.append(f"Despesa: {doc['nomeDeputado']} - {doc['tipoDespesa']} - R$ {doc['valorDocumento']:,.2f}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Erro ao recuperar documentos: {str(e)}")
            return []
    
    def generate_prompt(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Gera prompt para o LLM usando documentos recuperados"""
        try:
            # Organiza documentos por tipo
            deputados_docs = [doc for doc in retrieved_docs if 'Deputado' in doc]
            proposicoes_docs = [doc for doc in retrieved_docs if 'Proposição' in doc]
            despesas_docs = [doc for doc in retrieved_docs if 'Despesa' in doc]
            
            context = "Informações relevantes:\n\n"
            
            # Adiciona informações de deputados
            if deputados_docs:
                context += "Deputados:\n"
                for doc in deputados_docs:
                    context += f"{doc}\n"
                    
            # Adiciona informações de proposições
            if proposicoes_docs:
                context += "Proposições:\n"
                for doc in proposicoes_docs:
                    context += f"{doc}\n"
                    
            # Adiciona informações de despesas
            if despesas_docs:
                context += "Despesas:\n"
                for doc in despesas_docs:
                    context += f"{doc}\n"
                    
            # Gera o prompt final
            prompt = f"""Você é um assistente especializado em dados.

Contexto:
{context}

Pergunta: {query}

Instruções:
1. Use APENAS as informações fornecidas no contexto
2. Seja direto e objetivo na resposta
3. Cite números e percentuais quando disponíveis
4. Se não tiver certeza, indique claramente

Responda à pergunta usando as informações fornecidas:"""

            return prompt
            
        except Exception as e:
            logger.error(f"Erro ao gerar prompt: {str(e)}")
            return ""
