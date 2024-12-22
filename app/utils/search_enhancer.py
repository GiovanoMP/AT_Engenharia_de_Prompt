"""
Módulo responsável por melhorar as buscas usando técnicas avançadas de RAG
"""

import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import numpy as np
from .embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

class SearchEnhancer:
    """
    Classe que implementa técnicas avançadas de busca:
    1. Query Rewriting: Reescreve a pergunta para maximizar relevância
    2. Hybrid Search: Combina busca vetorial com BM25
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """Inicializa o SearchEnhancer"""
        self.embedding_manager = embedding_manager
        self.documents = []
        self.bm25 = None
        self.tokenized_docs = []
    
    def setup_bm25(self, documents: List[Dict[str, Any]]):
        """Configura o índice BM25"""
        try:
            self.documents = documents
            
            # Tokeniza documentos para BM25
            self.tokenized_docs = [
                doc['content'].lower().split()
                for doc in documents
            ]
            
            # Cria índice BM25
            self.bm25 = BM25Okapi(self.tokenized_docs)
            logger.info(f"Índice BM25 criado com {len(documents)} documentos")
            
        except Exception as e:
            logger.error(f"Erro ao configurar BM25: {str(e)}")
            raise
    
    def rewrite_query(self, query: str) -> str:
        """
        Reescreve a query para melhorar a recuperação
        Usa o Gemini para expandir e clarificar a pergunta
        """
        try:
            prompt = f"""Reescreva a seguinte pergunta para maximizar a recuperação de informações sobre a Câmara dos Deputados.

Pergunta original: {query}

Regras para reescrita:
1. Mantenha os termos principais (deputados, partidos, despesas, etc)
2. Adicione sinônimos relevantes
3. Expanda abreviações
4. Mantenha o contexto da Câmara dos Deputados
5. Use termos técnicos quando apropriado

Exemplo:
Original: "Quanto o PT gastou?"
Reescrita: "Qual é o total de despesas e gastos dos deputados do Partido dos Trabalhadores (PT) na Câmara?"

Forneça apenas a pergunta reescrita, sem explicações adicionais.
"""
            
            # Usa o Gemini para reescrever
            from google import generativeai as genai
            import os
            
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            model = genai.GenerativeModel('gemini-pro')
            
            response = model.generate_content(prompt)
            rewritten_query = response.text.strip()
            
            logger.info(f"Query reescrita: '{query}' -> '{rewritten_query}'")
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Erro ao reescrever query: {str(e)}")
            return query  # Retorna query original em caso de erro
    
    def hybrid_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Realiza busca híbrida combinando:
        1. Busca vetorial (embeddings)
        2. Busca por palavras-chave (BM25)
        """
        try:
            # Reescreve a query
            enhanced_query = self.rewrite_query(query)
            
            # Busca vetorial
            vector_results = self.embedding_manager.find_similar(enhanced_query, k)
            
            # Busca BM25
            tokenized_query = enhanced_query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Normaliza scores
            vector_scores = np.array([1.0 - (i/k) for i in range(k)])  # Scores decrescentes
            bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
            
            # Combina scores (média ponderada)
            final_scores = (0.7 * vector_scores + 0.3 * bm25_scores_norm)
            
            # Seleciona top-k resultados
            top_k_indices = np.argsort(final_scores)[-k:][::-1]
            
            # Retorna documentos
            results = []
            for idx in top_k_indices:
                doc = self.documents[idx]
                doc['score'] = float(final_scores[idx])
                results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Erro na busca híbrida: {str(e)}")
            # Fallback para busca vetorial simples
            return self.embedding_manager.find_similar(query, k)
