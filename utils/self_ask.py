"""
Implementação da técnica Self-Ask para o assistente virtual
"""

from typing import List, Dict, Any
import logging
from .embeddings import EmbeddingManager

class SelfAskAssistant:
    """
    Implementa a técnica Self-Ask para melhorar as respostas do assistente.
    A técnica consiste em quebrar perguntas complexas em sub-perguntas mais simples.
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Inicializa o assistente com Self-Ask
        
        Args:
            embedding_manager: Gerenciador de embeddings para busca semântica
        """
        self.embedding_manager = embedding_manager
        self.context = {}
        
    def _gerar_subperguntas(self, pergunta: str) -> List[str]:
        """
        Gera sub-perguntas para uma pergunta complexa
        
        Args:
            pergunta: Pergunta principal do usuário
            
        Returns:
            Lista de sub-perguntas
        """
        # Exemplos de sub-perguntas para diferentes tipos de consultas
        subperguntas = {
            "partido": [
                "Quais são todos os partidos representados?",
                "Quantos deputados cada partido tem?",
                "Qual partido tem mais representantes?"
            ],
            "despesas": [
                "Quais são os tipos de despesas registrados?",
                "Qual o valor total por deputado?",
                "Quais são as despesas mais comuns?"
            ],
            "proposicoes": [
                "Qual o tema da proposição?",
                "Quais são as palavras-chave relevantes?",
                "Existem proposições similares?"
            ]
        }
        
        # Identifica o tipo de pergunta e retorna sub-perguntas relevantes
        for tipo, perguntas in subperguntas.items():
            if tipo.lower() in pergunta.lower():
                return perguntas
                
        return ["Poderia especificar melhor sua pergunta?"]
        
    def _buscar_contexto(self, pergunta: str) -> Dict[str, Any]:
        """
        Busca informações relevantes usando embeddings
        
        Args:
            pergunta: Pergunta ou sub-pergunta
            
        Returns:
            Dicionário com informações relevantes
        """
        # Busca documentos similares
        resultados = self.embedding_manager.buscar_similares(pergunta)
        
        # Organiza o contexto
        contexto = {
            "documentos_relevantes": resultados,
            "confianca": 1.0 / (1.0 + resultados[0]["distancia"]) if resultados else 0
        }
        
        return contexto
    
    def responder(self, pergunta: str) -> str:
        """
        Responde à pergunta do usuário usando Self-Ask
        
        Args:
            pergunta: Pergunta do usuário
            
        Returns:
            Resposta elaborada
        """
        # Gera sub-perguntas
        subperguntas = self._gerar_subperguntas(pergunta)
        
        # Busca contexto para cada sub-pergunta
        contextos = []
        for subpergunta in subperguntas:
            contexto = self._buscar_contexto(subpergunta)
            contextos.append({
                "pergunta": subpergunta,
                "contexto": contexto
            })
            
        # Combina as informações para gerar resposta
        resposta = self._formatar_resposta(pergunta, contextos)
        
        return resposta
    
    def _formatar_resposta(self, pergunta: str, contextos: List[Dict]) -> str:
        """
        Formata a resposta final baseada nos contextos coletados
        
        Args:
            pergunta: Pergunta original
            contextos: Lista de contextos das sub-perguntas
            
        Returns:
            Resposta formatada
        """
        # Exemplo de formatação básica
        resposta = f"Analisando sua pergunta: '{pergunta}'\n\n"
        
        for ctx in contextos:
            resposta += f"- {ctx['pergunta']}\n"
            if ctx['contexto']['documentos_relevantes']:
                doc = ctx['contexto']['documentos_relevantes'][0]
                resposta += f"  → {doc['texto'][:200]}...\n\n"
                
        return resposta
