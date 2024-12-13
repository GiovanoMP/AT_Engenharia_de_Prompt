"""
Módulo responsável pela implementação do sistema Self-Ask para o assistente virtual
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .embedding_utils import EmbeddingManager
from .data_loader import DataLoader

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SubQuestion:
    """Classe para representar uma sub-pergunta no processo Self-Ask"""
    question: str
    context: str
    answer: Optional[str] = None
    confidence: float = 0.0

class SelfAskAssistant:
    """Assistente virtual que usa a técnica Self-Ask para responder perguntas"""
    
    def __init__(self):
        """Inicializa o assistente com os recursos necessários"""
        self.embedding_manager = EmbeddingManager()
        self.data_loader = DataLoader()
        self.load_resources()
        
    def load_resources(self):
        """Carrega recursos necessários para o assistente"""
        logger.info("Carregando recursos do assistente")
        try:
            # Tenta carregar índice existente
            self.embedding_manager.load_index("data/embeddings/faiss_index")
        except (FileNotFoundError, ValueError):
            # Se não existir, processa os dados
            logger.info("Índice não encontrado. Processando dados...")
            self.embedding_manager.process_data()
            
        # Carrega dados estruturados
        self.data_loader.load_all_data()
        
    def decompose_question(self, question: str) -> List[SubQuestion]:
        """
        Decompõe uma pergunta complexa em sub-perguntas usando Self-Ask
        
        Args:
            question: Pergunta principal do usuário
            
        Returns:
            Lista de sub-perguntas geradas
        """
        # Mapeamento de tipos de perguntas para sub-perguntas
        question_patterns = {
            "partido": [
                SubQuestion(
                    question="Quais são todos os partidos representados?",
                    context="distribuição partidária"
                ),
                SubQuestion(
                    question="Qual é o número de deputados por partido?",
                    context="contagem por partido"
                )
            ],
            "despesa": [
                SubQuestion(
                    question="Quais são os tipos de despesas registrados?",
                    context="categorias de despesas"
                ),
                SubQuestion(
                    question="Qual é o valor total por tipo de despesa?",
                    context="soma por categoria"
                )
            ],
            "proposição": [
                SubQuestion(
                    question="Quais são as proposições relacionadas ao tema?",
                    context="busca por tema"
                ),
                SubQuestion(
                    question="Quais são os principais pontos dessas proposições?",
                    context="análise de conteúdo"
                )
            ]
        }
        
        # Identifica o tipo de pergunta e retorna as sub-perguntas apropriadas
        for key, sub_questions in question_patterns.items():
            if key.lower() in question.lower():
                return sub_questions
                
        # Se não encontrar um padrão específico, retorna uma abordagem genérica
        return [
            SubQuestion(
                question="Qual é o contexto geral da pergunta?",
                context="contexto geral"
            ),
            SubQuestion(
                question="Quais dados são relevantes para esta pergunta?",
                context="dados relevantes"
            )
        ]
        
    def answer_sub_question(self, sub_question: SubQuestion) -> None:
        """
        Responde uma sub-pergunta usando os dados disponíveis
        
        Args:
            sub_question: Sub-pergunta a ser respondida
        """
        # Busca informações relevantes usando embeddings
        results = self.embedding_manager.search(
            sub_question.question,
            k=3  # Top 3 resultados mais relevantes
        )
        
        # Analisa os resultados e formula uma resposta
        if results:
            relevant_texts = [r['text'] for r in results]
            confidence = 1.0 - min(r['score'] for r in results)  # Converte distância em confiança
            
            # Formata a resposta baseada nos textos relevantes
            answer = self._format_answer(relevant_texts, sub_question.context)
            
            # Atualiza a sub-pergunta com a resposta e confiança
            sub_question.answer = answer
            sub_question.confidence = confidence
        else:
            sub_question.answer = "Não foi possível encontrar informações relevantes."
            sub_question.confidence = 0.0
            
    def _format_answer(self, texts: List[str], context: str) -> str:
        """
        Formata uma resposta baseada nos textos relevantes e contexto
        
        Args:
            texts: Lista de textos relevantes
            context: Contexto da sub-pergunta
            
        Returns:
            Resposta formatada
        """
        # Por enquanto, uma implementação simples
        # Pode ser expandida para usar técnicas mais sofisticadas de NLP
        return " ".join(texts[:2])  # Usa os 2 textos mais relevantes
        
    def answer_question(self, question: str) -> str:
        """
        Responde uma pergunta usando a técnica Self-Ask
        
        Args:
            question: Pergunta do usuário
            
        Returns:
            Resposta final formatada
        """
        logger.info(f"Processando pergunta: {question}")
        
        # Decompõe a pergunta em sub-perguntas
        sub_questions = self.decompose_question(question)
        
        # Responde cada sub-pergunta
        for sub_q in sub_questions:
            self.answer_sub_question(sub_q)
            
        # Combina as respostas em uma resposta final
        final_answer = self._combine_answers(sub_questions)
        
        return final_answer
        
    def _combine_answers(self, sub_questions: List[SubQuestion]) -> str:
        """
        Combina as respostas das sub-perguntas em uma resposta final
        
        Args:
            sub_questions: Lista de sub-perguntas respondidas
            
        Returns:
            Resposta final combinada
        """
        # Filtra sub-perguntas com respostas de alta confiança
        confident_answers = [
            sq for sq in sub_questions 
            if sq.answer and sq.confidence > 0.5
        ]
        
        if not confident_answers:
            return "Desculpe, não consegui encontrar informações suficientes para responder sua pergunta."
            
        # Combina as respostas em um texto coerente
        response_parts = []
        for sq in confident_answers:
            response_parts.append(f"{sq.answer}")
            
        return " ".join(response_parts)
