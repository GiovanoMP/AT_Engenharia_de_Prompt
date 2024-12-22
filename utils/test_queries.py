"""
Testa consultas usando os índices FAISS e Gemini
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import faiss
import pickle
import numpy as np
from embeddings import EmbeddingManager
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carrega variáveis de ambiente
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

def load_index_and_data(name):
    """Carrega índice FAISS e dados originais"""
    # Carrega dados
    with open(f"data/embeddings/{name}_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Carrega índice
    index = faiss.read_index(f"data/embeddings/{name}_index.faiss")
    
    return data, index

def search(manager, query, index_name, k=5):
    """Realiza busca em um índice específico"""
    data, index = load_index_and_data(index_name)
    
    # Gera embedding da query
    query_emb = manager.gerar_embeddings_batch([query])[0].reshape(1, -1)
    
    # Busca similares
    D, I = index.search(query_emb, k)
    
    # Retorna resultados
    results = []
    for i, (dist, idx) in enumerate(zip(D[0], I[0])):
        if idx < len(data):  # Garante índice válido
            results.append({
                'text': data[idx],
                'score': float(dist)
            })
    
    return results

def get_gemini_response(query, context):
    """Gera resposta usando Gemini"""
    model = genai.GenerativeModel('gemini-pro')
    
    SYSTEM_PROMPT = """Você é um assistente especializado em informações sobre a Câmara dos Deputados.
    Seu objetivo é responder perguntas sobre deputados, despesas, proposições e outros temas relacionados.
    Use APENAS as informações fornecidas no contexto para responder às perguntas.
    
    IMPORTANTE:
    1. Analise cuidadosamente todo o contexto fornecido antes de responder
    2. Se houver sumarizações disponíveis, use-as para fornecer uma visão geral do tema
    3. Se houver proposições específicas, mencione-as para dar exemplos concretos
    4. Se houver insights ou análises, use-os para enriquecer sua resposta
    5. Se não houver informações suficientes no contexto, diga claramente que não há dados suficientes
    6. Seja cauteloso ao fazer afirmações absolutas - se os dados forem limitados, indique isso
    7. Ao falar sobre frequências ou quantidades, mencione explicitamente se está se baseando em uma amostra limitada
    8. Se encontrar padrões nos dados, deixe claro que são baseados apenas no contexto fornecido
    
    Seja objetivo e direto em suas respostas, mas forneça o máximo de detalhes relevantes encontrados no contexto.
    Quando relevante, indique limitações nos dados disponíveis para ajudar o usuário a entender o escopo da resposta."""

    USER_PROMPT = """Pergunta: {query}

    Contexto relevante:
    {context}

    Por favor, responda à pergunta usando APENAS as informações do contexto fornecido.
    Se possível, estruture sua resposta em seções:
    1. Visão Geral (baseada em sumarizações, indicando claramente o escopo dos dados)
    2. Exemplos Específicos (baseados em proposições ou registros individuais)
    3. Impactos e Destaques (baseados em insights e análises)
    4. Limitações (se houver aspectos importantes não cobertos pelos dados disponíveis)"""

    prompt = SYSTEM_PROMPT + "\n\n" + USER_PROMPT.format(query=query, context=context)
    
    response = model.generate_content(prompt)
    return response.text

def main():
    # Inicializa manager
    manager = EmbeddingManager()
    
    # Lista de consultas
    queries = [
        ("Qual é o partido político com mais deputados na câmara?", ["insights_distribuicao", "deputados"]),
        ("Qual é o deputado com mais despesas na câmara?", ["insights_despesas", "despesas"]),
        ("Qual é o tipo de despesa mais declarada pelos deputados da câmara?", ["insights_despesas", "despesas"]),
        ("Informações sobre proposições de Economia", ["proposicoes", "sumarizacoes"]),
        ("Informações sobre proposições de Ciência, Tecnologia e Inovação", ["proposicoes", "sumarizacoes"])
    ]
    
    # Processa cada consulta
    for query, index_names in queries:
        print(f"\nPergunta: {query}")
        print("-" * 80)
        
        # Coleta contexto de todos os índices relevantes
        all_context = []
        for index_name in index_names:
            print(f"\nBuscando em {index_name}:")
            results = search(manager, query, index_name)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['text']}")
                print(f"   Score: {result['score']:.4f}")
                all_context.append(result['text'])
        
        # Gera resposta com Gemini
        context_text = "\n".join(all_context)
        print("\nResposta do Gemini:")
        print("-" * 40)
        response = get_gemini_response(query, context_text)
        print(response)
        print("-" * 80)

if __name__ == "__main__":
    main()
