"""
Aplicativo Streamlit para consulta de informações sobre a Câmara dos Deputados
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import streamlit as st
import faiss
import pickle
import numpy as np
from utils.embeddings import EmbeddingManager
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
import glob

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carrega variáveis de ambiente
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

def get_available_indices():
    """Retorna lista de índices disponíveis"""
    embeddings_dir = Path("data/embeddings")
    index_files = glob.glob(str(embeddings_dir / "*_index.faiss"))
    indices = [Path(f).stem.replace("_index", "") for f in index_files]
    return sorted(indices)

def load_index_and_data(name):
    """Carrega índice FAISS e dados originais"""
    # Carrega dados
    with open(f"data/embeddings/{name}_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Carrega índice
    index = faiss.read_index(f"data/embeddings/{name}_index.faiss")
    
    return data, index

def search_all_indices(manager, query, k=10):
    """Realiza busca em todos os índices disponíveis"""
    indices = get_available_indices()
    all_results = []
    
    # Gera embedding da query uma única vez
    query_emb = manager.gerar_embeddings_batch([query])[0].reshape(1, -1)
    
    # Mapeamento de índices para prioridade
    priority_map = {
        'insights_distribuicao': 1,
        'insights_despesas': 1,
        'sumarizacoes': 2,
        'deputados': 3,
        'despesas': 3,
        'proposicoes': 3
    }
    
    for index_name in indices:
        try:
            data, index = load_index_and_data(index_name)
            
            # Ajusta k baseado na prioridade do índice
            index_priority = priority_map.get(index_name, 3)
            index_k = k * index_priority
            
            # Busca similares
            D, I = index.search(query_emb, index_k)
            
            # Adiciona resultados válidos
            for dist, idx in zip(D[0], I[0]):
                if idx < len(data):
                    # Ajusta o score baseado na prioridade do índice
                    adjusted_score = float(dist) / index_priority
                    all_results.append({
                        'text': data[idx],
                        'score': adjusted_score,
                        'source': index_name
                    })
        except Exception as e:
            logger.error(f"Erro ao buscar no índice {index_name}: {e}")
            continue
    
    # Ordena todos os resultados por score ajustado
    all_results.sort(key=lambda x: x['score'])
    
    # Retorna mais resultados, mas mantém um limite razoável
    return all_results[:min(len(all_results), 30)]

def format_context(results):
    """Formata o contexto de forma estruturada"""
    context_by_source = {}
    
    # Agrupa resultados por fonte
    for result in results:
        source = result['source']
        if source not in context_by_source:
            context_by_source[source] = []
        context_by_source[source].append(result['text'])
    
    # Formata o contexto
    formatted_parts = []
    
    # Primeiro insights e sumarizações
    for source in ['insights_distribuicao', 'insights_despesas', 'sumarizacoes']:
        if source in context_by_source:
            formatted_parts.append(f"\n=== {source.upper()} ===\n")
            formatted_parts.extend(context_by_source[source])
            del context_by_source[source]
    
    # Depois os dados específicos
    for source, texts in context_by_source.items():
        formatted_parts.append(f"\n=== {source.upper()} ===\n")
        formatted_parts.extend(texts)
    
    return "\n".join(formatted_parts)

def get_gemini_response(query, context):
    """Gera resposta usando Gemini"""
    model = genai.GenerativeModel('gemini-pro')
    
    SYSTEM_PROMPT = """Você é um especialista em análise de dados parlamentares com vasta experiência em interpretar informações da Câmara dos Deputados.
    Como analista experiente, você deve usar a técnica Self-Ask para estruturar seu raciocínio:
    
    1. TÉCNICA SELF-ASK:
    - Faça perguntas específicas para si mesmo sobre o problema
    - Responda cada pergunta usando os dados disponíveis
    - Use as respostas para construir seu raciocínio
    - Questione suas próprias conclusões
    
    Exemplo de Self-Ask:
    Q: Quais dados são necessários para responder esta pergunta?
    A: Preciso de X, Y e Z...
    
    Q: Estes dados estão disponíveis no contexto?
    A: Sim/Não, encontrei/não encontrei...
    
    Q: Que análises posso fazer com estes dados?
    A: Posso comparar X com Y...
    
    Q: Esta conclusão é suportada pelos dados?
    A: Sim/Não, porque...
    
    2. ANÁLISE DE DADOS:
    - Analise criticamente os dados disponíveis
    - Identifique padrões e tendências relevantes
    - Faça inferências razoáveis quando apropriado
    - Use seu conhecimento especializado para contextualizar
    
    3. COMUNICAÇÃO:
    - Estruture sua resposta de forma clara
    - Explique seu raciocínio passo a passo
    - Destaque limitações e incertezas
    - Forneça exemplos específicos dos dados
    
    Lembre-se: Use Self-Ask para guiar seu pensamento e tornar seu raciocínio transparente."""

    USER_PROMPT = """Pergunta: {query}

    Contexto disponível:
    {context}

    Use a técnica Self-Ask para analisar a pergunta:

    1. QUESTÕES INICIAIS:
    - Do que precisamente trata esta pergunta?
    - Quais dados são necessários para respondê-la?
    - Estes dados estão disponíveis no contexto?

    2. ANÁLISE DOS DADOS:
    - Que padrões posso identificar nos dados?
    - Como posso verificar se estes padrões são significativos?
    - Que conclusões posso tirar com segurança?

    3. VALIDAÇÃO:
    - Minhas conclusões são suportadas pelos dados?
    - Existem limitações importantes a considerar?
    - Que ressalvas preciso fazer?

    4. RESPOSTA FINAL:
    Estruture sua resposta incluindo:
    - Processo de raciocínio usado
    - Dados e análises realizadas
    - Conclusões e limitações
    - Recomendações, se apropriado"""

    prompt = SYSTEM_PROMPT + "\n\n" + USER_PROMPT.format(query=query, context=context)
    
    response = model.generate_content(prompt)
    return response.text

def main():
    st.title("🏛️ Assistente Virtual da Câmara dos Deputados")
    
    st.markdown("""
    Este assistente pode responder perguntas sobre:
    - 👥 Deputados e partidos
    - 💰 Despesas e gastos
    - 📜 Proposições e projetos de lei
    - 📊 Análises e insights
    """)
    
    # Inicializa o histórico de chat se não existir
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Mostra histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Campo de entrada do usuário
    if prompt := st.chat_input("Digite sua pergunta..."):
        # Adiciona pergunta do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Mostra indicador de "pensando"
        with st.chat_message("assistant"):
            with st.spinner("Buscando informações..."):
                # Inicializa manager
                manager = EmbeddingManager()
                
                # Busca em todos os índices
                results = search_all_indices(manager, prompt)
                
                # Formata o contexto de forma estruturada
                context = format_context(results)
                
                # Gera resposta
                response = get_gemini_response(prompt, context)
                
                # Adiciona resposta ao histórico
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)
                
                # Opcionalmente, mostra o contexto usado (para debug)
                if st.checkbox("Mostrar contexto usado"):
                    st.text_area("Contexto", context, height=300)

if __name__ == "__main__":
    main()
