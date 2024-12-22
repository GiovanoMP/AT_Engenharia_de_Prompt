"""
Aplicativo Streamlit para consulta de informações sobre a Câmara dos Deputados
"""
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Adiciona o diretório raiz ao path para importar utils corretamente
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

import streamlit as st
import faiss
import pickle
import numpy as np
import google.generativeai as genai
import logging
from pathlib import Path
from app.utils.embeddings import EmbeddingManager

# Configuração da página
st.set_page_config(
    page_title="Assistente Virtual - Câmara dos Deputados",
    page_icon="🏛️",
    layout="wide"
)

def get_available_indices():
    """Retorna lista de índices disponíveis"""
    index_files = Path(f"{root_dir}/data/embeddings").glob("*_index.faiss")
    indices = [Path(f).stem.replace("_index", "") for f in index_files]
    return sorted(indices)

def load_index_and_data(name):
    """Carrega índice FAISS e dados originais"""
    # Carrega dados
    with open(f"{root_dir}/data/embeddings/{name}_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Carrega índice
    index = faiss.read_index(f"{root_dir}/data/embeddings/{name}_index.faiss")
    
    return data, index

def search_all_indices(manager, query, k=10):
    """Realiza busca em todos os índices disponíveis"""
    indices = get_available_indices()
    all_results = []
    
    # Gera embedding da query uma única vez
    query_emb = manager.gerar_embeddings_batch([query])[0].reshape(1, -1)
    
    # Mapeamento de índices para prioridade (score menor = mais relevante)
    priority_map = {
        'sumarizacoes': 0.5,  # Maior prioridade para sumarizações
        'proposicoes': 1,     # Segunda maior prioridade para proposições específicas
        'insights_distribuicao': 2,
        'insights_despesas': 2,
        'deputados': 3,
        'despesas': 3
    }
    
    # Ajusta k baseado no tipo de consulta
    if "proposição" in query.lower() or "proposicoes" in query.lower():
        k = k * 2  # Dobra o número de resultados para consultas sobre proposições
    
    for index_name in indices:
        try:
            data, index = load_index_and_data(index_name)
            
            # Ajusta k baseado na prioridade do índice
            index_priority = priority_map.get(index_name, 3)
            index_k = k * (1 + (1/index_priority))  # Mais resultados para índices prioritários
            
            # Busca similares
            D, I = index.search(query_emb, min(index_k, len(data)))
            
            # Adiciona resultados válidos
            for dist, idx in zip(D[0], I[0]):
                if idx < len(data):
                    # Ajusta o score baseado na prioridade do índice
                    adjusted_score = float(dist) * index_priority
                    all_results.append({
                        'text': data[idx],
                        'score': adjusted_score,
                        'source': index_name
                    })
        except Exception as e:
            logging.error(f"Erro ao buscar no índice {index_name}: {e}")
            continue
    
    # Ordena todos os resultados por score ajustado
    all_results.sort(key=lambda x: x['score'])
    
    # Retorna mais resultados para dar contexto suficiente
    return all_results[:min(len(all_results), 40)]

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
    
    # Primeiro as sumarizações
    if 'sumarizacoes' in context_by_source:
        formatted_parts.append("\n=== SUMARIZAÇÕES DE PROPOSIÇÕES ===\n")
        formatted_parts.extend(context_by_source['sumarizacoes'])
        del context_by_source['sumarizacoes']
    
    # Depois proposições específicas
    if 'proposicoes' in context_by_source:
        formatted_parts.append("\n=== PROPOSIÇÕES ESPECÍFICAS ===\n")
        formatted_parts.extend(context_by_source['proposicoes'])
        del context_by_source['proposicoes']
    
    # Depois outros insights
    for source in ['insights_distribuicao', 'insights_despesas']:
        if source in context_by_source:
            formatted_parts.append(f"\n=== INSIGHTS: {source.upper()} ===\n")
            formatted_parts.extend(context_by_source[source])
            del context_by_source[source]
    
    # Por fim, dados complementares
    for source, texts in context_by_source.items():
        formatted_parts.append(f"\n=== DADOS COMPLEMENTARES: {source.upper()} ===\n")
        formatted_parts.extend(texts)
    
    return "\n".join(formatted_parts)

def get_gemini_response(query, context):
    """Gera resposta usando Gemini"""
    model = genai.GenerativeModel('gemini-pro')
    
    SYSTEM_PROMPT = """Você é um especialista em análise legislativa da Câmara dos Deputados, com foco especial em proposições e seus impactos.
    Como analista experiente, você deve usar a técnica Self-Ask para estruturar seu raciocínio:

    1. ANÁLISE HIERÁRQUICA COM SELF-ASK:
    
    Nível 1 - Análise das Sumarizações:
    Q: Existem sumarizações relevantes sobre o tema no contexto?
    A: Procure na seção "SUMARIZAÇÕES" ou resumos gerais
    
    Q: Quais são os principais pontos dessas sumarizações?
    A: Liste os temas, quantidades e destaques encontrados
    
    Nível 2 - Análise das Proposições Específicas:
    Q: Quais proposições específicas tratam do tema?
    A: Identifique números e conteúdos relevantes
    
    Q: Como essas proposições se relacionam com a sumarização?
    A: Compare proposições específicas com o resumo geral
    
    Nível 3 - Síntese e Impactos:
    Q: Quais são os principais impactos esperados?
    A: Analise as consequências potenciais
    
    Q: Existem padrões ou tendências relevantes?
    A: Identifique temas recorrentes
    
    2. REGRAS DE ANÁLISE:
    - Comece SEMPRE pelas sumarizações gerais
    - Cite números específicos de proposições quando relevante
    - Identifique temas recorrentes e padrões
    - Destaque impactos potenciais na sociedade
    
    3. ESTRUTURA DA RESPOSTA:
    - Visão Geral (baseada nas sumarizações)
    - Proposições Específicas (números e conteúdos)
    - Temas Recorrentes
    - Impactos Esperados
    
    Lembre-se: Mantenha o foco nas proposições e seus impactos."""

    USER_PROMPT = """Pergunta: {query}

    Contexto disponível:
    {context}

    Use a técnica Self-Ask para analisar as proposições:

    1. ANÁLISE DAS SUMARIZAÇÕES:
    Q: O que mostram as sumarizações sobre este tema?
    A: [Analise os resumos gerais]
    
    Q: Quais são os principais pontos e números?
    A: [Liste os dados encontrados]

    2. ANÁLISE DAS PROPOSIÇÕES:
    Q: Quais proposições específicas são relevantes?
    A: [Identifique números e conteúdos]
    
    Q: Como elas se relacionam com o tema?
    A: [Analise a relevância]

    3. SÍNTESE E IMPACTOS:
    - Apresente os principais achados
    - Destaque proposições específicas
    - Identifique padrões
    - Analise impactos potenciais"""

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
    # Carrega variáveis de ambiente
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configura Google API
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    
    main()
