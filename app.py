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
        'insights_distribuicao': 0.5,  # Maior prioridade (score menor = mais relevante)
        'insights_despesas': 1,
        'sumarizacoes': 1.5,
        'deputados': 2,
        'despesas': 2,
        'proposicoes': 2
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
    
    # Primeiro os insights de distribuição (mais importantes)
    if 'insights_distribuicao' in context_by_source:
        formatted_parts.append("\n=== DISTRIBUIÇÃO PARTIDÁRIA (ANÁLISE GERAL) ===\n")
        formatted_parts.extend(context_by_source['insights_distribuicao'])
        del context_by_source['insights_distribuicao']
    
    # Depois outros insights e sumarizações
    for source in ['insights_despesas', 'sumarizacoes']:
        if source in context_by_source:
            formatted_parts.append(f"\n=== {source.upper()} ===\n")
            formatted_parts.extend(context_by_source[source])
            del context_by_source[source]
    
    # Por fim, dados específicos
    for source, texts in context_by_source.items():
        formatted_parts.append(f"\n=== DADOS ESPECÍFICOS: {source.upper()} ===\n")
        formatted_parts.extend(texts)
    
    return "\n".join(formatted_parts)

def get_gemini_response(query, context):
    """Gera resposta usando Gemini"""
    model = genai.GenerativeModel('gemini-pro')
    
    SYSTEM_PROMPT = """Você é um especialista em análise de dados parlamentares com vasta experiência em interpretar informações da Câmara dos Deputados.
    Como analista experiente, você deve usar a técnica Self-Ask para estruturar seu raciocínio, seguindo uma análise hierárquica:

    1. ANÁLISE HIERÁRQUICA COM SELF-ASK:
    
    Nível 1 - Análise Geral:
    Q: Existe uma análise geral ou distribuição global no contexto?
    A: Procure na seção "DISTRIBUIÇÃO PARTIDÁRIA (ANÁLISE GERAL)"
    
    Q: Quais são os principais números e percentuais dessa análise?
    A: Liste os números encontrados
    
    Nível 2 - Validação:
    Q: Os dados específicos confirmam a análise geral?
    A: Compare com dados específicos se necessário
    
    Q: Existem discrepâncias ou pontos a esclarecer?
    A: Identifique possíveis inconsistências
    
    Nível 3 - Conclusão:
    Q: Qual é a resposta mais precisa baseada em todos os dados?
    A: Combine análise geral com validações específicas
    
    2. REGRAS DE ANÁLISE:
    - SEMPRE comece pela análise geral/distribuição global
    - Use dados específicos apenas para validação
    - Cite números e percentuais exatos quando disponíveis
    - Indique claramente a fonte dos dados (geral ou específica)
    
    3. ESTRUTURA DA RESPOSTA:
    - Comece com a conclusão principal (baseada na análise geral)
    - Forneça os números exatos e percentuais
    - Adicione contexto e validações
    - Mencione limitações apenas se relevantes
    
    Lembre-se: A análise geral e distribuições globais são mais confiáveis que exemplos específicos."""

    USER_PROMPT = """Pergunta: {query}

    Contexto disponível:
    {context}

    Use a técnica Self-Ask com análise hierárquica:

    1. ANÁLISE DA DISTRIBUIÇÃO GERAL:
    Q: O que mostra a análise geral de distribuição partidária?
    A: [Procure na seção específica]
    
    Q: Quais são os números e percentuais principais?
    A: [Liste os dados encontrados]

    2. VALIDAÇÃO COM DADOS ESPECÍFICOS:
    Q: Os dados específicos confirmam a análise geral?
    A: [Compare se necessário]
    
    Q: Existem informações adicionais relevantes?
    A: [Identifique dados complementares]

    3. CONCLUSÃO FINAL:
    - Apresente a conclusão baseada principalmente na análise geral
    - Cite números e percentuais exatos
    - Adicione contexto relevante
    - Mencione fonte dos dados"""

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
