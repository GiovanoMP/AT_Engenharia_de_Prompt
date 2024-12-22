"""
Módulo responsável pela interface de chat do dashboard
"""

import streamlit as st
import logging
from typing import List, Dict, Optional
from pathlib import Path
from .self_ask import SelfAskAssistant

# Configuração de logging
logger = logging.getLogger(__name__)

class ChatInterface:
    """Classe responsável pela interface de chat"""
    
    def __init__(self):
        """Inicializa a interface de chat"""
        logger.info("Iniciando ChatInterface...")
        
        # Garante que o diretório de logs existe
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        self._initialize_session_state()
        logger.info("ChatInterface inicializada com sucesso")
        
    def _initialize_session_state(self):
        """Inicializa o estado da sessão para o chat"""
        logger.info("Inicializando estado da sessão...")
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            logger.info("Histórico do chat inicializado")
            
        if 'chat_assistant' not in st.session_state:
            try:
                logger.info("Tentando inicializar o assistente virtual...")
                st.session_state.chat_assistant = SelfAskAssistant()
                logger.info("Assistente virtual inicializado com sucesso!")
            except Exception as e:
                logger.error(f"Erro ao inicializar assistente: {str(e)}", exc_info=True)
                st.session_state.chat_assistant = None
                
    def render(self):
        """Renderiza a interface do chat"""
        try:
            st.subheader("💬 Assistente Virtual")
            
            # Verifica se o assistente está inicializado
            if st.session_state.chat_assistant is None:
                error_msg = "Não foi possível inicializar o assistente. Por favor, verifique os logs e tente novamente."
                logger.error(error_msg)
                st.error(error_msg)
                return
                
            # Mostra o histórico do chat
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Input do usuário
            if prompt := st.chat_input("Digite sua pergunta..."):
                logger.info(f"Nova pergunta recebida: {prompt}")
                
                # Adiciona mensagem do usuário ao histórico
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.write(prompt)
                    
                # Processa a resposta
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Pensando..."):
                            response = st.session_state.chat_assistant.answer_question(prompt)
                            st.write(response)
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            logger.info("Resposta gerada com sucesso")
                    except Exception as e:
                        error_msg = "Erro ao gerar resposta. Por favor, tente novamente."
                        logger.error(f"Erro ao processar pergunta: {str(e)}", exc_info=True)
                        st.error(error_msg)
                        
        except Exception as e:
            logger.error(f"Erro ao renderizar chat: {str(e)}", exc_info=True)
            st.error("Erro ao renderizar interface do chat.")
