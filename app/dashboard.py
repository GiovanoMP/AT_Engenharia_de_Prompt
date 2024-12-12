"""
Módulo principal do dashboard
"""

import streamlit as st
from pathlib import Path
import logging
import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from app.utils.data_loader import DataLoader
from app.utils.formatters import format_currency, format_percentage
from app.config.logging_config import setup_logging

# Configuração de logging
setup_logging()
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Inicializa o estado da sessão do Streamlit"""
    if 'page' not in st.session_state:
        st.session_state.page = 'Visão Geral'
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None

class Dashboard:
    """Classe principal do dashboard"""
    
    def __init__(self):
        self.setup_page()
        try:
            self.data_loader = DataLoader(data_dir="data")
            if not st.session_state.data_loaded:
                self.load_data()
                st.session_state.data_loaded = True
        except Exception as e:
            logger.error(f"Erro na inicialização: {str(e)}")
            st.error("Erro ao inicializar o dashboard. Por favor, verifique os logs.")
    
    def setup_page(self):
        """Configura a página do Streamlit"""
        st.set_page_config(
            page_title="Dashboard Câmara dos Deputados",
            page_icon="🏛️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def load_data(self):
        """Carrega os dados necessários"""
        try:
            self.config = self.data_loader.load_config()
            self.insights = self.data_loader.load_insights()
            st.session_state.error_message = None
            logger.info("Dados carregados com sucesso")
        except Exception as e:
            error_msg = f"Erro ao carregar dados: {str(e)}"
            logger.error(error_msg)
            st.session_state.error_message = error_msg
            raise
    
    def render_sidebar(self):
        """Renderiza a barra lateral"""
        with st.sidebar:
            st.title("Navegação")
            st.radio(
                "Escolha uma página",
                ["Visão Geral", "Despesas", "Proposições"],
                key="page"
            )
    
    def render_overview(self):
        """Renderiza a página de visão geral"""
        st.title("Visão Geral da Câmara dos Deputados")
        
        # Métricas principais
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_deputados = self.insights[0].get("total_deputados", 0)
            st.metric("Total de Deputados", total_deputados)
            
        with col2:
            media_gastos = self.insights[0].get("media_gastos", 0)
            st.metric("Média de Gastos", format_currency(media_gastos))
            
        with col3:
            total_proposicoes = self.insights[0].get("total_proposicoes", 0)
            st.metric("Total de Proposições", total_proposicoes)
        
        # Distribuição por partido
        st.subheader("Distribuição por Partido")
        distribuicao_path = self.data_loader.get_image_path("distribuicao_partidos.png")
        if distribuicao_path:
            st.image(str(distribuicao_path), use_column_width=True)
        else:
            st.warning("Imagem de distribuição não encontrada")
    
    def render_expenses(self):
        """Renderiza a página de despesas"""
        st.title("Análise de Despesas")
        st.info("Em desenvolvimento")
    
    def render_propositions(self):
        """Renderiza a página de proposições"""
        st.title("Análise de Proposições")
        st.info("Em desenvolvimento")
    
    def render(self):
        """Renderiza o dashboard"""
        self.render_sidebar()
        
        if st.session_state.page == "Visão Geral":
            self.render_overview()
        elif st.session_state.page == "Despesas":
            self.render_expenses()
        elif st.session_state.page == "Proposições":
            self.render_propositions()

def main():
    """Função principal do dashboard"""
    try:
        initialize_session_state()
        dashboard = Dashboard()
        
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
            return

        dashboard.render_sidebar()
        
        # Renderiza a página selecionada
        if st.session_state.page == "Visão Geral":
            dashboard.render_overview()
        elif st.session_state.page == "Despesas":
            dashboard.render_expenses()
        elif st.session_state.page == "Proposições":
            dashboard.render_propositions()
            
    except Exception as e:
        logger.error(f"Erro fatal: {str(e)}")
        st.error("Ocorreu um erro inesperado. Por favor, recarregue a página ou contate o suporte.")

if __name__ == "__main__":
    main()
