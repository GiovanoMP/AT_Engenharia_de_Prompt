"""
Módulo principal do dashboard
"""

import streamlit as st
from pathlib import Path
import logging
import sys
import os
import pandas as pd
import plotly.express as px
from datetime import datetime

# Adiciona o diretório raiz ao PYTHONPATH
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from app.utils.data_loader import DataLoader
from app.utils.formatters import format_currency, format_percentage
from app.utils.visualizations import Visualizer
from app.utils.analytics import AnalyticsEngine
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
    if 'selected_filters' not in st.session_state:
        st.session_state.selected_filters = {}

class Dashboard:
    """Classe principal do dashboard"""
    
    def __init__(self):
        """Inicializa o dashboard"""
        try:
            # Configuração inicial da página
            st.set_page_config(
                page_title="Dashboard - Câmara dos Deputados",
                page_icon="📊",
                layout="wide"
            )
            
            # Inicializa o estado da sessão
            initialize_session_state()
            
            # Inicializa os componentes
            self.data_loader = DataLoader(data_dir="data")
            self.visualizer = Visualizer(data_dir="data")
            
            # Carrega os dados apenas uma vez
            if not st.session_state.get('data_loaded', False):
                with st.spinner('Carregando dados...'):
                    self.data_loader.load_all_data()
                    st.session_state.data_loaded = True
                    logger.info("Dados carregados com sucesso")
            
            # Inicializa o analytics após carregar os dados
            self.analytics = AnalyticsEngine(self.data_loader)
            
        except Exception as e:
            logger.error(f"Erro na inicialização do dashboard: {str(e)}")
            st.error("Erro ao inicializar o dashboard. Por favor, recarregue a página.")
            raise

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
            self.data_loader.load_all_data()
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

    def render_filters(self):
        """Renderiza os filtros interativos"""
        with st.sidebar:
            st.subheader("Filtros")
            
            # Filtro de partido
            if self.data_loader.deputados_df is not None:
                partidos = ['Todos'] + sorted(self.data_loader.deputados_df['siglaPartido'].unique().tolist())
                selected_partido = st.selectbox("Partido", partidos)
                if selected_partido != 'Todos':
                    st.session_state.selected_filters['partido'] = selected_partido
                else:
                    st.session_state.selected_filters.pop('partido', None)

    def apply_filters(self, df):
        """Aplica os filtros selecionados ao DataFrame"""
        if df is None:
            return None
            
        try:
            # Filtro de período
            start_date = pd.Timestamp('2024-08-01')
            end_date = pd.Timestamp('2024-08-31')
            df = df[df['dataDocumento'].between(start_date, end_date)]
            
            # Filtro de partido
            if 'partido' in st.session_state.selected_filters:
                partido = st.session_state.selected_filters['partido']
                df = df[df['siglaPartido'] == partido]
            
            return df
        except Exception as e:
            logger.error(f"Erro ao aplicar filtros: {str(e)}")
            return df

    def render_insights(self):
        """Renderiza os insights gerados"""
        try:
            insights = self.analytics.get_top_insights()
            
            for insight in insights:
                with st.expander(f"📊 {insight.title}", expanded=False):
                    st.markdown(f"**Descrição:** {insight.description}")
                    st.markdown("**Métricas Relevantes:**")
                    for metric, value in insight.metrics.items():
                        if isinstance(value, float):
                            st.metric(metric, f"{value:.2f}")
                        else:
                            st.metric(metric, value)
                    
                    st.markdown("**Recomendações:**")
                    for rec in insight.recommendations:
                        st.markdown(f"- {rec}")
        except Exception as e:
            logger.error(f"Erro ao renderizar insights: {str(e)}")
            st.error("Erro ao carregar insights")

    def render_overview(self):
        """Renderiza a página de visão geral"""
        try:
            st.title("Visão Geral da Câmara dos Deputados")
            
            # Verifica se os dados estão disponíveis
            if self.data_loader.deputados_df is None or self.data_loader.deputados_df.empty:
                st.error("Dados não disponíveis. Por favor, recarregue a página.")
                return
            
            # Adiciona a descrição do config.yaml
            with st.expander("Sobre a Câmara dos Deputados", expanded=True):
                st.markdown(self.data_loader.config["overview_summary"]["description"])
            
            # Seção de Insights
            st.header("Insights Principais")
            self.render_insights()
            
            # Métricas principais
            st.header("Métricas Principais")
            metricas = self.data_loader.get_metricas_principais()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Deputados", metricas["Total de Deputados"])
            with col2:
                st.metric("Média de Gastos", format_currency(metricas["Média de Gastos"]))
            with col3:
                st.metric("Total de Proposições", metricas["Total de Proposições"])
            
            # Visualizações interativas
            st.header("Análises Detalhadas")
            
            # Distribuição por partido
            st.subheader("Distribuição por Partido")
            try:
                fig = self.visualizer.plot_distribuicao_partidos(interactive=True)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                logger.error(f"Erro ao exibir distribuição por partido: {str(e)}")
                st.error("Não foi possível exibir a distribuição por partido.")
        
        except Exception as e:
            logger.error(f"Erro ao renderizar visão geral: {str(e)}")
            st.error("Erro ao carregar a visão geral. Por favor, recarregue a página.")

    def render_expenses(self):
        """Renderiza a página de despesas"""
        st.title("Análise de Despesas")
        
        # Análise temporal de despesas
        st.subheader("Evolução Temporal das Despesas")
        df_despesas = self.apply_filters(self.data_loader.despesas_df)
        if df_despesas is not None:
            fig = px.line(
                df_despesas.groupby('dataDocumento')['valorDocumento'].sum().reset_index(),
                x='dataDocumento',
                y='valorDocumento',
                title='Evolução das Despesas ao Longo do Tempo'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detecção de outliers
            st.subheader("Despesas Atípicas")
            outliers = self.analytics.detect_outliers(df_despesas, 'valorDocumento')
            if len(outliers) > 0:
                st.warning(f"Foram detectadas {len(outliers)} despesas atípicas")
                st.dataframe(outliers)

    def render_propositions(self):
        """Renderiza a página de proposições"""
        st.title("Análise de Proposições")
        
        if self.data_loader.proposicoes_df is not None:
            # Status das proposições
            st.subheader("Status das Proposições")
            status_counts = self.data_loader.proposicoes_df['status'].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title='Distribuição de Status das Proposições'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Análise temporal
            st.subheader("Evolução Temporal")
            prop_por_mes = self.data_loader.proposicoes_df.groupby(
                pd.Grouper(key='dataApresentacao', freq='M')
            ).size().reset_index(name='count')
            
            fig = px.line(
                prop_por_mes,
                x='dataApresentacao',
                y='count',
                title='Proposições por Mês'
            )
            st.plotly_chart(fig, use_container_width=True)

    def render(self):
        """Renderiza o dashboard"""
        self.render_sidebar()
        self.render_filters()
        
        # Recarrega os dados quando voltar para a visão geral
        if st.session_state.page == "Visão Geral" and not st.session_state.data_loaded:
            try:
                self.load_data()
                st.session_state.data_loaded = True
            except Exception as e:
                logger.error(f"Erro ao recarregar dados: {str(e)}")
                st.error("Erro ao recarregar os dados. Por favor, recarregue a página.")
                return
        
        if st.session_state.page == "Visão Geral":
            self.render_overview()
        elif st.session_state.page == "Despesas":
            st.session_state.data_loaded = False
            self.render_expenses()
        elif st.session_state.page == "Proposições":
            st.session_state.data_loaded = False
            self.render_propositions()

def main():
    """Função principal do dashboard"""
    try:
        initialize_session_state()
        dashboard = Dashboard()
        
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
            return
        
        dashboard.render()
            
    except Exception as e:
        logger.error(f"Erro fatal: {str(e)}")
        st.error("Ocorreu um erro inesperado. Por favor, recarregue a página ou contate o suporte.")

if __name__ == "__main__":
    main()
