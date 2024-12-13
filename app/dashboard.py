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
import json

# Adiciona o diretório raiz ao PYTHONPATH
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from app.utils.data_loader import DataLoader
from app.utils.formatters import format_currency, format_percentage
from app.utils.visualizations import Visualizer
from app.config.logging_config import setup_logging

# Configuração de logging
setup_logging()
logger = logging.getLogger(__name__)

class Dashboard:
    """Classe principal do dashboard"""
    
    def __init__(self):
        """Inicializa o dashboard"""
        try:
            self.load_data()
            self.setup_page()
        except Exception as e:
            logger.error(f"Erro ao inicializar o dashboard: {str(e)}")
            st.error("Erro ao inicializar o dashboard.")

    def setup_page(self):
        """Configura a página do Streamlit"""
        st.set_page_config(
            page_title="Dashboard - Análise de Deputados",
            page_icon="📊",
            layout="wide"
        )
        st.title("Dashboard - Análise de Deputados")

    def load_data(self):
        """Carrega os dados necessários para o dashboard"""
        try:
            # Carrega os dados de despesas
            self.serie_despesas = pd.read_parquet(os.path.join(root_dir, 'data/processed/serie_despesas_diarias_deputados.parquet'))
            self.serie_despesas = self.serie_despesas.rename(columns={
                'dataDocumento': 'data',
                'nomeDeputado': 'nome',
                'valorDocumento': 'valor'
            })
            self.deputados = sorted(self.serie_despesas['nome'].unique().tolist())
            
            # Carrega os insights de despesas
            with open(os.path.join(root_dir, 'data/processed/insights_despesas_deputados.json'), 'r', encoding='utf-8') as f:
                self.insights_despesas = json.load(f)
            
            # Carrega os dados de proposições
            self.proposicoes = pd.read_parquet(os.path.join(root_dir, 'data/processed/proposicoes_deputados.parquet'))
            
            # Carrega a sumarização das proposições
            with open(os.path.join(root_dir, 'data/processed/sumarizacao_proposicoes.json'), 'r', encoding='utf-8') as f:
                self.sumarizacao = json.load(f)
                
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            raise

    def render_aba_visao_geral(self):
        """Renderiza a aba de visão geral"""
        try:
            st.header("Visão Geral da Câmara dos Deputados")
            
            # Adiciona a descrição
            with st.expander("Sobre a Câmara dos Deputados", expanded=True):
                st.markdown("""
                A Câmara dos Deputados é uma das casas do Congresso Nacional do Brasil. 
                Ela é composta por representantes do povo, eleitos pelo sistema proporcional 
                em cada estado, território e no Distrito Federal. Este dashboard apresenta 
                análises sobre as atividades dos deputados, incluindo suas despesas e proposições.
                """)
            
            # Seção de Insights
            st.header("Insights Principais")
            self.render_insights()
            
            # Métricas principais
            st.header("Métricas Principais")
            
            # Calcula métricas
            total_deputados = len(self.deputados)
            media_gastos = self.serie_despesas['valor'].mean() if not self.serie_despesas.empty else 0
            total_proposicoes = len(self.proposicoes) if hasattr(self, 'proposicoes') else 0
            
            # Exibe métricas em colunas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Deputados", f"{total_deputados:,}")
            with col2:
                st.metric("Média de Gastos", format_currency(media_gastos))
            with col3:
                st.metric("Total de Proposições", f"{total_proposicoes:,}")
            
            # Distribuição por partido
            st.subheader("Distribuição por Partido")
            self.plot_distribuicao_partidos_from_data()
                
        except Exception as e:
            logger.error(f"Erro ao renderizar visão geral: {str(e)}")
            st.error("Erro ao renderizar visão geral.")

    def plot_distribuicao_partidos_from_data(self):
        """Gera gráfico de distribuição por partido usando os dados disponíveis"""
        try:
            # Usa os dados de despesas para obter a distribuição por partido
            if hasattr(self, 'serie_despesas') and not self.serie_despesas.empty and 'partido' in self.serie_despesas.columns:
                distribuicao = self.serie_despesas.groupby('partido').size().reset_index()
                distribuicao.columns = ['Partido', 'Quantidade']
                distribuicao = distribuicao.sort_values('Quantidade', ascending=False)
                
                # Gera o gráfico
                fig = px.bar(
                    distribuicao,
                    x='Partido',
                    y='Quantidade',
                    title='Distribuição de Deputados por Partido',
                    labels={'Partido': 'Partido', 'Quantidade': 'Quantidade de Deputados'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Dados de distribuição por partido não disponíveis")
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico de distribuição por partido: {str(e)}")
            st.error("Erro ao gerar o gráfico de distribuição por partido.")

    def render_insights(self):
        """Renderiza os insights gerados"""
        try:
            with open(os.path.join(root_dir, 'data/processed/insights_distribuicao_deputados.json'), 'r', encoding='utf-8') as f:
                data = json.load(f)
                insights = data.get('insights', [])
            
            for insight in insights:
                st.markdown(f"• {insight}")
                
        except Exception as e:
            logger.error(f"Erro ao renderizar insights: {str(e)}")
            st.error("Não foi possível carregar os insights.")

    def render_aba_despesas(self):
        """Renderiza a aba de Despesas"""
        try:
            st.header("Análise de Despesas")
            
            # Exibe insights de despesas
            st.subheader("Insights sobre Despesas")
            insights_text = self.insights_despesas.get('insights', '')
            st.markdown(insights_text)
            
            # Análise dos Top 10 Deputados
            st.subheader("Top 10 Deputados com Maiores Gastos")
            if 'analises' in self.insights_despesas and 'top_10_deputados' in self.insights_despesas['analises']:
                top_10 = self.insights_despesas['analises']['top_10_deputados']
                
                # Criar DataFrame para visualização
                top_10_df = pd.DataFrame({
                    'Deputado': list(top_10['valores'].keys()),
                    'Valor': list(top_10['valores'].values()),
                    'Partido': [top_10['partidos'].get(dep, 'N/A') for dep in top_10['valores'].keys()]
                })
                
                # Gráfico de barras dos top 10
                fig_top10 = px.bar(
                    top_10_df,
                    x='Deputado',
                    y='Valor',
                    color='Partido',
                    title='Top 10 Deputados por Valor de Despesas',
                    labels={'Valor': 'Valor Total (R$)'}
                )
                fig_top10.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_top10, use_container_width=True)
            
            # Seleção de deputado individual
            st.subheader("Análise Individual de Deputado")
            if hasattr(self, 'deputados') and self.deputados:
                col1, col2 = st.columns([1, 2])
                with col1:
                    deputado_selecionado = st.selectbox(
                        "Selecione um Deputado para Análise",
                        options=["Selecione um deputado..."] + self.deputados,
                        index=0
                    )
                
                if deputado_selecionado and deputado_selecionado != "Selecione um deputado...":
                    fig_temporal, fig_tipo = self.plot_despesas_deputado(deputado_selecionado)
                    if fig_temporal is not None and fig_tipo is not None:
                        st.plotly_chart(fig_temporal, use_container_width=True)
                        st.plotly_chart(fig_tipo, use_container_width=True)
                        
                        # Estatísticas do deputado
                        dados_deputado = self.serie_despesas[self.serie_despesas['nome'] == deputado_selecionado]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total de Despesas", 
                                     format_currency(dados_deputado['valor'].sum()))
                        with col2:
                            st.metric("Média Mensal", 
                                     format_currency(dados_deputado['valor'].mean()))
                        with col3:
                            st.metric("Número de Despesas", 
                                     f"{len(dados_deputado):,}")
            else:
                st.warning("Lista de deputados não disponível")
                    
        except Exception as e:
            logger.error(f"Erro ao renderizar aba de despesas: {str(e)}")
            st.error("Erro ao renderizar aba de despesas.")

    def plot_despesas_deputado(self, deputado):
        """Gera gráfico de despesas para um deputado específico"""
        try:
            dados_deputado = self.serie_despesas[self.serie_despesas['nome'] == deputado]
            if dados_deputado.empty:
                st.warning(f"Não foram encontradas despesas para o deputado {deputado}")
                return None, None
            
            # Gráfico de linha temporal
            fig_temporal = px.line(
                dados_deputado,
                x='data',
                y='valor',
                title=f'Evolução das Despesas de {deputado}',
                labels={'data': 'Data', 'valor': 'Valor (R$)'}
            )
            
            # Gráfico de barras por tipo de despesa
            fig_tipo = px.bar(
                dados_deputado.groupby('tipoDespesa')['valor'].sum().reset_index(),
                x='tipoDespesa',
                y='valor',
                title=f'Despesas por Categoria - {deputado}',
                labels={'tipoDespesa': 'Tipo de Despesa', 'valor': 'Valor Total (R$)'}
            )
            fig_tipo.update_layout(xaxis_tickangle=-45)
            
            return fig_temporal, fig_tipo
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico: {str(e)}")
            st.error("Erro ao gerar o gráfico de despesas.")
            return None, None

    def render_aba_proposicoes(self):
        """Renderiza a aba de Proposições"""
        try:
            st.header("Análise de Proposições")
            
            # Exibe resumo das proposições por tema
            st.subheader("Resumo por Tema")
            if hasattr(self, 'sumarizacao'):
                for tema in self.sumarizacao.get('sumarizacoes_por_tema', []):
                    with st.expander(f"{tema['tema']} ({tema['quantidade_proposicoes']} proposições)"):
                        st.markdown(tema['sumarizacao'])
            
            # Exibe tabela de proposições com filtros
            st.subheader("Tabela de Proposições")
            if hasattr(self, 'proposicoes') and not self.proposicoes.empty:
                try:
                    # Verifica as colunas disponíveis
                    colunas_disponiveis = self.proposicoes.columns.tolist()
                    logger.info(f"Colunas disponíveis nas proposições: {colunas_disponiveis}")
                    
                    # Adiciona filtros apenas para colunas que existem
                    col1, col2 = st.columns(2)
                    
                    tema_filtro = []
                    if 'tema' in colunas_disponiveis:
                        with col1:
                            temas_disponiveis = sorted(self.proposicoes['tema'].unique().tolist())
                            tema_filtro = st.multiselect(
                                "Filtrar por Tema",
                                options=temas_disponiveis,
                                placeholder="Selecione os temas..."
                            )
                    
                    situacao_filtro = []
                    if 'situacao' in colunas_disponiveis:  # Mudamos de 'status' para 'situacao'
                        with col2:
                            situacoes_disponiveis = sorted(self.proposicoes['situacao'].unique().tolist())
                            situacao_filtro = st.multiselect(
                                "Filtrar por Situação",
                                options=situacoes_disponiveis,
                                placeholder="Selecione as situações..."
                            )
                    
                    # Aplica filtros
                    df_filtrado = self.proposicoes.copy()
                    if tema_filtro and 'tema' in colunas_disponiveis:
                        df_filtrado = df_filtrado[df_filtrado['tema'].isin(tema_filtro)]
                    if situacao_filtro and 'situacao' in colunas_disponiveis:
                        df_filtrado = df_filtrado[df_filtrado['situacao'].isin(situacao_filtro)]
                    
                    # Mostra estatísticas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total de Proposições", f"{len(df_filtrado):,}")
                    with col2:
                        if 'tema' in colunas_disponiveis:
                            st.metric("Temas Únicos", f"{df_filtrado['tema'].nunique():,}")
                    with col3:
                        if 'situacao' in colunas_disponiveis:
                            st.metric("Situações Únicas", f"{df_filtrado['situacao'].nunique():,}")
                    
                    # Configura as colunas para exibição
                    column_config = {}
                    for col in colunas_disponiveis:
                        # Mapeia os nomes das colunas para português
                        column_map = {
                            'tema': 'Tema',
                            'situacao': 'Situação',
                            'data': 'Data',
                            'autor': 'Autor',
                            'ementa': 'Ementa',
                            'tipo': 'Tipo',
                            'numero': 'Número'
                        }
                        if col in column_map:
                            column_config[col] = column_map[col]
                    
                    # Exibe a tabela filtrada
                    st.dataframe(
                        df_filtrado,
                        column_config=column_config,
                        hide_index=True
                    )
                    
                    # Adiciona visualizações se houver dados
                    if not df_filtrado.empty:
                        col1, col2 = st.columns(2)
                        
                        # Gráfico por tema
                        if 'tema' in colunas_disponiveis:
                            with col1:
                                contagem_temas = df_filtrado['tema'].value_counts()
                                fig_tema = px.pie(
                                    values=contagem_temas.values,
                                    names=contagem_temas.index,
                                    title='Distribuição por Tema'
                                )
                                st.plotly_chart(fig_tema, use_container_width=True)
                        
                        # Gráfico por situação
                        if 'situacao' in colunas_disponiveis:
                            with col2:
                                contagem_situacao = df_filtrado['situacao'].value_counts()
                                fig_situacao = px.bar(
                                    x=contagem_situacao.index,
                                    y=contagem_situacao.values,
                                    title='Quantidade por Situação',
                                    labels={'x': 'Situação', 'y': 'Quantidade'}
                                )
                                fig_situacao.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig_situacao, use_container_width=True)
                    else:
                        st.info("Nenhum dado encontrado para os filtros selecionados")
                        
                except Exception as e:
                    logger.error(f"Erro ao processar dados de proposições: {str(e)}")
                    st.error("Erro ao processar dados de proposições.")
            else:
                st.warning("Dados de proposições não disponíveis")
                
        except Exception as e:
            logger.error(f"Erro ao renderizar aba de proposições: {str(e)}")
            st.error("Erro ao renderizar aba de proposições.")

    def run(self):
        """Executa o dashboard"""
        try:
            # Criação das abas
            tab_visao_geral, tab_despesas, tab_proposicoes = st.tabs([
                "Visão Geral",
                "Despesas",
                "Proposições"
            ])

            # Renderiza cada aba
            with tab_visao_geral:
                self.render_aba_visao_geral()
            
            with tab_despesas:
                self.render_aba_despesas()
            
            with tab_proposicoes:
                self.render_aba_proposicoes()
                
        except Exception as e:
            logger.error(f"Erro ao executar o dashboard: {str(e)}")
            st.error("Erro ao executar o dashboard.")

if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run()
