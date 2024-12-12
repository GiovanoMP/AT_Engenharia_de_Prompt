"""
Módulo responsável pelos componentes de visualização
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    """Classe para gerenciamento de visualizações"""
    
    def __init__(self, data_dir: str = "data"):
        """Inicializa o visualizador com o diretório de dados"""
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.images_dir = self.data_dir / "images"
    
    @staticmethod
    def display_image(image_path: Path, caption: Optional[str] = None) -> None:
        """Exibe uma imagem com caption opcional"""
        try:
            image = Image.open(image_path)
            st.image(image, caption=caption, use_column_width=True)
        except Exception as e:
            logger.error(f"Erro ao carregar imagem: {str(e)}")
            st.error(f"Erro ao carregar imagem: {str(e)}")
    
    @staticmethod
    def create_metrics_row(metrics: dict) -> None:
        """Cria uma linha de métricas"""
        cols = st.columns(len(metrics))
        for col, (label, value) in zip(cols, metrics.items()):
            with col:
                st.metric(label=label, value=value)

    def plot_distribuicao_partidos(self, interactive: bool = True) -> go.Figure:
        """Cria e exibe o gráfico de distribuição dos partidos"""
        try:
            # Carrega os dados dos deputados
            df = pd.read_parquet(self.processed_dir / "deputados.parquet")
            
            # Verifica se há dados
            if df is None or df.empty:
                logger.error("Dados dos deputados não disponíveis ou vazios")
                return None
            
            # Calcula a distribuição por partido
            dist_partidos = df['siglaPartido'].value_counts()
            total_deputados = len(df)
            
            # Cria DataFrame para plotly
            plot_df = pd.DataFrame({
                'Partido': dist_partidos.index,
                'Quantidade': dist_partidos.values,
                'Porcentagem': (dist_partidos.values / total_deputados * 100).round(2)
            })
            
            # Ordena os partidos por porcentagem
            plot_df = plot_df.sort_values('Porcentagem', ascending=False)
            
            # Define cores
            bar_color = 'rgb(55, 83, 109)'
            
            # Cria o gráfico de barras
            fig = go.Figure(data=[
                go.Bar(
                    x=plot_df['Partido'],
                    y=plot_df['Porcentagem'],
                    text=plot_df['Porcentagem'].apply(lambda x: f'{x:.1f}%'),
                    textposition='auto',
                    textfont=dict(color='black'),
                    marker=dict(
                        color=bar_color,
                        line=dict(color='black', width=1)
                    ),
                    hovertemplate=(
                        "<b>%{x}</b><br>" +
                        "Porcentagem: %{text}<br>" +
                        "Quantidade: %{customdata} deputados" +
                        "<extra></extra>"
                    ),
                    customdata=plot_df['Quantidade']
                )
            ])
            
            # Atualiza o layout
            fig.update_layout(
                title=dict(
                    text="Distribuição de Deputados por Partido",
                    y=0.95,
                    x=0.5,
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=20, color='black')
                ),
                xaxis=dict(
                    title="Partido",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    tickangle=45,
                    tickfont=dict(size=12, color='black'),
                    title_font=dict(size=14, color='black')
                ),
                yaxis=dict(
                    title="Porcentagem de Deputados",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    tickformat='.1f',
                    ticksuffix='%',
                    tickfont=dict(size=12, color='black'),
                    title_font=dict(size=14, color='black')
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False,
                height=600,
                margin=dict(l=50, r=50, t=80, b=50),
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=14,
                    font_family="Arial",
                    font_color='black'
                )
            )
            
            # Configurações de interatividade
            if not interactive:
                fig.update_layout(
                    dragmode=False,
                    modebar_remove=[
                        'zoom', 'pan', 'select', 'lasso2d', 'zoomIn2d',
                        'zoomOut2d', 'autoScale2d', 'resetScale2d'
                    ]
                )
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro ao criar gráfico de distribuição dos partidos: {str(e)}")
            return None

    @staticmethod
    def create_tab_navigation(tabs: Dict[str, Any]) -> None:
        """Cria navegação por tabs"""
        for tab_name, tab_content in tabs.items():
            with st.tab(tab_name):
                tab_content()
