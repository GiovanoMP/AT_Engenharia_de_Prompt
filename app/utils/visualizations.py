"""
Módulo responsável pelos componentes de visualização
"""

import streamlit as st
from pathlib import Path
from PIL import Image
from typing import Optional

class Visualizer:
    """Classe para gerenciamento de visualizações"""
    
    @staticmethod
    def display_image(image_path: Path, caption: Optional[str] = None) -> None:
        """Exibe uma imagem com caption opcional"""
        try:
            image = Image.open(image_path)
            st.image(image, caption=caption, use_column_width=True)
        except Exception as e:
            st.error(f"Erro ao carregar imagem: {str(e)}")
    
    @staticmethod
    def create_metrics_row(metrics: dict) -> None:
        """Cria uma linha de métricas"""
        cols = st.columns(len(metrics))
        for col, (label, value) in zip(cols, metrics.items()):
            with col:
                st.metric(label=label, value=value)
    
    @staticmethod
    def create_tab_navigation(tabs: dict) -> None:
        """Cria navegação por tabs"""
        tab_list = st.tabs(list(tabs.keys()))
        for tab, content in zip(tab_list, tabs.values()):
            with tab:
                content()
