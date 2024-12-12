"""
Módulo responsável pelo carregamento e processamento de dados
"""

import yaml
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import streamlit as st
import logging

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoadError(Exception):
    """Erro ao carregar dados"""
    pass

# Funções de cache independentes
@st.cache_data(ttl=3600)  # Cache por 1 hora
def _load_yaml_file(file_path: str) -> Dict[str, Any]:
    """Carrega arquivo YAML com cache"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise DataLoadError("Arquivo YAML deve conter um dicionário")
            return data
    except Exception as e:
        error_msg = f"Erro ao carregar arquivo YAML: {str(e)}"
        logger.error(error_msg)
        raise DataLoadError(error_msg)

@st.cache_data(ttl=1800)  # Cache por 30 minutos
def _load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Carrega arquivo JSON com cache"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]  # Converte para lista se for um único objeto
            return data
    except Exception as e:
        error_msg = f"Erro ao carregar arquivo JSON: {str(e)}"
        logger.error(error_msg)
        raise DataLoadError(error_msg)

@st.cache_resource
def _verify_image_path(image_path: str) -> Optional[Path]:
    """Verifica existência de imagem com cache"""
    path = Path(image_path)
    if not path.exists():
        logger.warning(f"Imagem não encontrada: {image_path}")
        return None
    return path

class DataLoader:
    """Classe responsável pelo carregamento e gerenciamento de dados"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self._initialize_data_structures()
    
    def _initialize_data_structures(self):
        """Inicializa as estruturas de dados"""
        self.config = None
        self.insights = None
        self.last_update = None
    
    def load_config(self) -> Dict[str, Any]:
        """Carrega o arquivo de configuração YAML com cache"""
        with st.spinner('Carregando configurações...'):
            config_path = str(self.data_dir / "config.yaml")
            self.config = _load_yaml_file(config_path)
            self.last_update = datetime.now()
            logger.info("Arquivo de configuração carregado com sucesso")
            return self.config
    
    def load_insights(self) -> List[Dict[str, Any]]:
        """Carrega o arquivo de insights JSON com cache"""
        with st.spinner('Carregando insights...'):
            insights_path = str(self.data_dir / "processed" / "insights_distribuicao_deputados.json")
            self.insights = _load_json_file(insights_path)
            logger.info("Arquivo de insights carregado com sucesso")
            return self.insights
    
    def get_image_path(self, image_name: str) -> Optional[Path]:
        """Retorna o caminho para uma imagem com verificação de existência"""
        image_path = str(self.data_dir / "processed" / image_name)
        return _verify_image_path(image_path)
    
    def get_last_update(self) -> str:
        """Retorna a data da última atualização dos dados"""
        return self.last_update.strftime("%d/%m/%Y %H:%M:%S") if self.last_update else "Não disponível"
