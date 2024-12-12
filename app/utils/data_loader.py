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

@st.cache_data
def _load_parquet_file(file_path: str) -> pd.DataFrame:
    """Carrega arquivo parquet com cache"""
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        error_msg = f"Erro ao carregar arquivo parquet: {str(e)}"
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
        self.deputados_df = None
        self.proposicoes_df = None
        self.despesas_df = None
    
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

    def load_deputados(self) -> pd.DataFrame:
        """Carrega dados dos deputados"""
        with st.spinner('Carregando dados dos deputados...'):
            file_path = str(self.data_dir / "processed" / "deputados.parquet")
            self.deputados_df = _load_parquet_file(file_path)
            logger.info("Dados dos deputados carregados com sucesso")
            return self.deputados_df

    def load_proposicoes(self) -> pd.DataFrame:
        """Carrega dados das proposições"""
        with st.spinner('Carregando dados das proposições...'):
            file_path = str(self.data_dir / "processed" / "proposicoes_deputados.parquet")
            self.proposicoes_df = _load_parquet_file(file_path)
            logger.info("Dados das proposições carregados com sucesso")
            return self.proposicoes_df

    def load_despesas(self) -> pd.DataFrame:
        """Carrega dados das despesas"""
        with st.spinner('Carregando dados das despesas...'):
            file_path = str(self.data_dir / "processed" / "serie_despesas_diarias_deputados.parquet")
            self.despesas_df = _load_parquet_file(file_path)
            logger.info("Dados das despesas carregados com sucesso")
            return self.despesas_df

    def load_all_data(self) -> None:
        """Carrega todos os dados necessários"""
        try:
            self.load_config()
            self.load_insights()
            self.load_deputados()
            self.load_proposicoes()
            self.load_despesas()
            
            # Verifica se os dados principais foram carregados
            if self.deputados_df is None or self.deputados_df.empty:
                raise DataLoadError("Falha ao carregar dados dos deputados")
                
            logger.info("Todos os dados foram carregados com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            raise DataLoadError(f"Falha ao carregar dados: {str(e)}")
    
    def get_image_path(self, image_name: str) -> Optional[Path]:
        """Retorna o caminho para uma imagem com verificação de existência"""
        image_path = str(self.data_dir / "processed" / image_name)
        return _verify_image_path(image_path)
    
    def get_last_update(self) -> str:
        """Retorna a data da última atualização dos dados"""
        return self.last_update.strftime("%d/%m/%Y %H:%M:%S") if self.last_update else "Não disponível"

    def get_metricas_principais(self) -> Dict[str, Any]:
        """Retorna as métricas principais para o dashboard"""
        try:
            total_deputados = len(self.deputados_df) if self.deputados_df is not None else 0
            media_gastos = self.despesas_df['valorDocumento'].mean() if self.despesas_df is not None else 0
            total_proposicoes = len(self.proposicoes_df) if self.proposicoes_df is not None else 0
            
            return {
                "Total de Deputados": total_deputados,
                "Média de Gastos": media_gastos,
                "Total de Proposições": total_proposicoes
            }
        except Exception as e:
            logger.error(f"Erro ao calcular métricas principais: {str(e)}")
            return {
                "Total de Deputados": 0,
                "Média de Gastos": 0,
                "Total de Proposições": 0
            }
