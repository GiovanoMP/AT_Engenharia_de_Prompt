"""
Módulo responsável por análises estatísticas dos dados da Câmara
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Any, Tuple
from .data_loader import DataLoader

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Realiza análises estatísticas sobre os dados da Câmara"""
    
    def __init__(self):
        """Inicializa o analisador de dados"""
        self.data_loader = DataLoader()
        self._load_data()
        
    def _load_data(self):
        """Carrega todos os dados necessários"""
        self.deputados_df = self.data_loader.load_deputados()
        self.proposicoes_df = self.data_loader.load_proposicoes()
        self.despesas_df = self.data_loader.load_despesas()
        
    def analyze_partido_distribution(self) -> Dict[str, Any]:
        """Analisa a distribuição de deputados por partido"""
        try:
            # Contagem por partido
            partido_counts = self.deputados_df['siglaPartido'].value_counts()
            total_deputados = len(self.deputados_df)
            
            # Calcula percentuais
            partido_percentual = (partido_counts / total_deputados * 100).round(2)
            
            # Prepara resultado
            resultado = {
                'total_deputados': total_deputados,
                'total_partidos': len(partido_counts),
                'maior_partido': {
                    'sigla': partido_counts.index[0],
                    'quantidade': int(partido_counts.iloc[0]),
                    'percentual': float(partido_percentual.iloc[0])
                },
                'menor_partido': {
                    'sigla': partido_counts.index[-1],
                    'quantidade': int(partido_counts.iloc[-1]),
                    'percentual': float(partido_percentual.iloc[-1])
                },
                'distribuicao': [
                    {
                        'partido': partido,
                        'deputados': int(count),
                        'percentual': float(partido_percentual[partido])
                    }
                    for partido, count in partido_counts.items()
                ],
                'insights': [
                    f"O {partido_counts.index[0]} é o maior partido com {partido_counts.iloc[0]} deputados ({partido_percentual.iloc[0]:.1f}% do total)",
                    f"Existem {len(partido_counts)} partidos representados na Câmara",
                    f"Os 5 maiores partidos representam {partido_percentual.head(5).sum():.1f}% dos deputados",
                    f"O menor partido é o {partido_counts.index[-1]} com {partido_counts.iloc[-1]} deputado(s)"
                ]
            }
            
            return resultado
        except Exception as e:
            logger.error(f"Erro ao analisar distribuição partidária: {str(e)}")
            return {}
            
    def analyze_despesas(self) -> Dict[str, Any]:
        """Analisa as despesas dos deputados"""
        try:
            # Agrupamento por tipo de despesa
            despesas_grupo = self.despesas_df.groupby('tipoDespesa').agg({
                'valorDocumento': ['sum', 'count', 'mean']
            }).round(2)
            
            # Renomeia as colunas
            despesas_grupo.columns = ['total', 'quantidade', 'media']
            
            # Ordena por valor total
            despesas_grupo = despesas_grupo.sort_values('total', ascending=False)
            
            # Calcula totais gerais
            total_geral = float(self.despesas_df['valorDocumento'].sum())
            
            # Prepara resultado
            resultado = {
                'total_geral': total_geral,
                'total_lancamentos': len(self.despesas_df),
                'maior_tipo_despesa': {
                    'tipo': despesas_grupo.index[0],
                    'total': float(despesas_grupo['total'].iloc[0]),
                    'quantidade': int(despesas_grupo['quantidade'].iloc[0]),
                    'media': float(despesas_grupo['media'].iloc[0])
                },
                'por_tipo': [
                    {
                        'tipo': tipo,
                        'total': float(row['total']),
                        'quantidade': int(row['quantidade']),
                        'media': float(row['media']),
                        'percentual': float(row['total'] / total_geral * 100)
                    }
                    for tipo, row in despesas_grupo.iterrows()
                ]
            }
            
            return resultado
        except Exception as e:
            logger.error(f"Erro ao analisar despesas: {str(e)}")
            return {}
            
    def analyze_deputado_despesas(self) -> Dict[str, Any]:
        """Analisa despesas por deputado"""
        try:
            # Merge com dados dos deputados
            despesas_dep = pd.merge(
                self.despesas_df,
                self.deputados_df[['id', 'nome', 'siglaPartido']],
                on='id'
            )
            
            # Agrupamento por deputado
            deputado_despesas = despesas_dep.groupby(
                ['id', 'nome', 'siglaPartido']
            )['valorDocumento'].agg(['sum', 'count', 'mean']).round(2)
            
            # Ordena por valor total
            deputado_despesas = deputado_despesas.sort_values('sum', ascending=False)
            
            # Prepara resultado
            resultado = {
                'maior_gastador': {
                    'nome': deputado_despesas.index[0][1],
                    'partido': deputado_despesas.index[0][2],
                    'total': float(deputado_despesas['sum'].iloc[0]),
                    'quantidade': int(deputado_despesas['count'].iloc[0]),
                    'media': float(deputado_despesas['mean'].iloc[0])
                },
                'top_10': [
                    {
                        'nome': nome,
                        'partido': partido,
                        'total': float(row['sum']),
                        'quantidade': int(row['count']),
                        'media': float(row['mean'])
                    }
                    for (id_, nome, partido), row in deputado_despesas.head(10).iterrows()
                ]
            }
            
            return resultado
        except Exception as e:
            logger.error(f"Erro ao analisar despesas por deputado: {str(e)}")
            return {}
            
    def analyze_proposicoes_tema(self, tema: str) -> Dict[str, Any]:
        """Analisa proposições por tema"""
        try:
            # Filtra por tema
            tema_mask = self.proposicoes_df['tema'].str.contains(
                tema, case=False, na=False
            )
            proposicoes_tema = self.proposicoes_df[tema_mask]
            
            # Prepara resultado
            resultado = {
                'total_proposicoes': len(proposicoes_tema),
                'proposicoes': [
                    {
                        'tema': row['tema'],
                        'ementa': row['ementa'],
                        'keywords': row.get('keywords', ''),
                        'data': row.get('data', '')
                    }
                    for _, row in proposicoes_tema.iterrows()
                ]
            }
            
            return resultado
        except Exception as e:
            logger.error(f"Erro ao analisar proposições por tema: {str(e)}")
            return {}
