"""
Módulo responsável pela análise avançada de dados e geração de insights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Insight:
    """Classe para estruturar insights"""
    title: str
    description: str
    importance: int  # 1-5, onde 5 é mais importante
    category: str
    metrics: Dict[str, Any]
    recommendations: List[str]

class AnalyticsEngine:
    """Motor de análise de dados e geração de insights"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.insights: List[Insight] = []
    
    def detect_outliers(self, df: pd.DataFrame, column: str, threshold: float = 3) -> pd.DataFrame:
        """Detecta outliers usando o método do z-score"""
        try:
            if df is None or df.empty:
                return pd.DataFrame()
                
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            return df[z_scores > threshold]
        except Exception as e:
            logger.error(f"Erro na detecção de outliers: {str(e)}")
            return pd.DataFrame()

    def calculate_derived_metrics(self) -> Dict[str, Any]:
        """Calcula métricas derivadas dos dados"""
        metrics = {}
        try:
            # Métricas de despesas
            if self.data_loader.despesas_df is not None:
                df_desp = self.data_loader.despesas_df
                metrics['media_mensal_despesas'] = df_desp.groupby(
                    pd.Grouper(key='dataDocumento', freq='M')
                )['valorDocumento'].mean()
                
                # Tendência linear simples
                values = metrics['media_mensal_despesas'].values
                if len(values) > 1:
                    x = np.arange(len(values))
                    coeffs = np.polyfit(x, values, 1)
                    metrics['tendencia_despesas'] = coeffs[0]
                
            # Métricas de proposições
            if self.data_loader.proposicoes_df is not None:
                df_prop = self.data_loader.proposicoes_df
                total_proposicoes = len(df_prop)
                aprovadas = len(df_prop[df_prop['status'] == 'APROVADA'])
                metrics['taxa_aprovacao'] = aprovadas / total_proposicoes if total_proposicoes > 0 else 0
                
            # Métricas de deputados
            if self.data_loader.deputados_df is not None:
                df_dep = self.data_loader.deputados_df
                metrics['distribuicao_partidos'] = df_dep['siglaPartido'].value_counts()
                
            return metrics
        except Exception as e:
            logger.error(f"Erro no cálculo de métricas derivadas: {str(e)}")
            return {}

    def find_correlations(self) -> Dict[str, float]:
        """Encontra correlações importantes entre diferentes métricas"""
        correlations = {}
        try:
            if self.data_loader.despesas_df is not None and self.data_loader.proposicoes_df is not None:
                # Correlação entre gastos e número de proposições por deputado
                despesas_por_deputado = self.data_loader.despesas_df.groupby('idDeputado')['valorDocumento'].sum()
                proposicoes_por_deputado = self.data_loader.proposicoes_df.groupby('idDeputado').size()
                
                # Merge dos dados
                df_correlacao = pd.DataFrame({
                    'despesas': despesas_por_deputado,
                    'proposicoes': proposicoes_por_deputado
                }).fillna(0)
                
                if not df_correlacao.empty and len(df_correlacao) > 1:
                    correlations['despesas_vs_proposicoes'] = df_correlacao.corr().iloc[0,1]
                
            return correlations
        except Exception as e:
            logger.error(f"Erro no cálculo de correlações: {str(e)}")
            return {}

    def generate_insights(self) -> List[Insight]:
        """Gera insights baseados nas análises"""
        try:
            # Limpa insights anteriores
            self.insights = []
            
            # 1. Análise de Despesas
            if self.data_loader.despesas_df is not None:
                df_desp = self.data_loader.despesas_df
                outliers = self.detect_outliers(df_desp, 'valorDocumento')
                
                if len(outliers) > 0:
                    self.insights.append(Insight(
                        title="Despesas Atípicas Detectadas",
                        description=f"Foram identificadas {len(outliers)} despesas com valores significativamente diferentes do padrão.",
                        importance=4,
                        category="Despesas",
                        metrics={'quantidade_outliers': len(outliers),
                                'valor_medio_outliers': outliers['valorDocumento'].mean()},
                        recommendations=[
                            "Investigar as despesas atípicas para garantir sua legitimidade",
                            "Estabelecer alertas automáticos para valores acima do padrão"
                        ]
                    ))

            # 2. Análise de Proposições
            if self.data_loader.proposicoes_df is not None:
                df_prop = self.data_loader.proposicoes_df
                total = len(df_prop)
                if total > 0:
                    aprovadas = len(df_prop[df_prop['status'] == 'APROVADA'])
                    taxa_aprovacao = aprovadas / total
                    
                    self.insights.append(Insight(
                        title="Efetividade Legislativa",
                        description=f"A taxa de aprovação de proposições é de {taxa_aprovacao:.1%}",
                        importance=3,
                        category="Proposições",
                        metrics={'taxa_aprovacao': taxa_aprovacao},
                        recommendations=[
                            "Focar em áreas com maior taxa de aprovação",
                            "Analisar fatores que contribuem para aprovação"
                        ]
                    ))

            # 3. Análise de Correlações
            correlations = self.find_correlations()
            if correlations:
                self.insights.append(Insight(
                    title="Relação entre Gastos e Produtividade",
                    description=f"Correlação entre despesas e número de proposições: {correlations.get('despesas_vs_proposicoes', 0):.2f}",
                    importance=5,
                    category="Correlações",
                    metrics=correlations,
                    recommendations=[
                        "Otimizar alocação de recursos baseado na produtividade",
                        "Investigar casos de alta despesa e baixa produtividade"
                    ]
                ))

            return sorted(self.insights, key=lambda x: x.importance, reverse=True)
        
        except Exception as e:
            logger.error(f"Erro na geração de insights: {str(e)}")
            return []

    def get_top_insights(self, n: int = 5) -> List[Insight]:
        """Retorna os N insights mais importantes"""
        return sorted(self.insights, key=lambda x: x.importance, reverse=True)[:n]
