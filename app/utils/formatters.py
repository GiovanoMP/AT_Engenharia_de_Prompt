"""
Módulo responsável pela formatação de dados e textos
"""

import streamlit as st
from typing import Any, List, Dict, Union
from datetime import datetime
import locale
from decimal import Decimal

# Configuração para formato brasileiro
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

class DataFormatter:
    """Classe para formatação de dados numéricos e datas"""
    
    @staticmethod
    def format_currency(value: Union[float, int, Decimal]) -> str:
        """Formata valor monetário no padrão brasileiro"""
        try:
            return locale.currency(value, grouping=True, symbol=True)
        except Exception:
            return f"R$ {value:,.2f}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """Formata percentual com número específico de casas decimais"""
        return f"{value:.{decimals}f}%"
    
    @staticmethod
    def format_date(date: datetime, format_str: str = "%d/%m/%Y") -> str:
        """Formata data no padrão brasileiro"""
        return date.strftime(format_str)
    
    @staticmethod
    def format_number(value: Union[int, float], decimal_places: int = 0) -> str:
        """Formata número com separadores de milhar"""
        try:
            return locale.format_string(f"%.{decimal_places}f", value, grouping=True)
        except Exception:
            return f"{value:,.{decimal_places}f}"

class TextFormatter:
    """Classe para formatação de textos e elementos visuais"""
    
    @staticmethod
    def format_markdown(text: str) -> str:
        """Formata texto para markdown"""
        return text.replace('\n', '\n\n')
    
    @staticmethod
    def create_expandable_section(title: str, content: str) -> None:
        """Cria uma seção expansível no Streamlit"""
        with st.expander(title):
            st.markdown(content)
    
    @staticmethod
    def format_insights(insights: List[Dict[str, Any]]) -> None:
        """Formata e exibe insights em cards"""
        for insight in insights:
            st.markdown("---")
            st.subheader(insight.get('titulo', 'Insight'))
            st.markdown(insight.get('descricao', ''))
            if 'metricas' in insight:
                cols = st.columns(len(insight['metricas']))
                for col, metrica in zip(cols, insight['metricas']):
                    with col:
                        st.metric(
                            label=metrica.get('nome', ''),
                            value=metrica.get('valor', ''),
                            delta=metrica.get('variacao', None)
                        )
    
    @staticmethod
    def create_info_card(title: str, content: str, icon: str = "ℹ️") -> None:
        """Cria um card informativo estilizado"""
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 0.5rem; background-color: #f0f2f6;">
            <h3>{icon} {title}</h3>
            <p>{content}</p>
        </div>
        """, unsafe_allow_html=True)

def format_currency(value: float, symbol: str = 'R$') -> str:
    """
    Formata um valor como moeda brasileira
    
    Args:
        value (float): Valor a ser formatado
        symbol (str, optional): Símbolo da moeda. Defaults to 'R$'.
    
    Returns:
        str: Valor formatado como moeda
    """
    try:
        return f"{symbol} {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return f"{symbol} 0,00"

def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Formata um valor como percentual
    
    Args:
        value (float): Valor a ser formatado
        decimals (int, optional): Número de casas decimais. Defaults to 1.
    
    Returns:
        str: Valor formatado como percentual
    """
    try:
        return f"{value:.{decimals}f}%".replace(".", ",")
    except (ValueError, TypeError):
        return "0%"

def format_date(date_str: str, input_format: str = "%Y-%m-%d", output_format: str = "%d/%m/%Y") -> str:
    """
    Formata uma data para o padrão brasileiro
    
    Args:
        date_str (str): String da data a ser formatada
        input_format (str, optional): Formato de entrada. Defaults to "%Y-%m-%d".
        output_format (str, optional): Formato de saída. Defaults to "%d/%m/%Y".
    
    Returns:
        str: Data formatada
    """
    try:
        date_obj = datetime.strptime(date_str, input_format)
        return date_obj.strftime(output_format)
    except (ValueError, TypeError):
        return date_str

def format_number(value: float, decimals: int = 0) -> str:
    """
    Formata um número com separadores de milhar
    
    Args:
        value (float): Valor a ser formatado
        decimals (int, optional): Número de casas decimais. Defaults to 0.
    
    Returns:
        str: Número formatado
    """
    try:
        if decimals == 0:
            return f"{int(value):,}".replace(",", ".")
        return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return "0"
