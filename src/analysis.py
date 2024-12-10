import pandas as pd
import json
import logging
from pathlib import Path
import google.generativeai as genai
from config import GOOGLE_API_KEY

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def realizar_analises_despesas():
    """
    Realiza três análises diferentes sobre as despesas dos deputados usando prompt-chaining.
    Salva os resultados das análises e os insights gerados.
    """
    try:
        # Carregar dados
        logger.info("Carregando dados de despesas...")
        df = pd.read_parquet("data/processed/serie_despesas_diarias_deputados.parquet")
        
        # Primeira análise: Top 10 deputados com maiores gastos totais
        logger.info("Realizando análise 1: Top 10 deputados com maiores gastos...")
        analise1 = df.groupby('nomeDeputado').agg({
            'valorDocumento': 'sum',
            'siglaPartido': 'first'
        }).sort_values('valorDocumento', ascending=False).head(10)
        
        # Segunda análise: Distribuição de gastos por tipo de despesa
        logger.info("Realizando análise 2: Distribuição por tipo de despesa...")
        analise2 = df.groupby('tipoDespesa')['valorDocumento'].agg(['sum', 'count']).sort_values('sum', ascending=False)
        
        # Terceira análise: Média de gastos por partido
        logger.info("Realizando análise 3: Média de gastos por partido...")
        analise3 = df.groupby('siglaPartido').agg({
            'valorDocumento': ['mean', 'count']
        }).sort_values(('valorDocumento', 'mean'), ascending=False)
        
        # Converter DataFrames para formato compatível com JSON
        resultados = {
            "top_10_deputados": {
                "valores": analise1['valorDocumento'].to_dict(),
                "partidos": analise1['siglaPartido'].to_dict()
            },
            "distribuicao_despesas": {
                tipo: {
                    "soma": row['sum'],
                    "quantidade": row['count']
                } for tipo, row in analise2.iterrows()
            },
            "media_partido": {
                partido: {
                    "media": row[('valorDocumento', 'mean')],
                    "quantidade": row[('valorDocumento', 'count')]
                } for partido, row in analise3.iterrows()
            }
        }
        
        # Gerar prompt para insights
        prompt = f"""
        Analise os seguintes dados de despesas dos deputados federais brasileiros e gere insights relevantes:
        
        1. Top 10 Deputados com Maiores Gastos:
        {analise1.to_string()}
        
        2. Distribuição de Gastos por Tipo de Despesa:
        {analise2.to_string()}
        
        3. Média de Gastos por Partido:
        {analise3.to_string()}
        
        Por favor, gere insights significativos sobre estes dados, considerando:
        - Padrões de gastos entre deputados e partidos
        - Tipos de despesas mais significativas
        - Possíveis correlações entre partido e padrão de gastos
        - Quaisquer anomalias ou pontos de interesse
        
        Formate os insights como uma lista de observações.
        """
        
        # Gerar insights usando Gemini
        logger.info("Gerando insights com Gemini...")
        response = model.generate_content(prompt)
        insights = response.text
        
        # Criar estrutura final do JSON
        output = {
            "analises": resultados,
            "insights": insights
        }
        
        # Salvar resultados
        output_path = "data/processed/insights_despesas_deputados.json"
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
            
        logger.info(f"Resultados salvos com sucesso em {output_path}")
        return output
        
    except Exception as e:
        logger.error(f"Erro ao realizar análises: {e}")
        raise

if __name__ == "__main__":
    realizar_analises_despesas()
