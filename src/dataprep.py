import requests
import pandas as pd
from pathlib import Path
import logging
from config import API_BASE_URL, GOOGLE_API_KEY
import json
import google.generativeai as genai
import matplotlib.pyplot as plt

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar Gemini
genai.configure(api_key=GOOGLE_API_KEY)

def coletar_despesas_deputados(df_deputados):
    """
    Coleta as despesas dos deputados para agosto de 2024.
    
    Args:
        df_deputados (pd.DataFrame): DataFrame com os dados dos deputados
    
    Returns:
        pd.DataFrame: DataFrame com as despesas agrupadas
    """
    try:
        logger.info("Iniciando coleta de despesas dos deputados...")
        
        # Lista para armazenar todas as despesas
        todas_despesas = []
        
        # Parâmetros para agosto de 2024
        params = {
            'ano': [2024],  # Lista com o ano
            'mes': [8],     # Lista com o mês
            'ordenarPor': 'dataDocumento',
            'ordem': 'ASC',
            'itens': 100    # Máximo permitido pela API
        }
        
        # Iterar sobre cada deputado
        total_deputados = len(df_deputados)
        for idx, deputado in df_deputados.iterrows():
            id_deputado = deputado['id']
            nome_deputado = deputado['nome']
            
            logger.info(f"\nProcessando deputado {idx+1}/{total_deputados}: {nome_deputado} (ID: {id_deputado})")
            
            # URL para despesas do deputado
            url = f"{API_BASE_URL}deputados/{id_deputado}/despesas"
            
            try:
                logger.info(f"Fazendo requisição para: {url}")
                logger.info(f"Parâmetros: {params}")
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    dados = response.json()
                    despesas = dados.get('dados', [])
                    
                    if despesas:
                        # Adicionar informações do deputado em cada despesa
                        for despesa in despesas:
                            despesa['idDeputado'] = id_deputado
                            despesa['nomeDeputado'] = nome_deputado
                            despesa['siglaPartido'] = deputado['siglaPartido']
                            todas_despesas.append(despesa)
                            
                        logger.info(f"Coletadas {len(despesas)} despesas")
                    else:
                        logger.info("Nenhuma despesa encontrada")
                        
                else:
                    logger.error(f"Erro na requisição: Status {response.status_code}")
                    logger.error(f"Resposta: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Erro ao coletar despesas do deputado {id_deputado}: {e}")
                continue
                
        # Converter para DataFrame
        if not todas_despesas:
            logger.warning("Nenhuma despesa encontrada para o período especificado")
            return pd.DataFrame()
            
        df_despesas = pd.DataFrame(todas_despesas)
        
        # Converter coluna de data
        if 'dataDocumento' in df_despesas.columns:
            df_despesas['dataDocumento'] = pd.to_datetime(df_despesas['dataDocumento'])
        
        # Agrupar por dia, deputado e tipo de despesa
        colunas_agrupamento = ['dataDocumento', 'nomeDeputado', 'tipoDespesa']
        colunas_disponiveis = [col for col in colunas_agrupamento if col in df_despesas.columns]
        
        df_agrupado = df_despesas.groupby(colunas_disponiveis).agg({
            'valorDocumento': 'sum',
            'idDeputado': 'first',
            'siglaPartido': 'first'
        }).reset_index()
        
        # Salvar em parquet
        output_path = "data/processed/serie_despesas_diarias_deputados.parquet"
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        df_agrupado.to_parquet(output_path, index=False)
        
        logger.info(f"\nDados de despesas salvos com sucesso em {output_path}")
        logger.info(f"Total de registros agrupados: {len(df_agrupado)}")
        
        # Mostrar uma amostra dos dados
        if not df_agrupado.empty:
            logger.info("\nAmostra dos dados coletados:")
            logger.info(f"\nColunas disponíveis: {df_agrupado.columns.tolist()}")
            logger.info("\nPrimeiros registros:")
            logger.info(df_agrupado.head().to_string())
        
        return df_agrupado
        
    except Exception as e:
        logger.error(f"Erro inesperado ao coletar despesas: {e}")
        raise

def coletar_dados_deputados():
    """
    Coleta dados dos deputados atuais da Câmara dos Deputados
    e salva em formato parquet.
    """
    try:
        # Criar diretório se não existir
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        
        # URL para deputados em exercício
        url = f"{API_BASE_URL}deputados"
        
        # Parâmetros da requisição
        params = {
            'ordem': 'ASC',
            'ordenarPor': 'nome',
            'itens': 513  # Número total de deputados
        }
        
        logger.info("Iniciando coleta de dados dos deputados...")
        
        # Fazer requisição à API
        response = requests.get(url, params=params)
        response.raise_for_status()  # Levanta exceção para erros HTTP
        
        # Converter resposta para DataFrame
        dados = response.json()['dados']
        df_deputados = pd.DataFrame(dados)
        
        # Salvar em formato parquet
        output_path = "data/processed/deputados.parquet"
        df_deputados.to_parquet(output_path, index=False)
        
        logger.info(f"Dados salvos com sucesso em {output_path}")
        logger.info(f"Total de deputados coletados: {len(df_deputados)}")
        
        return df_deputados
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro na requisição à API: {e}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        raise

def gerar_grafico_distribuicao_partidos(df):
    """
    Gera um gráfico de pizza mostrando a distribuição de deputados por partido.
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados dos deputados
    
    Returns:
        tuple: (distribuicao, percentuais) onde distribuicao é a Series com contagem
               por partido e percentuais são os valores percentuais correspondentes
    """
    try:
        logger.info("Gerando gráfico de distribuição por partido...")
        
        # Calcular distribuição
        distribuicao = df['siglaPartido'].value_counts()
        total_deputados = len(df)
        percentuais = (distribuicao / total_deputados * 100).round(2)
        
        # Configurar o gráfico
        plt.figure(figsize=(15, 10))
        plt.pie(distribuicao, labels=distribuicao.index, autopct='%1.1f%%')
        plt.title('Distribuição de Deputados por Partido')
        
        # Ajustar layout
        plt.axis('equal')
        
        # Salvar gráfico
        output_path = "data/processed/distribuicao_partidos.png"
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico salvo com sucesso em {output_path}")
        
        return distribuicao, percentuais.values
        
    except Exception as e:
        logger.error(f"Erro ao gerar gráfico: {e}")
        raise

def gerar_insights_distribuicao(distribuicao, percentuais):
    """
    Gera insights sobre a distribuição de deputados por partido usando Gemini.
    """
    try:
        logger.info("Gerando insights sobre a distribuição dos partidos...")
        
        # Configurar o modelo Gemini
        model = genai.GenerativeModel('gemini-pro')
        
        # Preparar os dados para o prompt
        dados_partidos = [f"{partido}: {count} deputados ({perc}%)"
                         for partido, count, perc 
                         in zip(distribuicao.index, distribuicao.values, percentuais)]
        
        # Criar o prompt
        prompt = f"""
        Analise a seguinte distribuição de deputados por partido na Câmara dos Deputados do Brasil:

        {'\n'.join(dados_partidos)}

        Gere uma análise em formato JSON com a seguinte estrutura exata:
        {{
            "insights": [
                "insight 1",
                "insight 2",
                "insight 3",
                "insight 4",
                "insight 5"
            ],
            "analise_geral": "texto com análise geral em 2-3 parágrafos",
            "recomendacoes": [
                "recomendação 1",
                "recomendação 2",
                "recomendação 3"
            ]
        }}

        Mantenha um tom analítico e imparcial, focando em:
        1. Concentração/fragmentação do poder
        2. Principais blocos partidários
        3. Implicações para governabilidade
        4. Representatividade partidária

        IMPORTANTE: Responda APENAS com o JSON, sem texto adicional.
        """
        
        # Gerar insights
        response = model.generate_content(prompt)
        
        try:
            # Tentar fazer parse do JSON diretamente
            logger.info("Resposta do modelo:")
            logger.info(response.text)
            
            # Limpar a resposta antes de tentar fazer o parse
            cleaned_response = response.text.strip()
            # Remover possíveis caracteres de formatação
            cleaned_response = cleaned_response.replace('\n', ' ').replace('\r', '')
            # Remover possíveis textos antes ou depois do JSON
            start_idx = cleaned_response.find('{')
            end_idx = cleaned_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = cleaned_response[start_idx:end_idx]
                try:
                    insights = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.error(f"Erro no parsing do JSON limpo: {e}")
                    logger.error(f"JSON que falhou: {json_str}")
                    # Criar estrutura básica em caso de erro
                    insights = {
                        "insights": ["Erro ao processar insights do modelo"],
                        "analise_geral": "Não foi possível gerar a análise devido a um erro de formato.",
                        "recomendacoes": ["Tente executar novamente"]
                    }
            else:
                logger.error("Não foi possível encontrar JSON válido na resposta")
                insights = {
                    "insights": ["Erro ao processar insights do modelo"],
                    "analise_geral": "Não foi possível gerar a análise devido a um erro de formato.",
                    "recomendacoes": ["Tente executar novamente"]
                }
        
        except Exception as e:
            logger.error(f"Erro ao processar resposta do modelo: {e}")
            insights = {
                "insights": ["Erro ao processar insights do modelo"],
                "analise_geral": "Não foi possível gerar a análise devido a um erro de formato.",
                "recomendacoes": ["Tente executar novamente"]
            }
        
        # Salvar insights
        output_path = "data/processed/insights_distribuicao_deputados.json"
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(insights, f, ensure_ascii=False, indent=4)
            
        logger.info(f"Insights salvos com sucesso em {output_path}")
        return insights
        
    except Exception as e:
        logger.error(f"Erro ao gerar insights: {e}")
        raise

def coletar_proposicoes():
    """
    Coleta proposições que tramitam em agosto/2024 para os temas específicos.
    
    Temas coletados:
    - Economia (código 40)
    - Educação (código 46)
    - Ciência, Tecnologia e Inovação (código 62)
    
    Returns:
        pd.DataFrame: DataFrame com as proposições coletadas
    """
    try:
        logger.info("Iniciando coleta de proposições...")
        
        # Temas e seus códigos
        temas = {
            'Economia': 40,
            'Educação': 46,
            'Ciência, Tecnologia e Inovação': 62
        }
        
        # Lista para armazenar todas as proposições
        todas_proposicoes = []
        
        # Parâmetros base
        params = {
            'dataInicio': '2024-08-01',
            'dataFim': '2024-08-30',
            'ordem': 'ASC',
            'ordenarPor': 'id',
            'itens': 10  # 10 proposições por tema
        }
        
        # Coletar proposições para cada tema
        for tema, codigo in temas.items():
            logger.info(f"\nColetando proposições do tema: {tema} (código: {codigo})")
            
            # Adicionar código do tema aos parâmetros
            params_tema = params.copy()
            params_tema['codTema'] = codigo
            
            # URL para proposições
            url = f"{API_BASE_URL}proposicoes"
            
            try:
                logger.info(f"Fazendo requisição para: {url}")
                logger.info(f"Parâmetros: {params_tema}")
                
                response = requests.get(url, params=params_tema)
                
                if response.status_code == 200:
                    dados = response.json()
                    proposicoes = dados.get('dados', [])
                    
                    if proposicoes:
                        # Adicionar tema a cada proposição
                        for prop in proposicoes:
                            prop['tema'] = tema
                            prop['codTema'] = codigo
                            todas_proposicoes.append(prop)
                            
                        logger.info(f"Coletadas {len(proposicoes)} proposições")
                    else:
                        logger.warning(f"Nenhuma proposição encontrada para o tema {tema}")
                else:
                    logger.error(f"Erro na requisição: Status {response.status_code}")
                    logger.error(f"Resposta: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Erro ao coletar proposições do tema {tema}: {e}")
                continue
        
        if not todas_proposicoes:
            logger.error("Nenhuma proposição foi coletada")
            return pd.DataFrame()
            
        # Converter para DataFrame
        df_proposicoes = pd.DataFrame(todas_proposicoes)
        
        # Salvar em parquet
        output_path = "data/processed/proposicoes_deputados.parquet"
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        df_proposicoes.to_parquet(output_path, index=False)
        
        logger.info(f"\nDados salvos com sucesso em {output_path}")
        logger.info(f"Total de proposições coletadas: {len(df_proposicoes)}")
        
        # Mostrar uma amostra dos dados
        if not df_proposicoes.empty:
            logger.info("\nAmostra dos dados coletados:")
            logger.info(f"\nColunas disponíveis: {df_proposicoes.columns.tolist()}")
            logger.info("\nPrimeiros registros:")
            logger.info(df_proposicoes.head().to_string())
        
        return df_proposicoes
        
    except Exception as e:
        logger.error(f"Erro inesperado ao coletar proposições: {e}")
        raise

def visualizar_dados_parquet(caminho_arquivo):
    """
    Carrega e exibe os dados do arquivo parquet.
    
    Args:
        caminho_arquivo (str): Caminho para o arquivo parquet
    """
    try:
        logger.info(f"Carregando dados do arquivo {caminho_arquivo}...")
        df = pd.read_parquet(caminho_arquivo)
        return df
    
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        raise

def _processar_chunk(proposicoes_chunk, tema):
    """
    Processa um chunk de proposições usando o modelo Gemini.
    
    Args:
        proposicoes_chunk (pd.DataFrame): Chunk de proposições para processar
        tema (str): Tema das proposições
    
    Returns:
        dict: Dicionário com o resumo do chunk
    """
    try:
        # Log do início do processamento
        logger.info(f"\nProcessando chunk de {len(proposicoes_chunk)} proposições do tema {tema}")
        
        # Formata as proposições para o prompt
        proposicoes_texto = []
        for _, row in proposicoes_chunk.iterrows():
            prop_texto = f"ID: {row['id']}\n"
            prop_texto += f"Tipo: {row['siglaTipo']} {row['numero']}/{row['ano']}\n"
            prop_texto += f"Ementa: {row['ementa']}\n"
            proposicoes_texto.append(prop_texto)
        
        # Monta o prompt para o Gemini
        prompt = f"""Analise as seguintes proposições legislativas e forneça um resumo estruturado.

CONTEXTO: Estas são proposições legislativas sobre {tema}.

PROPOSIÇÕES:
{'\n'.join(proposicoes_texto)}

INSTRUÇÕES:
Retorne APENAS um objeto JSON com a seguinte estrutura (sem markdown, sem texto adicional):
{{
    "resumo": "breve resumo das principais propostas",
    "temas": ["tema1", "tema2", "tema3"],
    "destaques": [
        {{"id": "ID_DA_PROPOSICAO", "resumo": "resumo desta proposição"}}
    ]
}}"""
        
        # Configura e chama o modelo Gemini
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Log da resposta completa
        logger.info(f"Resposta do Gemini:\n{response.text}")
        
        # Tenta processar a resposta
        try:
            json_str = response.text.strip()
            
            # Remove marcadores de código markdown se presentes
            if json_str.startswith('```'):
                json_str = json_str.split('```')[1]  # Pega o conteúdo entre os marcadores
                if json_str.startswith('json'):
                    json_str = json_str[4:]  # Remove 'json'
                json_str = json_str.strip()
            
            resultado = json.loads(json_str)
            logger.info("JSON processado com sucesso")
            return resultado
            
        except json.JSONDecodeError as je:
            logger.error(f"Erro ao decodificar JSON: {str(je)}\nTexto recebido: {json_str[:200]}")
            return None
            
    except Exception as e:
        logger.error(f"Erro inesperado ao processar chunk: {str(e)}")
        return None

def sumarizar_proposicoes(df_proposicoes):
    """
    Sumariza as proposições coletadas usando o modelo Gemini.
    
    Args:
        df_proposicoes (pd.DataFrame): DataFrame com as proposições coletadas
    
    Returns:
        dict: Dicionário com os resumos por tema
    """
    logger.info(f"Iniciando sumarização de {len(df_proposicoes)} proposições...")
    
    # Dicionário para armazenar resultados
    resultados = {}
    
    # Processa cada tema
    for tema in df_proposicoes['tema'].unique():
        logger.info(f"\nProcessando tema: {tema}")
        proposicoes_tema = df_proposicoes[df_proposicoes['tema'] == tema]
        logger.info(f"Total de proposições para {tema}: {len(proposicoes_tema)}")
        
        # Lista para armazenar resultados do tema
        resultados_tema = []
        
        # Processa em chunks de 5 proposições
        for i in range(0, len(proposicoes_tema), 5):
            chunk = proposicoes_tema.iloc[i:i+5]
            logger.info(f"Processando chunk {i//5 + 1} de {(len(proposicoes_tema) + 4)//5}")
            
            resultado_chunk = _processar_chunk(chunk, tema)
            if resultado_chunk:
                resultados_tema.append(resultado_chunk)
                logger.info("Chunk processado com sucesso")
            else:
                logger.warning("Falha ao processar chunk")
        
        # Consolida resultados do tema
        if resultados_tema:
            logger.info(f"Consolidando {len(resultados_tema)} resultados para {tema}")
            
            # Combina todos os temas únicos
            todos_temas = list(set(
                tema for resultado in resultados_tema 
                for tema in resultado.get('temas', [])
            ))
            
            # Combina todos os destaques
            todos_destaques = [
                destaque for resultado in resultados_tema 
                for destaque in resultado.get('destaques', [])
            ]
            
            # Consolida resumos
            resumo_geral = " ".join(
                resultado.get('resumo', '') for resultado in resultados_tema
            )
            
            # Estrutura final do tema
            resultados[tema] = {
                "resumo_geral": resumo_geral,
                "temas_principais": todos_temas[:5],
                "proposicoes_destaque": todos_destaques
            }
            logger.info(f"Tema {tema} processado com sucesso")
        else:
            logger.warning(f"Nenhum resultado válido para o tema {tema}")
    
    # Salva resultados em JSON
    caminho_arquivo = "data/sumarizacao_proposicoes.json"
    try:
        with open(caminho_arquivo, 'w', encoding='utf-8') as f:
            json.dump(resultados, f, ensure_ascii=False, indent=4)
        logger.info(f"Sumarização das proposições salva em {caminho_arquivo}")
    except Exception as e:
        logger.error(f"Erro ao salvar sumarização: {str(e)}")
    
    return resultados

if __name__ == "__main__":
    # Executar pipeline completo
    
    # 1. Coletar e analisar dados dos deputados
    print("\nIniciando coleta de dados dos deputados...")
    df_deputados = coletar_dados_deputados()
    
    if not df_deputados.empty:
        print("\nGerando gráfico de distribuição por partido...")
        distribuicao, percentuais = gerar_grafico_distribuicao_partidos(df_deputados)
        
        print("\nGerando insights sobre a distribuição...")
        gerar_insights_distribuicao(distribuicao, percentuais)
    
    # 2. Coletar e analisar despesas
    print("\nIniciando coleta de despesas...")
    df_despesas = coletar_despesas_deputados(df_deputados)
    
    if not df_despesas.empty:
        print("\nAmostra das despesas coletadas:")
        print("\nColunas disponíveis:", df_despesas.columns.tolist())
        print("\nPrimeiros 5 registros:")
        print(df_despesas.head().to_string())
        print(f"\nTotal de registros: {len(df_despesas)}")
        
    # 3. Coletar e sumarizar proposições
    print("\nIniciando coleta de proposições...")
    df_proposicoes = coletar_proposicoes()
    
    if not df_proposicoes.empty:
        print("\nIniciando sumarização das proposições...")
        resultado_sumarizacao = sumarizar_proposicoes(df_proposicoes)
        
        print("\nSumarização concluída. Resultados:")
        for sumarizacao in resultado_sumarizacao.values():
            print(f"\nTema: {sumarizacao['temas_principais']}")
            print(f"Resumo Geral: {sumarizacao['resumo_geral']}")
            print(f"Proposições em Destaque: {sumarizacao['proposicoes_destaque']}")