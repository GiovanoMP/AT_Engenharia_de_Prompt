# 🏛️ Assistente Virtual da Câmara dos Deputados

Sistema de análise e assistente virtual especializado em dados da Câmara dos Deputados do Brasil, utilizando processamento de linguagem natural avançado e técnicas de IA para fornecer insights e análises contextualizadas.

## 🎯 Visão Geral

O projeto implementa uma solução completa para análise de dados da Câmara dos Deputados, incluindo:
- Coleta e processamento de dados via API oficial
- Análises usando Large Language Models (LLMs)
- Base vetorial para consultas semânticas
- Interface interativa com dashboard e assistente virtual
- Geração de insights e visualizações

## 🚀 Funcionalidades Principais

### 1. Coleta e Processamento de Dados
- Integração com a API da Câmara dos Deputados
- Processamento de dados de deputados, despesas e proposições
- Geração de embeddings usando BERT português
- Armazenamento otimizado em base vetorial FAISS

### 2. Análise e Insights
- Sumarização de proposições legislativas
- Análise de despesas e gastos
- Processamento de textos legislativos
- Geração de insights usando LLMs

### 3. Interface do Usuário
- Dashboard interativo com Streamlit
- Visualizações e gráficos dinâmicos
- Assistente virtual especializado
- Sistema de busca semântica

### 4. Assistente Virtual
- Técnica Self-Ask para respostas estruturadas
- Análise hierárquica de informações
- Busca semântica em base vetorial
- Respostas contextualizadas

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**: Linguagem principal
- **Streamlit**: Interface do usuário
- **FAISS**: Base vetorial para busca semântica
- **BERT**: Modelo "neuralmind/bert-base-portuguese-cased"
- **Gemini**: LLM para geração de respostas
- **Pandas/Numpy**: Processamento de dados
- **Plotly**: Visualizações interativas

## 📁 Estrutura do Projeto

```
AT_eng_prompt_local/
├── app/
│   ├── pages/
│   │   └── 01_Assistente_Virtual.py
│   ├── utils/
│   │   ├── embeddings.py
│   │   ├── analytics.py
│   │   └── ...
│   └── Home.py
├── data/
│   ├── embeddings/
│   ├── raw/
│   └── processed/
├── docs/
│   └── project_charter/
└── requirements.txt
```

## 🚗 Como Executar

1. Clone o repositório
```bash
git clone [URL_DO_REPOSITÓRIO]
```

2. Instale as dependências
```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente
```bash
cp .env.example .env
# Edite .env com suas chaves de API
```

4. Execute o aplicativo
```bash
streamlit run app/Home.py
```

## 🔑 Requisitos

- Python 3.8+
- Chave API do Google (Gemini)
- Ambiente virtual Python (recomendado)
- Mínimo 8GB RAM

## 📊 Funcionalidades do Dashboard

1. **Visão Geral**
   - Indicadores principais
   - Resumo de atividades
   - Gráficos de tendências

2. **Análise de Despesas**
   - Visualização temporal
   - Filtros por deputado/partido
   - Análises comparativas

3. **Proposições**
   - Lista interativa
   - Sumarizações
   - Análises temáticas

4. **Assistente Virtual**
   - Chat interativo
   - Busca semântica
   - Análises contextualizadas

## 🤖 Capacidades do Assistente Virtual

O assistente utiliza a técnica Self-Ask para estruturar respostas em três níveis:

1. **Nível 1 - Visão Geral**
   - Análise de sumarizações
   - Identificação de temas principais
   - Contextualização inicial

2. **Nível 2 - Detalhamento**
   - Proposições específicas
   - Conexões entre temas
   - Análise aprofundada

3. **Nível 3 - Síntese**
   - Impactos potenciais
   - Tendências identificadas
   - Recomendações

## 📈 Performance e Limitações

### Pontos Fortes
- Excelente em análises temáticas
- Respostas bem estruturadas
- Contextualização eficiente

### Áreas de Melhoria
- Análises quantitativas específicas
- Dados estatísticos em tempo real
- Integração com mais fontes de dados


## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## 👥 Autore

- Giovano Montemezzo Panatta
