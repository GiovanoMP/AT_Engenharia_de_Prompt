# Guia do Usuário - Dashboard da Câmara dos Deputados

## Introdução

O Dashboard da Câmara dos Deputados é uma ferramenta interativa que permite visualizar e analisar dados sobre os deputados federais brasileiros. Este guia fornecerá todas as informações necessárias para utilizar o dashboard de forma efetiva.

## Instalação

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITORIO]
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute o dashboard:
```bash
python -m app.dashboard
```

## Funcionalidades

### 1. Visão Geral

A página principal do dashboard apresenta:

- **Descrição da Câmara**: Uma visão geral sobre o funcionamento da Câmara dos Deputados
- **Insights Principais**: Análises importantes sobre a distribuição dos deputados
- **Métricas Principais**: 
  - Total de Deputados
  - Média de Gastos
  - Total de Proposições
- **Distribuição por Partido**: Gráfico interativo mostrando a distribuição dos deputados por partido

### 2. Interatividade

O dashboard oferece várias formas de interação:

- **Gráficos Interativos**: 
  - Passe o mouse sobre as barras para ver detalhes
  - Use os controles de zoom para explorar partes específicas
  - Clique duplo para resetar a visualização

- **Tour Guiado**: 
  - Disponível para novos usuários
  - Explica as principais funcionalidades
  - Pode ser desativado após a primeira visualização

### 3. Performance

O dashboard foi otimizado para:

- Carregamento rápido de dados
- Cache eficiente
- Atualização dinâmica de visualizações

## Troubleshooting

### Problemas Comuns

1. **Dados não carregam**:
   - Verifique sua conexão com a internet
   - Recarregue a página
   - Confirme se os arquivos de dados existem no diretório correto

2. **Gráficos não aparecem**:
   - Limpe o cache do navegador
   - Verifique se JavaScript está habilitado
   - Tente usar outro navegador

3. **Erros de Performance**:
   - Feche outras abas do navegador
   - Recarregue a página
   - Verifique o uso de memória do sistema

### Contato e Suporte

Para reportar problemas ou sugerir melhorias:

- Abra uma issue no GitHub
- Entre em contato através do email de suporte
- Consulte a documentação técnica para mais detalhes

## Dicas de Uso

1. **Melhor Experiência**:
   - Use um navegador moderno (Chrome, Firefox, Edge)
   - Mantenha a janela em tela cheia para melhor visualização
   - Utilize um monitor com resolução mínima de 1366x768

2. **Análise de Dados**:
   - Explore os tooltips nos gráficos
   - Verifique as métricas principais regularmente
   - Leia os insights para entender tendências

3. **Atualizações**:
   - Verifique regularmente por novas versões
   - Leia as notas de atualização
   - Mantenha as dependências atualizadas

## Conclusão

Este dashboard foi desenvolvido para fornecer uma visão clara e interativa dos dados da Câmara dos Deputados. Use este guia como referência para aproveitar ao máximo todas as funcionalidades disponíveis.

Para mais informações técnicas, consulte a documentação de desenvolvimento no diretório `docs/`.
