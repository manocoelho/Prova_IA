# Projeto de Classificação de Sinais de Libras com Machine Learning

**Disciplina:** Paradigmas de aprendizagem de máquina

**Instituição:** Universidade Federal da Paraíba (UFPB)

**Autor:** [Antonio Rocha Lima Filho]

---

## 1. Visão Geral do Projeto

Este projeto tem como objetivo principal desenvolver e avaliar modelos de Machine Learning para a classificação de sinais da Língua Brasileira de Sinais (Libras). A partir de dados de keypoints (pontos-chave do corpo e mãos) extraídos de vídeos, o trabalho explora abordagens de aprendizagem supervisionada e não supervisionada para identificar e agrupar os diferentes sinais.

O projeto inclui o pré-processamento dos dados, a aplicação de múltiplos algoritmos, o uso de validação cruzada e a análise de métricas de desempenho.

## 2. Metodologia e Pipeline

O projeto foi estruturado em um notebook do Google Colab, seguindo um pipeline claro de processamento e modelagem.

### 2.1. Extração e Engenharia de Features

A etapa inicial foi a mais crítica, transformando dados brutos de arquivos JSON em um conjunto de features significativo.

* **Processamento de JSONs:** Os dados de cada vídeo, contidos em arquivos JSON separados
* **Normalização de Coordenadas:** As coordenadas `x` e `y` dos keypoints foram normalizadas pela largura e altura do vídeo. Essa etapa foi crucial para que o modelo aprendesse os padrões de movimento de forma independente da resolução do vídeo ou do enquadramento da câmera.
* **Agregação Estatística:** Para cada vídeo, as séries temporais das coordenadas de cada keypoint foram resumidas em métricas estatísticas (média, desvio padrão, mínimo e máximo).
* **Features de Movimento Relativo:** Reconhecendo que sinais são definidos por movimento, foram criadas features mais robustas que calculam a **distância euclidiana** entre pontos-chave (nariz e pulsos) ao longo do tempo. As estatísticas dessas distâncias (ex: média e variação) foram adicionadas como features, capturando a dinâmica espacial dos gestos.

### 2.2. Pré-processamento

Antes da modelagem, o dataset passou por uma rigorosa etapa de preparação:

* **Tratamento de Outliers:** Foi aplicado o método do **Intervalo Interquartil (IQR)** para identificar e remover amostras com valores discrepantes, tornando o modelo mais robusto. A técnica foi aplicada após a divisão dos dados para evitar *data leakage*.
* **Codificação de Variáveis:** O `LabelEncoder` foi utilizado para converter os rótulos dos sinais (target) e os nomes dos intérpretes (grupos) para formato numérico.
* **Divisão em Treino e Teste:** O conjunto de dados foi dividido em 80% para treino e 20% para teste, utilizando a estratificação (`stratify`) para manter a proporção de classes no conjunto de teste.
* **Padronização de Features:** O `StandardScaler` foi aplicado para normalizar a escala de todas as features, garantindo que algoritmos baseados em distância (como k-NN) e em gradiente (como MLP) funcionassem de forma otimizada.

### 2.3. Modelagem Supervisionada

Foram testados os três algoritmos solicitados: Random Forest, k-NN e MLP.

* **Validação Cruzada por Grupo (`GroupKFold`):** Esta foi a decisão metodológica central do projeto. Em vez de uma validação cruzada estratificada simples, optou-se pelo `GroupKFold`, utilizando o `interprete` como grupo. Esta abordagem garante que o modelo seja avaliado em sua capacidade de generalizar para sinais feitos por uma pessoa que ele nunca viu durante o treinamento, fornecendo uma estimativa de desempenho muito mais realista e evitando overfitting relacionado ao estilo de cada intérprete.
* **Otimização (`GridSearchCV`):** Para cada modelo, foi realizada uma busca exaustiva por hiperparâmetros para encontrar a melhor combinação com base no `F1-Score`.
* **Análise com PCA:** Um pipeline foi utilizado para avaliar o impacto da **Redução de Dimensionalidade (PCA)** no desempenho do melhor modelo, cumprindo um dos itens do critério de avaliação.

### 2.4. Modelagem Não Supervisionada

Foi explorada a capacidade dos algoritmos de encontrar agrupamentos naturais nos dados sem o uso de rótulo.

* **Algoritmos:** Foram aplicados o K-Means e o Cluster Hierárquico, este com dois métodos de `linkage` ('ward' e 'complete') para comparação.
* **Avaliação:**
    * O **Método do Cotovelo** foi utilizado para estimar o número ideal de clusters (`K`) para o K-Means.
    * O **Coeficiente de Silhueta** foi a métrica escolhida para avaliar numericamente a qualidade e a separação dos clusters encontrados.

## 3. Estrutura dos Arquivos

```
/
├── Prova_Pratica_Aprendizagem_De_Maquina.ipynb            # Notebook principal com todo o código e análise
├── sinais.csv                    # Arquivo de metadados que mapeia JSONs para rótulos
└── dados_json/                   # Pasta contendo todos os arquivos JSON com os keypoints
    ├── Adição_AP_10.json
    └── ...
```

## 4. Dados e Como Executar

Os dados utilizados neste projeto não estão incluídos diretamente neste repositório devido ao seu tamanho. Para executar o notebook, siga estes passos:

1.  **Faça o download dos dados:** Baixe a pasta completa do projeto (contendo o `sinais.csv` e a pasta `dados_json`) através do seguinte link do Google Drive:
    
    **[Link para os Dados do Projeto](https://drive.google.com/drive/folders/1MYO2dzO0P6Iz8cejSQMvzzdCLAvkqFBL?usp=sharing)**
    
2.  **Estrutura de Pastas:** Faça o upload do conteúdo baixado para o seu próprio Google Drive, garantindo que a estrutura de pastas seja a seguinte:
    
    ```
    /MyDrive/
    └── Projeto_IA_Libras/
        ├── sinais.csv
        └── dados_json/
            └── ... (todos os arquivos .json)
    ```
    
3.  **Execute o Notebook:** Abra o arquivo `.ipynb` no Google Colab e execute as células em sequência. O notebook irá montar seu Drive e encontrar os arquivos automaticamente.

## 5. Resultados e Conclusões

### 5.1. Classificação Supervisionada

A validação rigorosa com `GroupKFold` revelou scores realistas, mostrando que o **Random Forest foi o melhor modelo**, com um **F1-Score (macro) de aproximadamente 0.45** no conjunto de teste. Embora não seja um score perfeito, é um resultado muito significativo para um problema de 25 classes, demonstrando que as features de movimento criadas são eficazes. A análise da Matriz de Confusão e da Curva ROC permitiu uma análise detalhada dos pontos fortes e fracos do modelo.

### 5.2. Clusterização

A análise não supervisionada mostrou que a tarefa de agrupar os sinais sem rótulos é extremamente desafiadora. O melhor algoritmo (`Hierárquico com linkage 'ward'`) alcançou um **Coeficiente de Silhueta de apenas ~0.15**. Este valor baixo, confirmado pelas visualizações com PCA, indica que os clusters têm alta sobreposição, sugerindo que os movimentos entre diferentes sinais são muito sutis para serem separados sem um treinamento supervisionado.

### 5.3. Conclusão Geral

O projeto demonstrou com sucesso a viabilidade de classificar sinais de Libras a partir de dados de keypoints, ressaltando a importância de uma **engenharia de features robusta** e de uma **metodologia de validação rigorosa** (`GroupKFold`) para obter resultados confiáveis. Também concluiu que, embora classificáveis, os sinais possuem uma complexidade que dificulta sua separação por métodos não supervisionados.
