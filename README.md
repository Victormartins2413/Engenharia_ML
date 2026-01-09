📈 Tech Challenge - Fase 4: Predição de Ativos com LSTM
Este repositório contém a solução para o Tech Challenge da Fase 4. O objetivo é prever o preço de fechamento de ações utilizando Redes Neurais Profundas, integrando desde a coleta de dados até o deploy conteinerizado.

🛠️ Tecnologias Utilizadas
Python 3.11

TensorFlow/Keras: Para a criação da Rede Neural LSTM.

Streamlit: Para a interface de usuário e monitoramento.

YFinance: Para coleta de dados em tempo real.

Docker: Para garantir a portabilidade e escalabilidade.

Scikit-Learn: Para pré-processamento e métricas.

🧠 Fundamentos do Projeto
1. Coleta e Pré-processamento
O sistema utiliza a biblioteca yfinance para baixar dados históricos. Aplicamos o MinMaxScaler para normalizar os dados entre 0 e 1, o que é essencial para o funcionamento de redes neurais, evitando que a magnitude dos preços (ex: R$ 100,00 vs R$ 1,00) cause instabilidade no treinamento.

Trabalhamos com uma Janela Temporal (Sliding Window) de 60 dias: o modelo aprende que a sequência dos últimos 60 dias resulta no preço do dia seguinte.

2. Arquitetura do Modelo (LSTM)
A escolha da LSTM (Long Short-Term Memory) é justificada por sua capacidade de manter informações de longo prazo, ideal para o mercado financeiro onde tendências passadas influenciam o futuro. A arquitetura conta com camadas de Dropout para prevenir o overfitting (quando o modelo decora os dados mas não consegue generalizar).

3. Métricas de Avaliação
O modelo é avaliado com três métricas principais exibidas no dashboard:

MAE (Mean Absolute Error): Indica o erro médio em termos monetários.

RMSE (Root Mean Square Error): Penaliza erros maiores, indicando a confiabilidade.

MAPE (Mean Absolute Percentage Error): Mostra o erro em porcentagem (ex: "o modelo erra em média 2%").

4. Deploy e Escalabilidade (Docker)
Para cumprir os requisitos de produção, o projeto foi empacotado em um Contêiner Docker. Isso garante que a aplicação rode exatamente da mesma forma no seu computador ou em um servidor na nuvem.

🚀 Como Executar
Via Docker (Recomendado)
Construa a imagem:

Bash

docker build -t tech-challenge-lstm .
Execute o contêiner:

Bash

docker run -p 8501:8501 tech-challenge-lstm
Acesse: http://localhost:8501

Via Python Local
Instale as dependências: pip install -r requirements.txt

Execute o Streamlit: streamlit run api.py

📊 Monitoramento e Interface
A interface do Streamlit permite monitorar o desempenho do modelo através de um gráfico comparativo entre o Preço Real e a Previsão. Se as linhas divergirem drasticamente, o usuário pode utilizar o botão de "Treinar" para atualizar o modelo com dados mais recentes.