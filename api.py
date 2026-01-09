import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import os
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==========================
# CONFIGURAÇÕES TÉCNICAS
# ==========================
WINDOW_SIZE = 60
MODEL_PATH = "modelo_lstm_acoes.h5"
SCALER_PATH = "scaler.pkl"

# ==========================
# FUNÇÕES DE LÓGICA
# ==========================

def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        return None
    return df[['Close']].dropna()

def train_model(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled_data)):
        X.append(scaled_data[i-WINDOW_SIZE:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Divisão para Avaliação (80% treino, 20% teste) [cite: 28, 29]
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)

    # Cálculo de Métricas 
    predictions = model.predict(X_test)
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    preds_unscaled = scaler.inverse_transform(predictions)

    mae = mean_absolute_error(y_test_unscaled, preds_unscaled)
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, preds_unscaled))
    mape = np.mean(np.abs((y_test_unscaled - preds_unscaled) / y_test_unscaled)) * 100

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    return mae, rmse, mape, y_test_unscaled, preds_unscaled

# ==========================
# INTERFACE STREAMLIT (DASHBOARD)
# ==========================

st.set_page_config(page_title="Tech Challenge Fase 4", layout="wide")

st.title("📊 Dashboard de Engenharia de Machine Learning")
st.markdown("Previsão de Preços de Ações utilizando Redes Neurais Recorrentes (LSTM)")

# Sidebar para inputs [cite: 17, 21]
st.sidebar.header("Configurações")
ticker = st.sidebar.text_input("Ticker da Ação", value="PETR4.SA")
btn_treinar = st.sidebar.button("Treinar e Avaliar Modelo")

if btn_treinar:
    with st.spinner("Coletando dados e treinando modelo..."):
        df = load_data(ticker, "2018-01-01", datetime.now().strftime("%Y-%m-%d"))
        if df is not None:
            mae, rmse, mape, y_real, y_pred = train_model(df)
            
            # --- SEÇÃO DE MÉTRICAS --- 
            st.subheader("🎯 Métricas de Avaliação")
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE (Erro Médio Absoluto)", f"{mae:.2f}")
            c2.metric("RMSE (Raiz do Erro Quadrático)", f"{rmse:.2f}")
            c3.metric("MAPE (Erro Percentual)", f"{mape:.2f}%")

            # --- GRÁFICO DE PERFORMANCE ---
            st.subheader("📈 Performance do Modelo (Dados de Teste)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_real.flatten(), name="Preço Real", line=dict(color='Black')))
            fig.add_trace(go.Scatter(y=y_pred.flatten(), name="Previsão", line=dict(color='#00ffcc', dash='dash')))
            fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("Modelo pronto para inferência!")
        else:
            st.error("Erro ao baixar dados. Verifique o Ticker.")

# --- SEÇÃO DE PREDIÇÃO FUTURA --- [cite: 34]
st.divider()
st.subheader("🔮 Predição de Fechamento")
if st.button("Prever valor para amanhã"):
    if os.path.exists(MODEL_PATH):
        # Pega os últimos 60 dias reais
        data_recent = yf.download(ticker, period="3mo")['Close'].tail(WINDOW_SIZE).values.reshape(-1, 1)
        
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        scaled_input = scaler.transform(data_recent).reshape(1, WINDOW_SIZE, 1)
        pred_scaled = model.predict(scaled_input)
        price_final = scaler.inverse_transform(pred_scaled)[0][0]

        st.info(f"O valor previsto para o próximo fechamento de **{ticker}** é: **R$ {price_final:.2f}**")
    else:
        st.warning("Por favor, realize o treinamento do modelo primeiro.")