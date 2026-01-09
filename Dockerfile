# Usa uma imagem oficial do Python (leve)
FROM python:3.11-slim

# Define a pasta de trabalho dentro do contêiner
WORKDIR /app

# Copia os arquivos da sua pasta atual para dentro do contêiner
COPY . .

# Instala as bibliotecas do requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta que o Streamlit usa
EXPOSE 8501

# Comando para ligar o sistema assim que o contêiner iniciar
CMD ["streamlit", "run", "api.py", "--server.port=8501", "--server.address=0.0.0.0"]