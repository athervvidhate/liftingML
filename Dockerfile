FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install uv
RUN uv pip install --system torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache-dir --system -r requirements.txt

COPY streamlit_app.py .
COPY CustomSentenceEmbedder.py .

RUN mkdir -p data albert_finetuned

COPY data/cleaned_600k.csv ./data/
COPY data/program_features.csv ./data/
COPY data/final_features_albert.csv ./data/

COPY albert_finetuned/ ./albert_finetuned/

RUN useradd -m -u 1000 streamlit && \
    chown -R streamlit:streamlit /app
USER streamlit

EXPOSE 8080

ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_SERVER_ENABLE_WEBSOCKET=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false


CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]