# Use uma imagem base Python slim para reduzir o tamanho
FROM python:3.12-slim

# Definir o diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema necessárias para OpenCV e outras bibliotecas
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    libgtk-3-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo o código da aplicação
COPY . .

# Criar diretórios necessários
RUN mkdir -p Entrada Saida

# Definir variável de ambiente para Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=7000

# Expor a porta
EXPOSE 7000

# Adicionar healthcheck para verificar se a aplicação está funcionando
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7000/ || exit 1

# Comando para executar a aplicação
CMD ["python", "app.py"]
