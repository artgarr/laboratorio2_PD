FROM python:3.10-slim  

WORKDIR /app

# Instalamos dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el script
COPY train_model.py /app/train_model.py
COPY config.yaml /app/config.yaml

# Punto de entrada
ENTRYPOINT ["python", "train_model.py"]
