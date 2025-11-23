# Imagen base ligera
FROM python:3.10-slim

WORKDIR /app

# Instalamos dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el script principal
COPY app.py /app/app.py

# Punto de entrada por defecto: API
ENTRYPOINT ["python", "app.py"]
