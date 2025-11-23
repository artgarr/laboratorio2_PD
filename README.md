# laboratorio2_PD

## Requisitos
Docker instalado

Python 3.10 dentro de las imágenes

Librerías: pandas==2.2.3, pyarrow==19.0.0, scikit-learn, joblib, fastapi, uvicorn

## Pasos para reproductibilidad

### 1. Clonar el proyecto 
git clone https://github.com/artgarr/laboratorio2_PD.git
cd laboratorio2_PD

### 2. Construir imágenes en Docker
docker build -t model-train:latest ./train
docker build -t model-serve:latest ./serve

### 3. Entrenar el modelo

docker run --rm `
  -v "$(pwd)/data:/data" `
  -v "$(pwd)/model:/output" `
  model-train:latest

Esto genera:

model/model.pkl → modelo entrenado

model/metrics.json → métricas del entrenamiento

### 4. Despliegue en producción

#### Modo API
docker run --rm -p 8000:8000 `
   -v "${PWD}/model:/app/model" `
   model-serve:latest

Esto genera el api en la direccion: http://localhost:8000

Con dos endpoints
GET /metrics → Genera la información relevante como 
 {
   "model": "RandomForest",
   "accuracy": 0.83,
   "predictions_served": 250,
   "avg_response_time_ms": 14.6,
   "last_training": "2025-11-08"
 }
 
GET /docs → Nos muestra la documentación del API para consultar

POST /predics →

#### Modo Batch
docker run --rm `
   -v "${PWD}/data:/data" `
   -v "${PWD}/model:/app/model" `
   model-serve:latest `
   --input /data/input_test.parquet `
   --output /data/output_preds.parquet

Esto genera :

data/input_test.parquet → muestra aleatorio del dataset original
data/output_preds.parquet → predicciones de la muestra de datos
