import os
import time
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
import uvicorn

# Inicializamos la aplicación FastAPI
api = FastAPI()

def to_str(x):
    return x.astype(str)

# Ruta del modelo entrenado
MODEL_FILE = "/app/model/model.pkl"
METRICS_FILE = "/app/model/metrics.json"

# Se carga el modelo al iniciar
modelo = joblib.load(MODEL_FILE)

def actualizar_metricas(tiempo_respuesta: float, cantidad_predicciones: int):
    """
    Actualiza el archivo de métricas con el tiempo de respuesta y número de predicciones.
    """
    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {"avg_response_time_seconds": 0.0, "predictions_served": 0}

    # Promedio simple (última ejecución)
    metrics["avg_response_time_seconds"] = tiempo_respuesta
    metrics["predictions_served"] += cantidad_predicciones

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    return metrics

@api.post("/predict")
def generar_predicciones(payload: dict = Body(...)):
    """
    Endpoint para recibir datos en formato JSON y devolver predicciones.
    """
    try:
        inicio = time.time()

        registros = payload.get("data", [])
        if not isinstance(registros, list) or not registros:
            raise HTTPException(status_code=400, detail="El cuerpo debe incluir 'data' como lista no vacía.")

        df = pd.DataFrame(registros)
        resultados = modelo.predict(df)

        tiempo = time.time() - inicio
        actualizar_metricas(tiempo, len(resultados))

        return {"predictions": resultados.tolist()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")

@api.get("/metrics")
def obtener_metricas():
    """
    Devuelve las métricas actuales del modelo en formato JSON.
    """
    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        return metrics
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Archivo de métricas no encontrado.")

def modo_batch():
    """
    Ejecuta predicciones en modo batch: lee un archivo Parquet y guarda las predicciones en otro.
    """
    inicio = time.time()

    parser = argparse.ArgumentParser(description="Modo batch para predicciones")
    parser.add_argument("--input", required=True, help="Ruta al archivo Parquet de entrada")
    parser.add_argument("--output", required=True, help="Ruta al archivo Parquet de salida")
    args = parser.parse_args()

    datos = pd.read_parquet(args.input)
    predicciones = modelo.predict(datos)

    salida = pd.DataFrame({"prediction": predicciones})
    salida.to_parquet(args.output, index=False)

    tiempo = time.time() - inicio
    actualizar_metricas(tiempo, len(predicciones))

if __name__ == "__main__":
    # Si se pasa argumento --input, ejecutamos batch; si no, levantamos API
    if any(arg.startswith("--input") for arg in os.sys.argv):
        modo_batch()
    else:
        uvicorn.run("app:api", host="0.0.0.0", port=8000)
