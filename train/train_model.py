import pandas as pd
import numpy as np
import yaml
import optuna
import joblib
import json
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

# ============================
# Cargar configuración y datos
# ============================
config = yaml.safe_load(open("config.yaml"))
dataset = pd.read_csv("/data/hotel_bookings.csv")

# Variables por tipo
num_features = ['lead_time', 'stays_in_weekend_nights', 'adults', 'babies', 'adr']
cat_features = ['hotel', 'market_segment', 'distribution_channel']
high_card_features = ['country']
ord_features = ['reserved_room_type', 'assigned_room_type']
date_features = ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']
target_col = 'is_canceled'

# ============================
# Ingeniería de fechas
# ============================
dataset['arrival_date'] = pd.to_datetime(
    dataset['arrival_date_year'].astype(str) + '-' +
    dataset['arrival_date_month'].astype(str) + '-' +
    dataset['arrival_date_day_of_month'].astype(str),
    errors='coerce'
)
dataset['day_of_week'] = dataset['arrival_date'].dt.dayofweek
dataset['is_weekend'] = dataset['day_of_week'].isin([5, 6]).astype(int)

# ============================
# Generando Parquet
# ============================

# Eliminar la columna objetivo
df_features = dataset.drop(columns=["is_canceled"])

# Tomar una muestra random de 100 filas
df_sample = df_features.sample(100, random_state=42)

# Guardar en formato Parquet
df_sample.to_parquet("/data/input_test.parquet", engine="pyarrow", index=False)


X = dataset.drop(columns=[target_col])
y = dataset[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============================
# Preparación de categorías ordinales
# ============================
ord_reserved = sorted(dataset['reserved_room_type'].astype(str).dropna().unique())
ord_assigned = sorted(dataset['assigned_room_type'].astype(str).dropna().unique())
ord_weekday = sorted(dataset['day_of_week'].astype(str).dropna().unique())

def to_str(x):
    return x.astype(str)

# ============================
# Pipelines de preprocesamiento
# ============================
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

high_card_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # simplificado para estabilidad
])

ord_pipeline_reserved = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('to_str', FunctionTransformer(to_str)),
    ('ordinal', OrdinalEncoder(categories=[ord_reserved]))
])

ord_pipeline_assigned = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('to_str', FunctionTransformer(to_str)),
    ('ordinal', OrdinalEncoder(categories=[ord_assigned]))
])

ord_pipeline_weekday = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('to_str', FunctionTransformer(to_str)),
    ('ordinal', OrdinalEncoder(categories=[ord_weekday]))
])

# ============================
# ColumnTransformer final
# ============================
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features + ['is_weekend']),
    ('high_card', high_card_pipeline, high_card_features),
    ('ord_reserved', ord_pipeline_reserved, ['reserved_room_type']),
    ('ord_assigned', ord_pipeline_assigned, ['assigned_room_type']),
    ('ord_weekday', ord_pipeline_weekday, ['day_of_week'])
])

# ============================
# Función objetivo para Optuna
# ============================
def optuna_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", config['search_space']["n_estimators"][0], config['search_space']["n_estimators"][1])
    max_depth = trial.suggest_int("max_depth", config['search_space']["max_depth"][0], config['search_space']["max_depth"][1])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train.values.ravel(), cv=cv, scoring=config['optimizer']['metric'])
    return scores.mean()

# ============================
# Optimización con Optuna
# ============================
study = optuna.create_study(direction=config['optimizer']['direction'])
study.optimize(optuna_objective, n_trials=config['optimizer']['n_trials'])

best_params = study.best_params

# ============================
# Entrenamiento final
# ============================
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        random_state=42,
        n_jobs=-1
    ))
])

final_pipeline.fit(X_train, y_train.values.ravel())

# ============================
# Evaluación y exportación
# ============================
y_pred = final_pipeline.predict(X_test)
acc_test = round(accuracy_score(y_test, y_pred), 4)

metrics = {
    "model": "RandomForestClassifier",
    "accuracy": acc_test,
    "predictions_served": 0,
    "avg_response_time_seconds": 0,
    "last_training": date.today().strftime("%Y-%m-%d")
}

with open("/output/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f)

joblib.dump(final_pipeline, "/output/model.pkl")
