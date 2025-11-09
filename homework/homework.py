# flake8: noqa: E501
import os, json, gzip, pickle
from typing import List, Dict
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
)

# ----------------------------- Rutas ---------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(ROOT, os.pardir))

TRAIN_ZIP = os.path.join(BASE, "files/input/train_data.csv.zip")
TEST_ZIP  = os.path.join(BASE, "files/input/test_data.csv.zip")
MODEL_OUT = os.path.join(BASE, "files/models/model.pkl.gz")
METRICS_OUT = os.path.join(BASE, "files/output/metrics.json")

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)

# ---------------------- Carga y limpieza ------------------------------
def load_csv_zip(path: str) -> pd.DataFrame:
    return pd.read_csv(path, compression="zip")

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    if "EDUCATION" in df.columns:
        df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
        df.loc[df["EDUCATION"] == 0, "EDUCATION"] = 4
    df = df.dropna(axis=0).reset_index(drop=True)
    for c in df.select_dtypes(include=["int64", "int32"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    return df

train_df = clean_df(load_csv_zip(TRAIN_ZIP))
test_df  = clean_df(load_csv_zip(TEST_ZIP))

# ----------------------------- Split ---------------------------------
assert "default" in train_df.columns and "default" in test_df.columns
X_train = train_df.drop(columns=["default"])
y_train = train_df["default"].astype(int)
X_test  = test_df.drop(columns=["default"])
y_test  = test_df["default"].astype(int)

# ------------- Pipeline: OHE -> PCA -> Std -> SelectKBest -> SVC -----
# OHE solo a SEX, EDUCATION, MARRIAGE; PAY_* quedan como numéricas (ordinales)
categorical_cols: List[str] = [c for c in ["SEX", "EDUCATION", "MARRIAGE"] if c in X_train.columns]
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ],
    remainder="drop",
    sparse_threshold=0.0,  # salida densa para PCA
)

pca = PCA(n_components=None, svd_solver="full", random_state=42)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("pca", pca),
    ("scaler", StandardScaler()),
    ("select", SelectKBest(f_classif, k=20)),  # valor base; se afina en grid
    ("clf", SVC(kernel="rbf", cache_size=1500, random_state=42)),
])

# ---------------------- Búsqueda de hiperparámetros -------------------
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
param_grid = {
    "select__k": [10, 20, 30],         # cuida que no supere n_features post-PCA
    "clf__C": [1.0, 2.0],
    "clf__gamma": ["scale", "auto"],
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=cv,
    n_jobs=1,          # estable para RAM y evita loky warnings
    pre_dispatch=1,
    refit=True,
    verbose=0,
)

grid.fit(X_train, y_train)

# ------------------------- Guardar modelo -----------------------------
# IMPORTANTE: el test espera un GridSearchCV
with gzip.open(MODEL_OUT, "wb") as f:
    pickle.dump(grid, f)

# --------------------------- Métricas ---------------------------------
def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

# Usa el grid para predecir (internamente usa best_estimator_)
ytr_pred = grid.predict(X_train)
yte_pred = grid.predict(X_test)

train_metrics = {"type": "metrics", "dataset": "train", **compute_metrics(y_train, ytr_pred)}
test_metrics  = {"type": "metrics", "dataset": "test",  **compute_metrics(y_test,  yte_pred)}

def cm_dict(y_true, y_pred) -> Dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return {"true_0": {"predicted_0": tn, "predicted_1": fp},
            "true_1": {"predicted_0": fn, "predicted_1": tp}}

cm_train = {"type": "cm_matrix", "dataset": "train", **cm_dict(y_train, ytr_pred)}
cm_test  = {"type": "cm_matrix", "dataset": "test",  **cm_dict(y_test,  yte_pred)}

with open(METRICS_OUT, "w", encoding="utf-8") as f:
    f.write(json.dumps(train_metrics, ensure_ascii=False) + "\n")
    f.write(json.dumps(test_metrics, ensure_ascii=False) + "\n")
    f.write(json.dumps(cm_train, ensure_ascii=False) + "\n")
    f.write(json.dumps(cm_test, ensure_ascii=False) + "\n")

# --------------------------- Info consola -----------------------------
print("Mejores hiperparámetros:", grid.best_params_)
print("Mejor balanced_accuracy (CV):", grid.best_score_)
print("Accuracy train:", grid.score(X_train, y_train))
print("Accuracy test :", grid.score(X_test, y_test))
print(f"Modelo guardado en: {MODEL_OUT}")
print(f"Métricas guardadas en: {METRICS_OUT}")
