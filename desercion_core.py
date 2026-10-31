# -*- coding: utf-8 -*-
"""Lógica del modelo de deserción estudiantil (sin dependencias de Streamlit).

Este módulo concentra todo lo "científico": carga del artefacto, preparación de
variables, predicción y análisis de sensibilidad. Al no importar Streamlit puede
probarse de forma aislada y reutilizarse en notebooks o scripts por lotes.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Definición de variables
# --------------------------------------------------------------------------- #
NUM_COLS: list[str] = [
    "TERM_ID", "GPA_TERM", "CREDITS_ATTEMPTED_TERM", "CREDITS_FAILED_TERM",
    "GPA_CUM", "CREDITS_APPROVED_CUM_ALL", "CREDITS_FAILED_CUM",
]
CAT_COLS: list[str] = [
    "COD_ESCUELA", "ACADEMIC_STATUS_TERM", "GENERO",
    "RECIBIO_APOYO_FINANCIERO", "RECIBIO_APOYO_ACADEMICO", "RECIBIO_OTROS_APOYOS",
]
FEATURE_ORDER: list[str] = NUM_COLS + CAT_COLS

# Las comillas simples en los estados forman parte del valor con el que se
# entrenó el modelo (columnas dummy "ACADEMIC_STATUS_TERM_'...'"); no quitarlas.
CAT_OPTIONS: dict[str, list[str]] = {
    "COD_ESCUELA": ["PS", "PI", "PD", "PE", "PP"],
    "ACADEMIC_STATUS_TERM": [
        "'Estado académico normal'", "'Advertencia académica'",
        "'Primer reintegro'", "'Segundo reintegro'", "'Tercer reintegro'",
        "'Cuarto reintegro'", "'Quinto reintegro'",
        "'Primera suspensión académica'", "'Segunda suspensión académica'",
        "'Tercera suspensión académica'", "'Cuarta suspensión académica'",
    ],
    "GENERO": ["F", "M"],
    "RECIBIO_APOYO_FINANCIERO": ["N", "S"],
    "RECIBIO_APOYO_ACADEMICO": ["N", "S"],
    "RECIBIO_OTROS_APOYOS": ["N", "S"],
}

LABELS: dict[str, str] = {
    "TERM_ID": "Periodo académico (AAAAPP)",
    "GPA_TERM": "Promedio del periodo",
    "CREDITS_ATTEMPTED_TERM": "Créditos inscritos (periodo)",
    "CREDITS_FAILED_TERM": "Créditos reprobados (periodo)",
    "GPA_CUM": "Promedio acumulado",
    "CREDITS_APPROVED_CUM_ALL": "Créditos aprobados (acumulado)",
    "CREDITS_FAILED_CUM": "Créditos reprobados (acumulado)",
    "COD_ESCUELA": "Escuela",
    "ACADEMIC_STATUS_TERM": "Estado académico",
    "GENERO": "Género",
    "RECIBIO_APOYO_FINANCIERO": "Apoyo financiero",
    "RECIBIO_APOYO_ACADEMICO": "Apoyo académico",
    "RECIBIO_OTROS_APOYOS": "Otros apoyos",
}

# Valores iniciales razonables para la interfaz.
DEFAULTS: dict[str, object] = {
    "TERM_ID": 202610, "GPA_TERM": 3.5, "CREDITS_ATTEMPTED_TERM": 16,
    "CREDITS_FAILED_TERM": 0, "GPA_CUM": 3.5, "CREDITS_APPROVED_CUM_ALL": 60,
    "CREDITS_FAILED_CUM": 0, "COD_ESCUELA": "PS",
    "ACADEMIC_STATUS_TERM": "'Estado académico normal'", "GENERO": "F",
    "RECIBIO_APOYO_FINANCIERO": "N", "RECIBIO_APOYO_ACADEMICO": "N",
    "RECIBIO_OTROS_APOYOS": "N",
}
FLOAT_COLS = {"GPA_TERM", "GPA_CUM"}   # escala de notas 0–5
GPA_MAX = 5.0
POSITIVE_CLASS = "SI"                  # etiqueta de la clase "deserta"


# --------------------------------------------------------------------------- #
# Carga del artefacto
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Artifacts:
    model: object
    label_encoder: object
    features: list[str]
    scaler: object


def load_artifacts(path: str | Path) -> Artifacts:
    """Carga el .pkl (modelo, labelencoder, variables, min_max_scaler).

    Nota de seguridad: pickle ejecuta código al deserializar; cargar solo
    archivos de origen confiable.
    """
    with open(path, "rb") as fh:
        model, encoder, features, scaler = pickle.load(fh)
    return Artifacts(model, encoder, list(features), scaler)


def bounds(art: Artifacts) -> dict[str, tuple[float, float]]:
    """Rango observado en entrenamiento para cada variable numérica."""
    out: dict[str, tuple[float, float]] = {}
    for col, lo, hi in zip(NUM_COLS, art.scaler.data_min_, art.scaler.data_max_):
        if col in FLOAT_COLS:
            out[col] = (0.0, GPA_MAX)
        else:
            out[col] = (int(lo), int(hi))
    return out


# --------------------------------------------------------------------------- #
# Preparación y predicción (vectorizadas: aceptan 1 o N filas)
# --------------------------------------------------------------------------- #
def prepare(df: pd.DataFrame, art: Artifacts) -> pd.DataFrame:
    """Dummies + alineación con las columnas del entrenamiento + MinMax."""
    X = pd.get_dummies(df[FEATURE_ORDER], columns=CAT_COLS, dtype=int)
    X = X.reindex(columns=art.features, fill_value=0)
    X[NUM_COLS] = X[NUM_COLS].astype(float)
    X[NUM_COLS] = art.scaler.transform(X[NUM_COLS])  # solo transform, nunca fit
    return X


def decision_scores(df: pd.DataFrame, art: Artifacts) -> np.ndarray:
    """Margen de la SVM: > 0 → clase 'SI' (deserta)."""
    return art.model.decision_function(prepare(df, art))


def risk_index(score: np.ndarray | float) -> np.ndarray | float:
    """Índice de riesgo 0–1 = sigmoide del margen SVM.

    OJO: el SVC fue entrenado con probability=False, por lo que esto NO es una
    probabilidad calibrada; es una escala monótona donde 0.5 equivale a la
    frontera de decisión.
    """
    return 1.0 / (1.0 + np.exp(-np.asarray(score, dtype=float)))


@dataclass(frozen=True)
class Prediction:
    label: str        # "SI" / "NO"
    score: float      # margen SVM
    index: float      # índice de riesgo 0–1
    high_risk: bool


def predict(profile: dict, art: Artifacts) -> Prediction:
    df = pd.DataFrame([profile])
    X = prepare(df, art)
    label = str(art.label_encoder.inverse_transform(art.model.predict(X))[0])
    score = float(art.model.decision_function(X)[0])
    return Prediction(label, score, float(risk_index(score)), label == POSITIVE_CLASS)


# --------------------------------------------------------------------------- #
# Análisis complementario
# --------------------------------------------------------------------------- #
def _candidate_values(var: str, art: Artifacts) -> list:
    if var in CAT_OPTIONS:
        return CAT_OPTIONS[var]
    lo, hi = bounds(art)[var]
    return [lo, hi]


def factor_impact(profile: dict, art: Artifacts, exclude: tuple[str, ...] = ("TERM_ID",)) -> pd.DataFrame:
    """Sensibilidad 'una variable a la vez' (tipo tornado).

    Para cada variable se mueve solo ella (a los extremos observados si es
    numérica, o a cada categoría si es categórica), manteniendo el resto del
    perfil fijo, y se registra el rango del índice de riesgo (0–100).
    Todas las variantes se evalúan en UNA sola llamada al modelo.
    """
    variables = [v for v in FEATURE_ORDER if v not in exclude]
    rows, owner = [], []
    for var in variables:
        for val in _candidate_values(var, art):
            row = dict(profile)
            row[var] = val
            rows.append(row)
            owner.append(var)
    idx = risk_index(decision_scores(pd.DataFrame(rows), art)) * 100
    tmp = pd.DataFrame({"variable": owner, "indice": idx})
    current = predict(profile, art).index * 100
    agg = tmp.groupby("variable")["indice"].agg(bajo="min", alto="max").reset_index()
    agg["bajo"] = agg["bajo"].clip(upper=current)
    agg["alto"] = agg["alto"].clip(lower=current)
    agg["rango"] = agg["alto"] - agg["bajo"]
    agg["etiqueta"] = agg["variable"].map(LABELS)
    agg["actual"] = current
    return agg.sort_values("rango", ascending=True).reset_index(drop=True)


def sensitivity_curve(profile: dict, var: str, art: Artifacts, n: int = 60) -> pd.DataFrame:
    """Índice de riesgo (0–100) al variar una variable numérica en todo su rango."""
    lo, hi = bounds(art)[var]
    xs = np.linspace(lo, hi, n)
    df = pd.DataFrame([profile] * n)
    df[var] = xs
    return pd.DataFrame({"x": xs, "indice": risk_index(decision_scores(df, art)) * 100})


def profile_table(profile: dict, prediction: Prediction) -> pd.DataFrame:
    """Tabla vertical Variable/Valor lista para mostrar."""
    rows = [(LABELS[k], str(profile[k]).strip("'")) for k in FEATURE_ORDER]
    rows.append(("Predicción", "Deserta (SI)" if prediction.high_risk else "No deserta (NO)"))
    return pd.DataFrame(rows, columns=["Variable", "Valor"])
