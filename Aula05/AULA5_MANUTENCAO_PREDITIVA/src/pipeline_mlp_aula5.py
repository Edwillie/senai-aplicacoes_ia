"""
AULA 05 — Redes Neurais aplicadas à Manutenção Preditiva Industrial

Dataset: AI4I 2020 Predictive Maintenance Dataset
https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

O pipeline segue a estrutura:

Bronze -> Prata -> Transformação -> Escalonamento -> Modelos -> Ouro

O objetivo é prever possível falha de máquina a partir de dados de temperatura,
velocidade de rotação, torque, desgaste de ferramenta e tipo de produto.
"""

from __future__ import annotations

import os
import sqlite3
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "ai4i2020.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DB_PATH = BASE_DIR / "database" / "aula5_pipeline.db"

MODELOS_DIR = BASE_DIR / "modelos"
ARTEFATOS_PATH = MODELOS_DIR / "modelos_aula5.joblib"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
MODELOS_DIR.mkdir(parents=True, exist_ok=True)


def titulo(texto: str) -> None:
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


# =============================================================================
# KDD 1 — SELEÇÃO / CAMADA BRONZE
# =============================================================================

def etapa_1_bronze(conn: sqlite3.Connection) -> pd.DataFrame:
    titulo("KDD 1 - SELEÇÃO / CAMADA BRONZE")

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {RAW_PATH}\n"
            "Coloque o arquivo ai4i2020.csv em data/raw/."
        )

    bronze = pd.read_csv(RAW_PATH)

    bronze.to_sql(
        "bronze_manutencao",
        conn,
        if_exists="replace",
        index=False,
    )

    print(f"Registros brutos (Bronze): {len(bronze)}")
    print(f"Colunas originais: {list(bronze.columns)}")

    return bronze


# =============================================================================
# KDD 2 — PRÉ-PROCESSAMENTO / CAMADA PRATA
# =============================================================================

def etapa_2_prata(
    bronze: pd.DataFrame,
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    titulo("KDD 2 - PRÉ-PROCESSAMENTO / CAMADA PRATA")

    prata = bronze.copy()

    prata.columns = (
        prata.columns.str.replace(r"\[.*?\]", "", regex=True)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    duplicados = prata.duplicated().sum()
    prata = prata.drop_duplicates()

    ausentes_antes = prata.isna().sum().sum()
    prata = prata.dropna()

    prata["type"] = prata["type"].str.strip().str.upper()

    prata.to_sql(
        "prata_manutencao",
        conn,
        if_exists="replace",
        index=False,
    )

    print(f"Duplicidades removidas: {duplicados}")
    print(f"Valores ausentes removidos: {ausentes_antes}")
    print(f"Registros após limpeza (Prata): {len(prata)}")

    return prata


# =============================================================================
# KDD 3 — TRANSFORMAÇÃO / X E y
# =============================================================================

def etapa_3_transformacao(prata: pd.DataFrame):
    titulo("KDD 3 - TRANSFORMAÇÃO / X E y")

    y = prata["machine_failure"].astype(int)

    colunas_a_remover = [
        "udi",
        "product_id",
        "machine_failure",
        "twf",
        "hdf",
        "pwf",
        "osf",
        "rnf",
    ]

    atributos = prata.drop(columns=colunas_a_remover)

    X = pd.get_dummies(
        atributos,
        columns=["type"],
        drop_first=False,
        dtype=int,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Atributos após transformação: {X.shape[1]}")
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
    print(f"Taxa de falha no dataset completo: {y.mean():.2%}")
    print(
        f"Taxa de falha no treino: {y_train.mean():.2%} "
        f"| no teste: {y_test.mean():.2%}"
    )

    return X, X_train, X_test, y_train, y_test


# =============================================================================
# ESCALONAMENTO PARA A MLP
# =============================================================================

def etapa_4_escalonamento(X_train, X_test):
    titulo("AULA 05 - ESCALONAMENTO PARA A MLP")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("StandardScaler ajustado SOMENTE no conjunto de treino.")
    print("O teste foi apenas transformado, evitando vazamento de informação.")

    return X_train_scaled, X_test_scaled, scaler


# =============================================================================
# MODELO BASELINE — ÁRVORE DE DECISÃO
# =============================================================================

def treinar_arvore(X_train, X_test, y_train):
    titulo("BASELINE - ÁRVORE DE DECISÃO")

    modelo = DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    return y_pred, modelo


# =============================================================================
# MLP COM SCIKIT-LEARN
# =============================================================================

def treinar_mlp_sklearn(X_train_scaled, X_test_scaled, y_train):
    titulo("AULA 05 - MLP COM SCIKIT-LEARN")

    modelo = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        modelo.fit(X_train_scaled, y_train)

    y_pred = modelo.predict(X_test_scaled)

    print("Arquitetura: entrada -> 16 -> 8 -> saída")
    print(f"Épocas (iterações): {modelo.n_iter_}")

    return y_pred, modelo


# =============================================================================
# MLP COM TENSORFLOW / KERAS
# =============================================================================

def treinar_mlp_keras(X_train_scaled, X_test_scaled, y_train):
    titulo("AULA 05 - MLP COM TENSORFLOW / KERAS")

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

    except ImportError:
        print("TensorFlow não está instalado.")
        print("O pipeline continuará com Árvore e MLP scikit-learn.")
        return None, None

    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    modelo = keras.Sequential(
        [
            keras.Input(shape=(X_train_scaled.shape[1],)),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    modelo.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )

    historico = modelo.fit(
        X_train_scaled,
        y_train.to_numpy(),
        validation_split=0.20,
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0,
    )

    probabilidades = modelo.predict(
        X_test_scaled,
        verbose=0,
    ).ravel()

    y_pred = (probabilidades >= 0.5).astype(int)

    print("Arquitetura: entrada -> Dense(16) -> Dense(8) -> Dense(1)")
    print(f"Épocas executadas: {len(historico.history['loss'])}")

    return y_pred, historico


# =============================================================================
# KDD 5 — INTERPRETAÇÃO / CAMADA OURO
# =============================================================================

def etapa_5_ouro(
    y_test,
    previsoes,
    historico_keras,
    conn: sqlite3.Connection,
):
    titulo("KDD 5 - INTERPRETAÇÃO / CAMADA OURO")

    linhas_metricas = []

    for nome_modelo, y_pred in previsoes.items():
        acuracia = accuracy_score(y_test, y_pred)

        matriz = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = matriz.ravel()

        recall_falha = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"\n--- {nome_modelo} ---")
        print(f"Acurácia: {acuracia:.4f}")
        print(f"Matriz de confusão:\n{matriz}")

        print(
            classification_report(
                y_test,
                y_pred,
                target_names=["sem_falha", "falha"],
                zero_division=0,
            )
        )

        linhas_metricas.append(
            {
                "modelo": nome_modelo,
                "acuracia": acuracia,
                "falso_positivo": fp,
                "falso_negativo": fn,
                "recall_falha": recall_falha,
            }
        )

    comparacao = (
        pd.DataFrame(linhas_metricas)
        .sort_values("acuracia", ascending=False)
    )

    comparacao.to_csv(
        PROCESSED_DIR / "ouro_comparacao_modelos.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparacao.to_sql(
        "ouro_comparacao_modelos",
        conn,
        if_exists="replace",
        index=False,
    )

    if historico_keras is not None:
        historico_df = pd.DataFrame(historico_keras.history)

        historico_df.to_csv(
            PROCESSED_DIR / "historico_treinamento_keras.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\nResumo comparativo:")
    print(comparacao.to_string(index=False))

    melhor = comparacao.iloc[0]

    print(
        f"\nMelhor acurácia: {melhor['modelo']} "
        f"({melhor['acuracia']:.2%})"
    )

    print(
        "\nAtenção: neste dataset as classes são desbalanceadas. "
        "Por isso, o recall da classe falha deve ser analisado junto "
        "com a acurácia geral."
    )


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main():
    with sqlite3.connect(DB_PATH) as conn:
        bronze = etapa_1_bronze(conn)
        prata = etapa_2_prata(bronze, conn)

        X, X_train, X_test, y_train, y_test = etapa_3_transformacao(prata)

        X_train_scaled, X_test_scaled, scaler = etapa_4_escalonamento(
            X_train,
            X_test,
        )

        y_pred_arvore, modelo_arvore = treinar_arvore(
            X_train,
            X_test,
            y_train,
        )

        y_pred_mlp, modelo_mlp = treinar_mlp_sklearn(
            X_train_scaled,
            X_test_scaled,
            y_train,
        )

        joblib.dump(
            {
                "modelo_arvore": modelo_arvore,
                "modelo_mlp": modelo_mlp,
                "scaler": scaler,
                "colunas_modelo": X.columns.tolist(),
            },
            ARTEFATOS_PATH,
        )

        print(f"\nModelos salvos para o simulador: {ARTEFATOS_PATH}")

        y_pred_keras, historico_keras = treinar_mlp_keras(
            X_train_scaled,
            X_test_scaled,
            y_train,
        )

        previsoes = {
            "Árvore de Decisão": y_pred_arvore,
            "MLP scikit-learn": y_pred_mlp,
        }

        if y_pred_keras is not None:
            previsoes["MLP TensorFlow/Keras"] = y_pred_keras

        etapa_5_ouro(
            y_test,
            previsoes,
            historico_keras,
            conn,
        )


if __name__ == "__main__":
    main()