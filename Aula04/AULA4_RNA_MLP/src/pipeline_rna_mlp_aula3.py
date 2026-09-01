from __future__ import annotations

from pathlib import Path
import os
import sqlite3
import unicodedata
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw" / "dados_qualidade_pecas_kdd.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DB_PATH = BASE_DIR / "database" / "aula3_pipeline.db"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def titulo(texto: str) -> None:
    print("\n" + "=" * 80)
    print(texto)
    print("=" * 80)


def salvar_tabela(df: pd.DataFrame, nome: str, conn: sqlite3.Connection) -> None:
    df.to_sql(nome, conn, if_exists="replace", index=False)


def chave_texto(valor: object) -> str | None:
    if pd.isna(valor):
        return None

    texto = " ".join(str(valor).strip().lower().split())
    if not texto:
        return None

    return "".join(
        c
        for c in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(c)
    )


def padronizar_categorias(df: pd.DataFrame) -> pd.DataFrame:
    mapas = {
        "linha": {
            "linha a": "linha a", "a": "linha a",
            "linha b": "linha b", "b": "linha b",
            "linha c": "linha c", "c": "linha c",
        },
        "turno": {
            "manha": "manhã", "matutino": "manhã",
            "tarde": "tarde", "vespertino": "tarde",
            "noite": "noite", "noturno": "noite",
        },
        "fornecedor": {
            "forn a": "fornecedor a", "fornecedor a": "fornecedor a",
            "forn b": "fornecedor b", "fornecedor b": "fornecedor b",
            "forn c": "fornecedor c", "fornecedor c": "fornecedor c",
        },
        "resultado_inspecao": {
            "conforme": "conforme",
            "retrabalho": "retrabalho",
        },
    }

    saida = df.copy()

    for coluna, mapa in mapas.items():
        chaves = saida[coluna].map(chave_texto)
        saida[coluna] = chaves.map(mapa).astype("string")

    return saida


def etapa_1_bronze(conn: sqlite3.Connection) -> pd.DataFrame:
    titulo("KDD 1 - SELEÇÃO / CAMADA BRONZE")

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"CSV não encontrado em: {RAW_PATH}\n"
            "Coloque dados_qualidade_pecas_kdd.csv dentro de data/raw."
        )

    bronze = pd.read_csv(RAW_PATH)
    salvar_tabela(bronze, "bronze_qualidade_pecas", conn)

    print(f"Arquivo: {RAW_PATH.name}")
    print(f"Formato: {bronze.shape[0]} linhas x {bronze.shape[1]} colunas")
    print("\nAusências no dado bruto:")
    print(bronze.isna().sum().to_string())
    print(f"\nDuplicidades completas: {bronze.duplicated().sum()}")

    return bronze


def etapa_2_prata(bronze: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    titulo("KDD 2 - PRÉ-PROCESSAMENTO / CAMADA PRATA")

    prata = bronze.copy()

    antes = len(prata)
    prata = prata.drop_duplicates().copy()
    print(f"Duplicidades removidas: {antes - len(prata)}")

    prata.columns = prata.columns.str.strip().str.lower()
    prata = padronizar_categorias(prata)

    colunas_num = [
        "temp_forno_c",
        "tempo_ciclo_s",
        "pressao_bar",
        "umidade_pct",
    ]

    for coluna in colunas_num:
        prata[coluna] = pd.to_numeric(prata[coluna], errors="coerce")

    prata.loc[~prata["temp_forno_c"].between(100, 260), "temp_forno_c"] = np.nan
    prata.loc[~prata["tempo_ciclo_s"].between(1, 120), "tempo_ciclo_s"] = np.nan
    prata.loc[~prata["pressao_bar"].between(0.1, 20), "pressao_bar"] = np.nan
    prata.loc[~prata["umidade_pct"].between(0, 100), "umidade_pct"] = np.nan

    print("\nAusências após conversão e regras de validade:")
    print(prata.isna().sum().to_string())

    for coluna in colunas_num:
        mediana = prata[coluna].median()
        prata[coluna] = prata[coluna].fillna(mediana)

    for coluna in ["linha", "turno", "fornecedor"]:
        prata[coluna] = prata[coluna].fillna("desconhecido")

    sem_alvo = prata["resultado_inspecao"].isna().sum()
    if sem_alvo:
        print(f"Removendo {sem_alvo} linha(s) sem classe conhecida.")
        prata = prata.dropna(subset=["resultado_inspecao"]).copy()

    salvar_tabela(prata, "prata_qualidade_pecas", conn)
    prata.to_csv(
        PROCESSED_DIR / "qualidade_prata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nCamada prata: {len(prata)} registros")
    return prata


def etapa_3_transformacao(prata: pd.DataFrame):
    titulo("KDD 3 - TRANSFORMAÇÃO / X E y")

    y = prata["resultado_inspecao"].map({
        "conforme": 0,
        "retrabalho": 1,
    })

    if y.isna().any():
        raise ValueError("Há classes não reconhecidas em resultado_inspecao.")

    y = y.astype(int)

    atributos = prata.drop(columns=["id_lote", "resultado_inspecao"])

    X = pd.get_dummies(
        atributos,
        columns=["linha", "turno", "fornecedor"],
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

    return X, X_train, X_test, y_train, y_test


def etapa_4_escalonamento(X_train, X_test):
    titulo("AULA 03 - ESCALONAMENTO PARA A MLP")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("StandardScaler ajustado SOMENTE no conjunto de treino.")
    print("O teste foi apenas transformado, evitando vazamento de informação.")

    return X_train_scaled, X_test_scaled


def treinar_arvore(X_train, X_test, y_train):
    titulo("BASELINE - ÁRVORE DE DECISÃO DA AULA 02")

    modelo = DecisionTreeClassifier(
        max_depth=3,
        random_state=RANDOM_STATE,
    )

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    return y_pred


def treinar_mlp_sklearn(X_train_scaled, X_test_scaled, y_train):
    titulo("AULA 03 - MLP COM SCIKIT-LEARN")

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
    print("Ativação oculta: ReLU")
    print("Otimizador: Adam")
    print(f"Iterações executadas: {modelo.n_iter_}")
    print(f"Loss final: {modelo.loss_:.6f}")

    return y_pred


def treinar_mlp_keras(X_train_scaled, X_test_scaled, y_train):
    titulo("AULA 03 - MLP COM TENSORFLOW / KERAS")

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("TensorFlow não está instalado.")
        print("O pipeline continuará com a MLP do scikit-learn.")
        print("Para executar esta etapa: pip install tensorflow")
        return None, None

    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    modelo = keras.Sequential([
        keras.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
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
        batch_size=8,
        callbacks=[early_stopping],
        verbose=0,
    )

    probabilidades = modelo.predict(X_test_scaled, verbose=0).ravel()
    y_pred = (probabilidades >= 0.5).astype(int)

    print("Arquitetura: entrada -> Dense(16) -> Dense(8) -> Dense(1)")
    print("ReLU nas camadas ocultas e Sigmoid na saída.")
    print(f"Épocas executadas: {len(historico.history['loss'])}")

    return y_pred, historico


def avaliar_modelo(nome: str, y_test, y_pred):
    acuracia = accuracy_score(y_test, y_pred)
    matriz = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matriz.ravel()

    print(f"\n{nome}")
    print("-" * len(nome))
    print(f"Acurácia: {acuracia:.2%}")

    matriz_tela = pd.DataFrame(
        matriz,
        index=["real_conforme", "real_retrabalho"],
        columns=["prev_conforme", "prev_retrabalho"],
    )
    print(matriz_tela.to_string())

    print(f"Falso positivo: {fp}")
    print(f"Falso negativo: {fn}")

    relatorio = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["conforme", "retrabalho"],
        zero_division=0,
        output_dict=True,
    )

    metricas = pd.DataFrame({
        "modelo": [nome],
        "acuracia": [float(acuracia)],
        "verdadeiro_negativo": [int(tn)],
        "falso_positivo": [int(fp)],
        "falso_negativo": [int(fn)],
        "verdadeiro_positivo": [int(tp)],
    })

    relatorio_df = pd.DataFrame(relatorio).T.reset_index(names="classe_ou_media")
    relatorio_df.insert(0, "modelo", nome)

    return metricas, relatorio_df


def etapa_5_ouro(
    prata,
    X_test,
    y_test,
    previsoes,
    historico_keras,
    conn,
):
    titulo("KDD 5 - INTERPRETAÇÃO / CAMADA OURO")

    metricas_lista = []
    relatorios_lista = []
    resultados_lista = []

    ids_teste = prata.loc[X_test.index, "id_lote"].to_numpy()

    for nome, y_pred in previsoes.items():
        if y_pred is None:
            continue

        metricas, relatorio = avaliar_modelo(nome, y_test, y_pred)

        metricas_lista.append(metricas)
        relatorios_lista.append(relatorio)

        resultados = pd.DataFrame({
            "modelo": nome,
            "id_lote": ids_teste,
            "real": pd.Series(y_test.to_numpy()).map({
                0: "conforme",
                1: "retrabalho",
            }),
            "previsto": pd.Series(y_pred).map({
                0: "conforme",
                1: "retrabalho",
            }),
        })

        resultados["acertou"] = (
            resultados["real"] == resultados["previsto"]
        )

        resultados_lista.append(resultados)

    metricas_gerais = pd.concat(metricas_lista, ignore_index=True)
    relatorios_gerais = pd.concat(relatorios_lista, ignore_index=True)
    resultados_gerais = pd.concat(resultados_lista, ignore_index=True)

    comparacao = metricas_gerais[[
        "modelo",
        "acuracia",
        "falso_positivo",
        "falso_negativo",
    ]].sort_values(
        ["acuracia", "falso_negativo"],
        ascending=[False, True],
    )

    indicadores_linha = (
        prata.groupby("linha")["resultado_inspecao"]
        .apply(lambda s: (s == "retrabalho").mean())
        .reset_index(name="taxa_retrabalho")
        .sort_values("taxa_retrabalho", ascending=False)
    )

    ouro_lotes = prata[[
        "id_lote",
        "linha",
        "turno",
        "fornecedor",
        "resultado_inspecao",
    ]].copy()

    salvar_tabela(ouro_lotes, "ouro_lotes_qualidade", conn)
    salvar_tabela(indicadores_linha, "ouro_indicadores_linha", conn)
    salvar_tabela(metricas_gerais, "ouro_metricas_modelos", conn)
    salvar_tabela(relatorios_gerais, "ouro_relatorios_modelos", conn)
    salvar_tabela(resultados_gerais, "ouro_resultados_teste", conn)
    salvar_tabela(comparacao, "ouro_comparacao_modelos", conn)

    ouro_lotes.to_csv(
        PROCESSED_DIR / "ouro_lotes_qualidade.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metricas_gerais.to_csv(
        PROCESSED_DIR / "ouro_metricas_modelos.csv",
        index=False,
        encoding="utf-8-sig",
    )
    resultados_gerais.to_csv(
        PROCESSED_DIR / "ouro_resultados_teste.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparacao.to_csv(
        PROCESSED_DIR / "ouro_comparacao_modelos.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if historico_keras is not None:
        historico_df = pd.DataFrame(historico_keras.history)
        historico_df.insert(0, "epoca", range(1, len(historico_df) + 1))

        salvar_tabela(historico_df, "ouro_historico_keras", conn)

        historico_df.to_csv(
            PROCESSED_DIR / "historico_treinamento_keras.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\nCOMPARAÇÃO FINAL")
    print(
        comparacao.to_string(
            index=False,
            formatters={"acuracia": "{:.2%}".format},
        )
    )

    titulo("REFLEXÕES DA AULA 03")
    print("1. O pipeline permaneceu; o mecanismo de aprendizagem mudou.")
    print("2. A MLP exige atenção à escala dos atributos.")
    print("3. Acurácia não deve ser analisada sem a matriz de confusão.")
    print("4. A árvore gera regras explícitas; a MLP aprende pesos distribuídos.")
    print("5. Um modelo mais complexo não é automaticamente um modelo melhor.")


def main() -> None:
    titulo("AULA 03 - REDES NEURAIS MLP APLICADAS À QUALIDADE INDUSTRIAL")

    print(
        "Nesta aula reaproveitamos o pipeline da Aula 02 e comparamos "
        "a Árvore de Decisão com redes MLP."
    )

    with sqlite3.connect(DB_PATH) as conn:
        bronze = etapa_1_bronze(conn)
        prata = etapa_2_prata(bronze, conn)

        X, X_train, X_test, y_train, y_test = etapa_3_transformacao(prata)

        X_train_scaled, X_test_scaled = etapa_4_escalonamento(
            X_train,
            X_test,
        )

        y_pred_arvore = treinar_arvore(
            X_train,
            X_test,
            y_train,
        )

        y_pred_mlp = treinar_mlp_sklearn(
            X_train_scaled,
            X_test_scaled,
            y_train,
        )

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
            prata=prata,
            X_test=X_test,
            y_test=y_test,
            previsoes=previsoes,
            historico_keras=historico_keras,
            conn=conn,
        )

    titulo("PIPELINE DA AULA 03 CONCLUÍDO")
    print(f"Banco SQLite: {DB_PATH}")
    print(f"Arquivos processados: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
