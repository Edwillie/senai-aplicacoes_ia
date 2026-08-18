from __future__ import annotations

from pathlib import Path
import sqlite3
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

# -----------------------------------------------------------------------------
# CAMINHOS DO PROJETO
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw" / "dados_qualidade_pecas_kdd.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DB_PATH = BASE_DIR / "database" / "aula2_pipeline.db"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def titulo(texto: str) -> None:
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def salvar_tabela(df: pd.DataFrame, nome: str, conn: sqlite3.Connection) -> None:
    """Persiste um DataFrame como tabela SQLite, substituindo a versão anterior."""
    df.to_sql(nome, conn, if_exists="replace", index=False)


def chave_texto(valor: object) -> str | None:
    """Normaliza texto para comparação: minúsculas, sem acentos e sem espaços extras."""
    if pd.isna(valor):
        return None
    texto = " ".join(str(valor).strip().lower().split())
    if not texto:
        return None
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(c)
    )


def padronizar_categorias(df: pd.DataFrame) -> pd.DataFrame:
    """Consolida grafias equivalentes em nomes canônicos usados na camada prata."""
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

    resultado = df.copy()
    for coluna, mapa in mapas.items():
        chaves = resultado[coluna].map(chave_texto)
        resultado[coluna] = chaves.map(mapa).astype("string")
    return resultado


def etapa_1_selecao_e_bronze(conn: sqlite3.Connection) -> pd.DataFrame:
    titulo("KDD 1 — SELEÇÃO / CAMADA BRONZE")
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"CSV não encontrado em: {RAW_PATH}\n"
            "Confira se o arquivo está dentro de data/raw."
        )

    bronze = pd.read_csv(RAW_PATH)
    salvar_tabela(bronze, "bronze_qualidade_pecas", conn)

    print(f"Arquivo: {RAW_PATH.name}")
    print(f"Formato bruto: {bronze.shape[0]} linhas x {bronze.shape[1]} colunas")
    print("\nPrimeiras linhas do dado bruto:")
    print(bronze.head(6).to_string(index=False))
    print("\nValores ausentes detectados no bruto:")
    print(bronze.isna().sum().to_string())
    print(f"\nDuplicidades completas: {bronze.duplicated().sum()}")
    return bronze


def etapa_2_preprocessamento(bronze: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    titulo("KDD 2 — PRÉ-PROCESSAMENTO / CAMADA PRATA")
    prata = bronze.copy()

    # 1. Remover linhas completamente duplicadas.
    antes = len(prata)
    prata = prata.drop_duplicates().copy()
    print(f"Duplicidades removidas: {antes - len(prata)}")

    # 2. Padronizar nomes de colunas.
    prata.columns = prata.columns.str.strip().str.lower()

    # 3. Padronizar categorias equivalentes.
    prata = padronizar_categorias(prata)

    # 4. Converter variáveis numéricas. Valores como 'erro' viram NaN.
    colunas_num = ["temp_forno_c", "tempo_ciclo_s", "pressao_bar", "umidade_pct"]
    for coluna in colunas_num:
        prata[coluna] = pd.to_numeric(prata[coluna], errors="coerce")

    # 5. Regras para medições fisicamente incompatíveis com este processo.
    prata.loc[~prata["temp_forno_c"].between(100, 260), "temp_forno_c"] = np.nan
    prata.loc[~prata["tempo_ciclo_s"].between(1, 120), "tempo_ciclo_s"] = np.nan
    prata.loc[~prata["pressao_bar"].between(0.1, 20), "pressao_bar"] = np.nan
    prata.loc[~prata["umidade_pct"].between(0, 100), "umidade_pct"] = np.nan

    print("\nAusências após conversão/regras de validade:")
    print(prata.isna().sum().to_string())

    # 6. Imputação numérica pela mediana.
    medianas = {}
    for coluna in colunas_num:
        mediana = prata[coluna].median()
        medianas[coluna] = mediana
        prata[coluna] = prata[coluna].fillna(mediana)

    # 7. Ausências categóricas preservam a informação de desconhecimento.
    for coluna in ["linha", "turno", "fornecedor"]:
        prata[coluna] = prata[coluna].fillna("desconhecido")

    # A variável-alvo precisa ser conhecida para aprendizado supervisionado.
    linhas_sem_alvo = prata["resultado_inspecao"].isna().sum()
    if linhas_sem_alvo:
        print(f"ATENÇÃO: removendo {linhas_sem_alvo} linha(s) sem variável-alvo conhecida.")
        prata = prata.dropna(subset=["resultado_inspecao"]).copy()

    salvar_tabela(prata, "prata_qualidade_pecas", conn)
    prata.to_csv(PROCESSED_DIR / "qualidade_prata.csv", index=False, encoding="utf-8-sig")

    print("\nMedianas usadas na imputação:")
    for coluna, valor in medianas.items():
        print(f"  {coluna}: {valor:.2f}")

    print("\nCategorias depois da padronização:")
    for coluna in ["linha", "turno", "fornecedor", "resultado_inspecao"]:
        print(f"  {coluna}: {sorted(prata[coluna].dropna().unique().tolist())}")

    print(f"\nCamada prata: {prata.shape[0]} linhas x {prata.shape[1]} colunas")
    return prata


def etapa_3_transformacao(prata: pd.DataFrame):
    titulo("KDD 3 — TRANSFORMAÇÃO / PREPARAÇÃO DE X E y")

    y = prata["resultado_inspecao"].map({"conforme": 0, "retrabalho": 1})
    if y.isna().any():
        valores = prata.loc[y.isna(), "resultado_inspecao"].unique().tolist()
        raise ValueError(f"Há classes não reconhecidas na variável-alvo: {valores}")
    y = y.astype(int)

    atributos = prata.drop(columns=["id_lote", "resultado_inspecao"])
    X = pd.get_dummies(
        atributos,
        # aqui é com voce!!!
        drop_first=False,
        dtype=int,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        # aqui é com voce!!!
        # aqui é com voce!!!
        # aqui é com voce!!!
        stratify=y,
    )

    print(f"Atributos após transformação: {X.shape[1]}")
    print(f"Treino: {len(X_train)} registros | Teste: {len(X_test)} registros")
    print("\nDistribuição da variável-alvo no conjunto completo:")
    print(y.value_counts().rename(index={0: "conforme", 1: "retrabalho"}).to_string())
    print("\nAlguns atributos gerados:")
    for nome in list(X.columns)[:15]:
        print("  -", nome)

    return X, y, X_train, X_test, y_train, y_test


def etapa_4_mineracao(X_train, X_test, y_train):
    titulo("KDD 4 — MINERAÇÃO / ÁRVORE DE DECISÃO")

    modelo = DecisionTreeClassifier(
        # aqui é com voce!!!
        # aqui é com voce!!!
    )
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    print("Modelo treinado: DecisionTreeClassifier(max_depth=3, random_state=42)")
    return modelo, y_pred


def etapa_5_interpretacao(
    prata: pd.DataFrame,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    modelo: DecisionTreeClassifier,
    conn: sqlite3.Connection,
) -> None:
    titulo("KDD 5 — INTERPRETAÇÃO / CAMADA OURO")

    labels = [0, 1]
    nomes = ["conforme", "retrabalho"]
    acuracia = accuracy_score(y_test, y_pred)
    matriz = confusion_matrix(y_test, y_pred, labels=labels)
    relatorio = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=nomes,
        zero_division=0,
        output_dict=True,
    )

    print(f"Acurácia do conjunto de teste: {acuracia:.2%}")
    print("\nMatriz de confusão [linhas=real, colunas=previsto]:")
    matriz_df = pd.DataFrame(
        matriz,
        index=["real_conforme", "real_retrabalho"],
        columns=["prev_conforme", "prev_retrabalho"],
    )
    print(matriz_df.to_string())

    tn, fp, fn, tp = matriz.ravel()
    print("\nLeitura dos erros:")
    print(f"  Conforme → conforme (acerto): {tn}")
    print(f"  Conforme → retrabalho (falso positivo): {fp}")
    print(f"  Retrabalho → conforme (falso negativo): {fn}")
    print(f"  Retrabalho → retrabalho (acerto): {tp}")

    # Importância dos atributos.
    importancias = (
        pd.DataFrame({
            "atributo": X.columns,
            "importancia": modelo.feature_importances_,
        })
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    print("\nAtributos mais influentes na árvore:")
    print(importancias.head(10).to_string(index=False))

    # Camada ouro principal: colunas refinadas para acompanhamento de qualidade.
    ouro_lotes = prata[[
        "id_lote", "linha", "turno", "fornecedor", "resultado_inspecao"
    ]].copy()

    indicadores_linha = (
        prata.groupby("linha", dropna=False)["resultado_inspecao"]
        .apply(lambda x: (x == "retrabalho").mean())
        .reset_index(name="taxa_retrabalho")
        .sort_values("taxa_retrabalho", ascending=False)
    )

    # Relacionar previsões de teste ao id_lote usando o índice original preservado no split.
    resultados_teste = pd.DataFrame({
        "id_lote": prata.loc[X_test.index, "id_lote"].values,
        "real": pd.Series(y_test.values).map({0: "conforme", 1: "retrabalho"}),
        "previsto": pd.Series(y_pred).map({0: "conforme", 1: "retrabalho"}),
    })
    resultados_teste["acertou"] = resultados_teste["real"] == resultados_teste["previsto"]

    relatorio_df = pd.DataFrame(relatorio).T.reset_index(names="classe_ou_media")
    metricas = pd.DataFrame({
        "metrica": ["acuracia", "falso_positivo", "falso_negativo", "verdadeiro_negativo", "verdadeiro_positivo"],
        "valor": [float(acuracia), int(fp), int(fn), int(tn), int(tp)],
    })

    salvar_tabela(ouro_lotes, "ouro_lotes_qualidade", conn)
    salvar_tabela(indicadores_linha, "ouro_indicadores_linha", conn)
    salvar_tabela(resultados_teste, "ouro_resultado_teste", conn)
    salvar_tabela(importancias, "ouro_importancia_atributos", conn)
    salvar_tabela(relatorio_df, "ouro_relatorio_classificacao", conn)
    salvar_tabela(metricas, "ouro_metricas_modelo", conn)

    # Arquivos CSV facilitam a inspeção sem uma ferramenta SQLite.
    ouro_lotes.to_csv(PROCESSED_DIR / "ouro_lotes_qualidade.csv", index=False, encoding="utf-8-sig")
    indicadores_linha.to_csv(PROCESSED_DIR / "ouro_indicadores_linha.csv", index=False, encoding="utf-8-sig")
    resultados_teste.to_csv(PROCESSED_DIR / "ouro_resultado_teste.csv", index=False, encoding="utf-8-sig")
    importancias.to_csv(PROCESSED_DIR / "ouro_importancia_atributos.csv", index=False, encoding="utf-8-sig")
    metricas.to_csv(PROCESSED_DIR / "ouro_metricas_modelo.csv", index=False, encoding="utf-8-sig")

    # Regras textuais da árvore: úteis para interpretar o que foi aprendido.
    regras = export_text(modelo, feature_names=list(X.columns))
    (PROCESSED_DIR / "regras_arvore.txt").write_text(regras, encoding="utf-8")

    print("\nTaxa de retrabalho por linha:")
    print(indicadores_linha.to_string(index=False, formatters={"taxa_retrabalho": "{:.1%}".format}))
    print("\nArquivos gerados em data/processed e tabelas ouro salvas no SQLite.")


def main() -> None:
    titulo("AULA 02 — KDD E PIPELINE DE DADOS COM VS CODE")
    print("Objetivo: repetir em código o fluxo visto no Orange e persistir bronze, prata e ouro.")
    print(f"Banco SQLite: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        bronze = etapa_1_selecao_e_bronze(conn)
        prata = etapa_2_preprocessamento(bronze, conn)
        X, y, X_train, X_test, y_train, y_test = etapa_3_transformacao(prata)
        modelo, y_pred = etapa_4_mineracao(X_train, X_test, y_train)
        etapa_5_interpretacao(prata, X, X_test, y_test, y_pred, modelo, conn)

    titulo("PIPELINE CONCLUÍDO")
    print("Confira:")
    print(f"  - Banco: {DB_PATH}")
    print(f"  - Arquivos processados: {PROCESSED_DIR}")
    print("  - Próximo passo: abra o arquivo ROTEIRO_AULA.md e responda às questões de interpretação.")


if __name__ == "__main__":
    main()
