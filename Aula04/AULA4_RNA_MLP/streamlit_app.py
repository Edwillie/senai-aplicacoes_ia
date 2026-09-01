from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Aula 03 | Redes Neurais MLP",
    page_icon="🧠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DB_PATH = BASE_DIR / "database" / "aula3_pipeline.db"

COMPARACAO_PATH = PROCESSED_DIR / "ouro_comparacao_modelos.csv"
METRICAS_PATH = PROCESSED_DIR / "ouro_metricas_modelos.csv"
RESULTADOS_PATH = PROCESSED_DIR / "ouro_resultados_teste.csv"
HISTORICO_PATH = PROCESSED_DIR / "historico_treinamento_keras.csv"
PRATA_PATH = PROCESSED_DIR / "qualidade_prata.csv"

# -----------------------------------------------------------------------------
# ESTILO
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            color: #153b5b;
        }

        .senai-header {
            border-left: 8px solid #e30613;
            padding: 0.4rem 0 0.4rem 1rem;
            margin-bottom: 1rem;
        }

        .senai-header h1 {
            margin: 0;
            color: #153b5b;
            font-size: 2.1rem;
        }

        .senai-header p {
            margin: 0.35rem 0 0 0;
            color: #4b5563;
        }

        .concept-box {
            border: 1px solid #d7dde5;
            border-radius: 10px;
            padding: 1rem 1.1rem;
            background: #ffffff;
            min-height: 155px;
        }

        .concept-box strong {
            color: #153b5b;
        }

        .reflection {
            border-left: 5px solid #e30613;
            background: #f7f8fa;
            padding: 0.9rem 1rem;
            margin-bottom: 0.6rem;
            border-radius: 0 8px 8px 0;
        }

        .small-note {
            color: #5f6873;
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# LEITURA DOS DADOS
# -----------------------------------------------------------------------------
@st.cache_data
def ler_csv(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho)


@st.cache_data
def ler_tabela_sqlite(nome: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(f"SELECT * FROM {nome}", conn)


def validar_arquivos() -> None:
    obrigatorios = [
        COMPARACAO_PATH,
        METRICAS_PATH,
        RESULTADOS_PATH,
    ]

    faltando = [str(p) for p in obrigatorios if not p.exists()]

    if faltando:
        st.error(
            "Os arquivos de resultado ainda não foram encontrados. "
            "Execute primeiro o pipeline da Aula 03:\n\n"
            "`python src\\pipeline_rna_mlp_aula3.py`"
        )
        with st.expander("Arquivos esperados"):
            for item in obrigatorios:
                st.code(str(item))
        st.stop()


validar_arquivos()

comparacao = ler_csv(COMPARACAO_PATH)
metricas = ler_csv(METRICAS_PATH)
resultados = ler_csv(RESULTADOS_PATH)

historico = (
    ler_csv(HISTORICO_PATH)
    if HISTORICO_PATH.exists()
    else pd.DataFrame()
)

prata = (
    ler_csv(PRATA_PATH)
    if PRATA_PATH.exists()
    else pd.DataFrame()
)

bronze = ler_tabela_sqlite("bronze_qualidade_pecas")
indicadores_linha = ler_tabela_sqlite("ouro_indicadores_linha")

# Garantir tipos adequados para exibição no Streamlit/Arrow.
for df in [comparacao, metricas, resultados, historico, prata, bronze, indicadores_linha]:
    if not df.empty:
        for coluna in df.select_dtypes(include=["object"]).columns:
            df[coluna] = df[coluna].astype("string")

# -----------------------------------------------------------------------------
# CABEÇALHO
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="senai-header">
        <h1>Aula 03 — Redes Neurais MLP</h1>
        <p>Comparação entre Árvore de Decisão, MLP scikit-learn e MLP TensorFlow/Keras</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Nesta aplicação nós mantemos o mesmo problema e o mesmo pipeline da Aula 02, "
    "mas trocamos o mecanismo de aprendizagem. O objetivo é comparar desempenho, "
    "tipos de erro e comportamento de treinamento."
)

# -----------------------------------------------------------------------------
# VISÃO GERAL
# -----------------------------------------------------------------------------
st.subheader("1. Visão geral do experimento")

c1, c2, c3, c4 = st.columns(4)

qtd_bronze = len(bronze) if not bronze.empty else 99
qtd_prata = len(prata) if not prata.empty else 96

with c1:
    st.metric("Registros brutos", qtd_bronze)

with c2:
    st.metric("Registros após limpeza", qtd_prata)

with c3:
    st.metric("Registros de teste", resultados["id_lote"].nunique())

with c4:
    st.metric("Modelos comparados", comparacao["modelo"].nunique())

st.caption(
    "A base foi dividida em treino e teste. O conjunto de teste não participa "
    "do aprendizado e é usado para verificar generalização."
)

# -----------------------------------------------------------------------------
# PIPELINE
# -----------------------------------------------------------------------------
st.subheader("2. O que mudou da Aula 02 para a Aula 03?")

p1, p2, p3, p4, p5 = st.columns(5)

with p1:
    st.markdown(
        """
        <div class="concept-box">
        <strong>Bronze</strong><br><br>
        Dado bruto preservado, exatamente como chegou.
        </div>
        """,
        unsafe_allow_html=True,
    )

with p2:
    st.markdown(
        """
        <div class="concept-box">
        <strong>Prata</strong><br><br>
        Dado limpo, padronizado e preparado para análise.
        </div>
        """,
        unsafe_allow_html=True,
    )

with p3:
    st.markdown(
        """
        <div class="concept-box">
        <strong>Transformação</strong><br><br>
        Criação de X e y, codificação das categorias e separação treino/teste.
        </div>
        """,
        unsafe_allow_html=True,
    )

with p4:
    st.markdown(
        """
        <div class="concept-box">
        <strong>StandardScaler</strong><br><br>
        Coloca os atributos em escalas comparáveis antes do treinamento da MLP.
        </div>
        """,
        unsafe_allow_html=True,
    )

with p5:
    st.markdown(
        """
        <div class="concept-box">
        <strong>Modelos</strong><br><br>
        Árvore, MLP scikit-learn e MLP TensorFlow/Keras.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# COMPARAÇÃO DOS MODELOS
# -----------------------------------------------------------------------------
st.subheader("3. Comparação dos modelos")

comparacao_visual = comparacao.copy()
comparacao_visual["acuracia_pct"] = comparacao_visual["acuracia"] * 100

# Melhor modelo
melhor = comparacao_visual.sort_values(
    ["acuracia", "falso_negativo"],
    ascending=[False, True],
).iloc[0]

m1, m2, m3 = st.columns(3)

for coluna, (_, linha) in zip(
    [m1, m2, m3],
    comparacao_visual.sort_values("acuracia", ascending=False).iterrows(),
):
    with coluna:
        st.metric(
            label=str(linha["modelo"]),
            value=f'{linha["acuracia"]:.2%}',
        )
        st.caption(
            f'FP: {int(linha["falso_positivo"])} | '
            f'FN: {int(linha["falso_negativo"])}'
        )

st.success(
    f'Neste experimento, o melhor resultado foi obtido por **{melhor["modelo"]}**, '
    f'com **{melhor["acuracia"]:.2%} de acurácia**.'
)

g1, g2 = st.columns(2)

with g1:
    st.markdown("#### Acurácia")
    grafico_acuracia = (
        comparacao_visual[["modelo", "acuracia_pct"]]
        .set_index("modelo")
    )
    st.bar_chart(
        grafico_acuracia,
        y_label="Acurácia (%)",
        x_label="Modelo",
    )

with g2:
    st.markdown("#### Tipos de erro")
    erros = (
        comparacao_visual[
            ["modelo", "falso_positivo", "falso_negativo"]
        ]
        .set_index("modelo")
    )
    st.bar_chart(
        erros,
        y_label="Quantidade",
        x_label="Modelo",
    )

st.info(
    "Acurácia responde quantas previsões foram corretas. "
    "Para uma aplicação industrial, também precisamos observar **como o modelo errou**. "
    "Um falso negativo significa que um lote de retrabalho foi previsto como conforme."
)

# -----------------------------------------------------------------------------
# MATRIZES DE CONFUSÃO
# -----------------------------------------------------------------------------
st.subheader("4. Matriz de confusão por modelo")

modelos = resultados["modelo"].dropna().unique().tolist()

tabs = st.tabs(modelos)

for tab, modelo in zip(tabs, modelos):
    with tab:
        dados_modelo = resultados[resultados["modelo"] == modelo].copy()

        matriz = pd.crosstab(
            dados_modelo["real"],
            dados_modelo["previsto"],
            rownames=["Classe real"],
            colnames=["Classe prevista"],
            dropna=False,
        )

        # Garantir a ordem das classes.
        matriz = matriz.reindex(
            index=["conforme", "retrabalho"],
            columns=["conforme", "retrabalho"],
            fill_value=0,
        )

        a, b = st.columns([1, 1])

        with a:
            st.dataframe(
                matriz,
                width="stretch",
            )

        with b:
            total = len(dados_modelo)
            erros_total = int((dados_modelo["real"] != dados_modelo["previsto"]).sum())
            acertos_total = total - erros_total

            st.metric("Acertos", f"{acertos_total}/{total}")
            st.metric("Erros", erros_total)

            fn = int(
                (
                    (dados_modelo["real"] == "retrabalho")
                    & (dados_modelo["previsto"] == "conforme")
                ).sum()
            )

            fp = int(
                (
                    (dados_modelo["real"] == "conforme")
                    & (dados_modelo["previsto"] == "retrabalho")
                ).sum()
            )

            st.write(f"**Falso positivo:** {fp}")
            st.write(f"**Falso negativo:** {fn}")

# -----------------------------------------------------------------------------
# TREINAMENTO KERAS
# -----------------------------------------------------------------------------
st.subheader("5. Como a MLP TensorFlow/Keras aprendeu ao longo das épocas?")

if not historico.empty:
    total_epocas = int(historico["epoca"].max())

    h1, h2, h3 = st.columns(3)

    with h1:
        st.metric("Épocas executadas", total_epocas)

    with h2:
        st.metric(
            "Loss final",
            f'{historico["loss"].iloc[-1]:.4f}',
        )

    with h3:
        if "val_loss" in historico.columns:
            st.metric(
                "Validation loss final",
                f'{historico["val_loss"].iloc[-1]:.4f}',
            )

    curva_acc_cols = [
        c for c in ["accuracy", "val_accuracy"]
        if c in historico.columns
    ]

    st.markdown("#### Acurácia por época")
    if curva_acc_cols:
        st.line_chart(
            historico.set_index("epoca")[curva_acc_cols],
            y_label="Acurácia",
            x_label="Época",
        )
    else:
        st.info("O histórico não contém colunas de acurácia.")

    st.caption(
        "A rede ajusta seus pesos ao longo das épocas. "
        "O EarlyStopping interrompe o treinamento quando a validação deixa de melhorar "
        "por várias épocas consecutivas."
    )
else:
    st.warning(
        "O arquivo historico_treinamento_keras.csv não foi encontrado. "
        "Execute novamente o pipeline com TensorFlow/Keras disponível."
    )

# -----------------------------------------------------------------------------
# ARQUITETURA
# -----------------------------------------------------------------------------
st.subheader("6. Arquitetura da MLP utilizada")

st.graphviz_chart(
    """
    digraph MLP {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#f5f7fa", color="#153b5b"];

        X [label="15 atributos\\nde entrada"];
        S [label="StandardScaler"];
        H1 [label="Camada oculta\\n16 neurônios\\nReLU"];
        H2 [label="Camada oculta\\n8 neurônios\\nReLU"];
        O [label="Saída\\n1 neurônio\\nSigmoid"];
        C [label="Classe\\nConforme / Retrabalho"];

        X -> S -> H1 -> H2 -> O -> C;
    }
    """,
    width="stretch",
)

st.write(
    "A camada de saída possui um neurônio com função **sigmoid**, porque o problema possui "
    "duas classes. A saída é interpretada como uma probabilidade entre 0 e 1 e, no projeto, "
    "utilizamos 0,5 como limiar para decidir entre conforme e retrabalho."
)

# -----------------------------------------------------------------------------
# RESULTADOS INDIVIDUAIS
# -----------------------------------------------------------------------------
st.subheader("7. Quais registros cada modelo acertou ou errou?")

modelo_selecionado = st.selectbox(
    "Escolha um modelo",
    modelos,
)

detalhes = resultados[resultados["modelo"] == modelo_selecionado].copy()

filtro = st.radio(
    "Mostrar",
    ["Todos", "Somente erros", "Somente acertos"],
    horizontal=True,
)

if filtro == "Somente erros":
    detalhes = detalhes[detalhes["acertou"] == False]
elif filtro == "Somente acertos":
    detalhes = detalhes[detalhes["acertou"] == True]

st.dataframe(
    detalhes,
    width="stretch",
    hide_index=True,
)

# -----------------------------------------------------------------------------
# DADOS BRONZE E PRATA
# -----------------------------------------------------------------------------
st.subheader("8. Do dado bruto ao dado tratado")

d1, d2 = st.columns(2)

with d1:
    st.markdown("#### Camada Bronze")
    if not bronze.empty:
        st.dataframe(
            bronze.head(10),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Tabela Bronze não encontrada no SQLite.")

with d2:
    st.markdown("#### Camada Prata")
    if not prata.empty:
        st.dataframe(
            prata.head(10),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Arquivo da camada Prata não encontrado.")

# -----------------------------------------------------------------------------
# REFLEXÕES FINAIS
# -----------------------------------------------------------------------------
st.subheader("9. Reflexões")

reflexoes = [
    (
        "O pipeline permaneceu; o algoritmo mudou.",
        "Seleção, pré-processamento, transformação, treino, teste e interpretação "
        "continuam fazendo parte do problema independentemente do modelo escolhido.",
    ),
    (
        "A MLP precisou de escalonamento.",
        "Como a rede ajusta pesos numericamente, atributos em escalas muito diferentes "
        "podem prejudicar o treinamento. Por isso usamos StandardScaler.",
    ),
    (
        "Modelo mais complexo não significa modelo melhor.",
        "Neste conjunto pequeno e tabular, a árvore de decisão superou as duas MLPs. "
        "Isso é um resultado válido e deve ser interpretado, não escondido.",
    ),
    (
        "Acurácia sozinha não basta.",
        "O falso negativo é especialmente importante porque representa um lote de retrabalho "
        "classificado como conforme.",
    ),
]

for titulo_ref, texto_ref in reflexoes:
    st.markdown(
        f"""
        <div class="reflection">
            <strong>{titulo_ref}</strong><br>
            {texto_ref}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "Pós-graduação em Inteligência Artificial | "
    "UC Aplicações em Inteligência Artificial"
)
