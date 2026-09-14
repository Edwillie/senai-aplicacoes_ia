from __future__ import annotations

from pathlib import Path
import sqlite3

import joblib
import pandas as pd
import streamlit as st


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

st.set_page_config(
    page_title="Manutenção Preditiva | SENAI-SP",
    page_icon="SENAI",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DB_PATH = BASE_DIR / "database" / "aula5_pipeline.db"

COMPARACAO_PATH = PROCESSED_DIR / "ouro_comparacao_modelos.csv"
HISTORICO_KERAS_PATH = PROCESSED_DIR / "historico_treinamento_keras.csv"
ARTEFATOS_PATH = BASE_DIR / "modelos" / "modelos_aula5.joblib"


# =============================================================================
# ESTILO
# =============================================================================

st.markdown(
    """
    <style>
        :root {
            --navy: #102a43;
            --red: #e30613;
            --ink: #1f2937;
            --muted: #64748b;
            --line: #e6eaf0;
            --canvas: #f6f8fb;
            --white: #ffffff;
        }

        .stApp {
            background: var(--canvas);
            color: var(--ink);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
        }

        h1, h2, h3 {
            color: var(--navy);
            letter-spacing: -0.025em;
        }

        h1 a, h2 a, h3 a {
            display: none !important;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }

        .hero {
            background: linear-gradient(120deg, #102a43 0%, #173b61 58%, #1e4c79 100%);
            border-radius: 18px;
            color: white;
            padding: 2rem 2.2rem;
            margin-bottom: 1.3rem;
            box-shadow: 0 14px 32px rgba(16, 42, 67, 0.16);
        }

        .hero-label {
            color: #ffb6bc;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .hero h1 {
            color: white;
            margin: 0;
            font-size: 2rem;
        }

        .hero p {
            color: #d9e6f2;
            margin: 0.7rem 0 0;
            max-width: 800px;
            line-height: 1.55;
        }

        .section-intro {
            color: var(--muted);
            max-width: 900px;
            line-height: 1.6;
            margin-bottom: 1rem;
        }

        .kpi {
            background: var(--white);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            min-height: 110px;
            box-shadow: 0 3px 12px rgba(16, 42, 67, 0.04);
        }

        .kpi-label {
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 0.45rem;
        }

        .kpi-value {
            color: var(--navy);
            font-size: 1.7rem;
            font-weight: 750;
        }

        .kpi-detail {
            color: var(--red);
            font-size: 0.82rem;
            font-weight: 600;
            margin-top: 0.35rem;
        }

        .flow-card, .model-card {
            background: var(--white);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1rem;
            min-height: 150px;
            box-shadow: 0 3px 12px rgba(16, 42, 67, 0.04);
        }

        .flow-number, .model-card small {
            color: var(--red);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .flow-card h4, .model-card h3 {
            color: var(--navy);
            margin: 0.45rem 0;
        }

        .flow-card p, .model-card p {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 0;
        }

        .callout {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 1rem 1.15rem;
            line-height: 1.6;
            margin: 1rem 0;
        }

        .callout-alert {
            background: #fff8f8;
            border-color: #f4c8cb;
        }

        .architecture {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.4rem;
            background: white;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1.2rem;
            overflow-x: auto;
            margin: 0.9rem 0;
        }

        .architecture-node {
            min-width: 130px;
            padding: 0.9rem 0.7rem;
            border-radius: 12px;
            border: 1px solid #d8e0ea;
            background: #f8fafc;
            color: var(--navy);
            font-size: 0.9rem;
            font-weight: 650;
            text-align: center;
            line-height: 1.4;
        }

        .architecture-arrow {
            color: var(--red);
            font-size: 1.35rem;
            font-weight: 700;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# FUNÇÕES
# =============================================================================

@st.cache_data
def ler_csv(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho)


@st.cache_data
def ler_tabela_sqlite(nome_tabela: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(f"SELECT * FROM {nome_tabela}", conn)


@st.cache_resource
def carregar_modelos():
    if not ARTEFATOS_PATH.exists():
        return None

    return joblib.load(ARTEFATOS_PATH)


def percentual(valor) -> str:
    if pd.isna(valor):
        return "-"

    return f"{valor:.2%}"


def numero(valor) -> str:
    return f"{int(valor):,}".replace(",", ".")


# =============================================================================
# CARREGAMENTO DOS DADOS
# =============================================================================

bronze = ler_tabela_sqlite("bronze_manutencao")
prata = ler_tabela_sqlite("prata_manutencao")

comparacao = (
    ler_csv(COMPARACAO_PATH)
    if COMPARACAO_PATH.exists()
    else pd.DataFrame()
)

historico_keras = (
    ler_csv(HISTORICO_KERAS_PATH)
    if HISTORICO_KERAS_PATH.exists()
    else pd.DataFrame()
)

artefatos = carregar_modelos()

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">SENAI-SP · Pós-graduação em Inteligência Artificial</div>
        <h1>Manutenção Preditiva Industrial</h1>
        <p>
            Um experimento de classificação para apoiar a identificação de
            possíveis falhas em equipamentos a partir de dados operacionais.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if comparacao.empty:
    st.error(
        "Os resultados ainda não foram encontrados. Execute primeiro o pipeline."
    )
    st.code("python src\\pipeline_mlp_aula5.py", language="bat")
    st.stop()

for coluna in [
    "acuracia",
    "falso_positivo",
    "falso_negativo",
    "recall_falha",
]:
    if coluna in comparacao.columns:
        comparacao[coluna] = pd.to_numeric(
            comparacao[coluna],
            errors="coerce",
        )


# =============================================================================
# BARRA LATERAL
# =============================================================================

with st.sidebar:
    st.markdown("## Painel da Aula 05")
    st.caption("Aplicações em Inteligência Artificial")

    st.divider()

    st.markdown(
        """
        **Etapas do experimento**

        01. Dados originais  
        02. Preparação  
        03. Treino e teste  
        04. Modelos  
        05. Simulação  
        06. Interpretação
        """
    )

    st.divider()

    if st.button("Atualizar resultados", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


# =============================================================================
# ABAS
# =============================================================================

aba_visao, aba_dados, aba_preparacao, aba_modelos, aba_simulador, aba_resultados, aba_keras = st.tabs(
    [
        "Visão geral",
        "Base de dados",
        "Preparação",
        "Modelos",
        "Simulador",
        "Resultados",
        "TensorFlow/Keras",
    ]
)


# =============================================================================
# VISÃO GERAL
# =============================================================================

with aba_visao:
    st.markdown("## O desafio de negócio")

    st.markdown(
        """
        <p class="section-intro">
            Identificar indícios de falha antes de uma parada não planejada pode
            apoiar a priorização de inspeções e intervenções. Neste experimento,
            os modelos classificam os registros como funcionamento normal ou
            possível falha.
        </p>
        """,
        unsafe_allow_html=True,
    )

    taxa_falha = (
        prata["machine_failure"].mean()
        if not prata.empty and "machine_failure" in prata.columns
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">Registros originais</div>
                <div class="kpi-value">{numero(len(bronze)) if not bronze.empty else "-"}</div>
                <div class="kpi-detail">Camada Bronze</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">Registros preparados</div>
                <div class="kpi-value">{numero(len(prata)) if not prata.empty else "-"}</div>
                <div class="kpi-detail">Camada Prata</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">Taxa de falha</div>
                <div class="kpi-value">{percentual(taxa_falha)}</div>
                <div class="kpi-detail">Variável-alvo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">Modelos avaliados</div>
                <div class="kpi-value">{comparacao["modelo"].nunique()}</div>
                <div class="kpi-detail">Camada Ouro</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Fluxo do projeto")

    f1, f2, f3, f4, f5 = st.columns(5)

    etapas = [
        ("01", "Bronze", "Preserva o arquivo original recebido."),
        ("02", "Prata", "Organiza e prepara os dados."),
        ("03", "Transformação", "Separa atributos, alvo, treino e teste."),
        ("04", "Modelos", "Compara árvore de decisão e MLP."),
        ("05", "Ouro", "Organiza métricas para interpretação."),
    ]

    for coluna, etapa in zip([f1, f2, f3, f4, f5], etapas):
        numero_etapa, titulo_etapa, descricao = etapa

        with coluna:
            st.markdown(
                f"""
                <div class="flow-card">
                    <div class="flow-number">{numero_etapa}</div>
                    <h4>{titulo_etapa}</h4>
                    <p>{descricao}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="callout callout-alert">
            <strong>Leitura crítica:</strong> como a maior parte dos registros
            representa funcionamento normal, acurácia elevada não garante que um
            modelo seja eficiente para reconhecer falhas.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# BASE DE DADOS
# =============================================================================

with aba_dados:
    st.markdown("## Base de dados")

    st.markdown(
        """
        <p class="section-intro">
            A base representa um cenário industrial de manutenção preditiva.
            Cada linha reúne características do processo, como temperaturas,
            torque, velocidade de rotação e desgaste de ferramenta.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if not bronze.empty:
        st.markdown("### Dados originais")
        st.dataframe(
            bronze.head(20),
            width="stretch",
            hide_index=True,
        )

    if not prata.empty:
        st.markdown("### Dados preparados")
        st.dataframe(
            prata.head(20),
            width="stretch",
            hide_index=True,
        )

        if "machine_failure" in prata.columns:
            distribuicao = (
                prata["machine_failure"]
                .value_counts()
                .rename_axis("classe")
                .reset_index(name="quantidade")
            )

            distribuicao["classe"] = distribuicao["classe"].map(
                {
                    0: "Funcionamento normal",
                    1: "Falha",
                }
            )

            st.markdown("### Distribuição das classes")
            st.bar_chart(
                distribuicao.set_index("classe")["quantidade"],
                y_label="Quantidade de registros",
            )


# =============================================================================
# PREPARAÇÃO
# =============================================================================

with aba_preparacao:
    st.markdown("## Preparação dos dados")

    st.markdown(
        """
        <p class="section-intro">
            Antes do treino, o pipeline seleciona os atributos adequados,
            remove informações que antecipariam a resposta e divide os dados
            entre treino e teste.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Colunas removidas antes do treino")

    st.code(
        """colunas_a_remover = [
    "udi",
    "product_id",
    "machine_failure",
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
]""",
        language="python",
    )

    st.markdown(
        """
        <div class="callout callout-alert">
            <strong>Vazamento de dados:</strong> a coluna
            <code>machine_failure</code> é o resultado que desejamos prever.
            Já TWF, HDF, PWF, OSF e RNF descrevem tipos de falha e poderiam
            antecipar indevidamente a resposta ao modelo.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Categoria de produto")

        st.code(
            """X = pd.get_dummies(
    atributos,
    columns=["type"],
    drop_first=False,
    dtype=int,
)""",
            language="python",
        )

    with c2:
        st.markdown("### Treino e teste")

        st.code(
            """X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)""",
            language="python",
        )

    st.markdown("### Padronização para a MLP")

    st.code(
        """scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)""",
        language="python",
    )


# =============================================================================
# MODELOS
# =============================================================================

with aba_modelos:
    st.markdown("## Modelos avaliados")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="model-card">
                <small>Referência</small>
                <h3>Árvore de Decisão</h3>
                <p>
                    Modelo baseline utilizado para comparar os resultados das
                    redes neurais.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="model-card">
                <small>Rede neural</small>
                <h3>MLP scikit-learn</h3>
                <p>
                    Rede com duas camadas ocultas: a primeira com 16 neurônios
                    e a segunda com 8 neurônios.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="model-card">
                <small>Etapa complementar</small>
                <h3>MLP TensorFlow/Keras</h3>
                <p>
                    Segunda implementação de rede neural, executada quando o
                    TensorFlow está disponível no ambiente.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Estrutura da MLP")

    st.markdown(
        """
        <div class="architecture">
            <div class="architecture-node">Atributos<br>de entrada</div>
            <div class="architecture-arrow">→</div>
            <div class="architecture-node">Dados<br>padronizados</div>
            <div class="architecture-arrow">→</div>
            <div class="architecture-node">Camada oculta<br>16 neurônios</div>
            <div class="architecture-arrow">→</div>
            <div class="architecture-node">Camada oculta<br>8 neurônios</div>
            <div class="architecture-arrow">→</div>
            <div class="architecture-node">Estimativa<br>de falha</div>
            <div class="architecture-arrow">→</div>
            <div class="architecture-node">Normal<br>ou falha</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SIMULADOR
# =============================================================================

with aba_simulador:
    st.markdown("## Simulador de manutenção preditiva")

    st.markdown(
        """
        <p class="section-intro">
            Insira novos valores de operação e envie-os aos modelos treinados.
            A aplicação realiza a mesma preparação usada no treinamento e
            apresenta a previsão obtida por cada modelo.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if artefatos is None:
        st.warning(
            "Os modelos ainda não foram encontrados. Execute novamente o pipeline "
            "para gerar o arquivo modelos/modelos_aula5.joblib."
        )

    else:
        with st.form("formulario_simulador"):
            c1, c2, c3 = st.columns(3)

            with c1:
                tipo = st.selectbox(
                    "Variante de qualidade do produto",
                    options=["L", "M", "H"],
                )

                temperatura_ar = st.number_input(
                    "Temperatura do ar (K)",
                    min_value=280.0,
                    max_value=330.0,
                    value=300.0,
                    step=0.1,
                )

            with c2:
                temperatura_processo = st.number_input(
                    "Temperatura do processo (K)",
                    min_value=290.0,
                    max_value=340.0,
                    value=310.0,
                    step=0.1,
                )

                velocidade = st.number_input(
                    "Velocidade de rotação (rpm)",
                    min_value=100.0,
                    max_value=3500.0,
                    value=1500.0,
                    step=10.0,
                )

            with c3:
                torque = st.number_input(
                    "Torque (Nm)",
                    min_value=0.0,
                    max_value=100.0,
                    value=40.0,
                    step=0.1,
                )

                desgaste = st.number_input(
                    "Desgaste da ferramenta (min)",
                    min_value=0.0,
                    max_value=300.0,
                    value=100.0,
                    step=1.0,
                )

            prever = st.form_submit_button(
                "Executar previsão",
                width="stretch",
            )

        if prever:
            nova_entrada = pd.DataFrame(
                [
                    {
                        "air_temperature": temperatura_ar,
                        "process_temperature": temperatura_processo,
                        "rotational_speed": velocidade,
                        "torque": torque,
                        "tool_wear": desgaste,
                        "type": tipo,
                    }
                ]
            )

            entrada_transformada = pd.get_dummies(
                nova_entrada,
                columns=["type"],
                drop_first=False,
                dtype=int,
            )

            entrada_transformada = entrada_transformada.reindex(
                columns=artefatos["colunas_modelo"],
                fill_value=0,
            )

            previsao_arvore = artefatos["modelo_arvore"].predict(
                entrada_transformada
            )[0]

            entrada_escalonada = artefatos["scaler"].transform(
                entrada_transformada
            )

            previsao_mlp = artefatos["modelo_mlp"].predict(
                entrada_escalonada
            )[0]

            probabilidade_mlp = artefatos["modelo_mlp"].predict_proba(
                entrada_escalonada
            )[0][1]

            resultado_arvore = (
                "Possível falha"
                if previsao_arvore == 1
                else "Funcionamento normal"
            )

            resultado_mlp = (
                "Possível falha"
                if previsao_mlp == 1
                else "Funcionamento normal"
            )

            st.markdown("### Resultado da simulação")

            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric("Árvore de Decisão", resultado_arvore)

            with r2:
                st.metric("MLP scikit-learn", resultado_mlp)

            with r3:
                st.metric(
                    "Probabilidade estimada de falha",
                    f"{probabilidade_mlp:.2%}",
                )

            if previsao_mlp == 1:
                st.error(
                    "A MLP identificou um cenário compatível com possível falha. "
                    "Em uma operação real, esse resultado indicaria necessidade "
                    "de avaliação técnica."
                )
            else:
                st.success(
                    "A MLP classificou o cenário como funcionamento normal. "
                    "A previsão deve ser analisada junto ao histórico de "
                    "manutenção e às condições reais de operação."
                )

    # Esta explicação fica no fim da aba, fora do formulário e fora da previsão.
    st.divider()

    st.markdown("### O que representa a variante de qualidade do produto?")

    st.markdown(
        """
As letras representam uma **variante de qualidade do produto** presente no dataset:

- **L — Low:** qualidade baixa;
- **M — Medium:** qualidade média;
- **H — High:** qualidade alta.

Elas **não representam baixo, médio ou alto risco de falha**. São uma
característica do produto e do processo, analisada pelo modelo junto com
temperatura, torque, velocidade de rotação e desgaste da ferramenta.

No dataset, as variantes também influenciam o desgaste: produtos **H, M e L**
acrescentam, respectivamente, **5, 3 e 2 minutos** de desgaste à ferramenta
durante o processo.
        """
    )


# =============================================================================
# RESULTADOS
# =============================================================================

with aba_resultados:
    st.markdown("## Resultados do experimento")

    melhor_acuracia = comparacao.sort_values(
        "acuracia",
        ascending=False,
    ).iloc[0]

    melhor_recall = comparacao.sort_values(
        "recall_falha",
        ascending=False,
    ).iloc[0]

    menor_fn = comparacao.sort_values(
        "falso_negativo",
        ascending=True,
    ).iloc[0]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Maior acurácia",
            percentual(melhor_acuracia["acuracia"]),
            melhor_acuracia["modelo"],
        )

    with c2:
        st.metric(
            "Maior recall de falha",
            percentual(melhor_recall["recall_falha"]),
            melhor_recall["modelo"],
        )

    with c3:
        st.metric(
            "Menos falsos negativos",
            int(menor_fn["falso_negativo"]),
            menor_fn["modelo"],
        )

    st.markdown("### Comparativo completo")

    comparacao_exibicao = comparacao.copy()
    comparacao_exibicao["acuracia"] = comparacao_exibicao["acuracia"].map(percentual)
    comparacao_exibicao["recall_falha"] = comparacao_exibicao["recall_falha"].map(percentual)

    st.dataframe(
        comparacao_exibicao,
        width="stretch",
        hide_index=True,
    )

    g1, g2 = st.columns(2)

    with g1:
        st.markdown("### Acurácia e recall de falha")

        metricas = comparacao[
            ["modelo", "acuracia", "recall_falha"]
        ].set_index("modelo")

        st.bar_chart(metricas, y_label="Valor da métrica")

    with g2:
        st.markdown("### Tipos de erro")

        erros = comparacao[
            ["modelo", "falso_positivo", "falso_negativo"]
        ].set_index("modelo")

        st.bar_chart(erros, y_label="Quantidade de registros")

    st.markdown(
        """
        <div class="callout callout-alert">
            <strong>Falso positivo:</strong> o modelo prevê falha, mas a máquina
            estava normal. Pode gerar uma inspeção adicional.<br><br>
            <strong>Falso negativo:</strong> o modelo prevê funcionamento normal,
            mas havia falha. É o erro mais crítico neste cenário, pois uma
            intervenção necessária pode não ocorrer a tempo.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# TENSORFLOW / KERAS
# =============================================================================

with aba_keras:
    st.markdown("## Treinamento com TensorFlow/Keras")

    if historico_keras.empty:
        st.info(
            "Não há histórico TensorFlow/Keras disponível. Isso ocorre quando "
            "essa etapa não foi executada ou quando TensorFlow não está instalado."
        )

    else:
        historico = historico_keras.copy()
        historico.index = historico.index + 1
        historico.index.name = "época"

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Épocas executadas", len(historico))

        with c2:
            st.metric(
                "Loss final",
                f"{historico['loss'].iloc[-1]:.4f}",
            )

        with c3:
            if "val_loss" in historico.columns:
                st.metric(
                    "Validation loss final",
                    f"{historico['val_loss'].iloc[-1]:.4f}",
                )

        st.markdown("### Evolução da perda")

        colunas_loss = [
            coluna
            for coluna in ["loss", "val_loss"]
            if coluna in historico.columns
        ]

        st.line_chart(
            historico[colunas_loss],
            y_label="Loss",
        )

        if "accuracy" in historico.columns:
            st.markdown("### Evolução da acurácia")

            colunas_acuracia = [
                coluna
                for coluna in ["accuracy", "val_accuracy"]
                if coluna in historico.columns
            ]

            st.line_chart(
                historico[colunas_acuracia],
                y_label="Acurácia",
            )