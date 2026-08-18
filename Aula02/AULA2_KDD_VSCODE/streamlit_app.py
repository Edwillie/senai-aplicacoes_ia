from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "aula2_pipeline.db"

st.set_page_config(
    page_title="Painel KDD — Qualidade Industrial",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2.5rem;}
        .kdd-card {
            border: 1px solid #d9d9d9;
            border-radius: 14px;
            padding: 16px 18px;
            min-height: 132px;
            background: rgba(250,250,250,0.65);
        }
        .kdd-card h4 {margin: 0 0 8px 0;}
        .kdd-card p {margin: 4px 0; line-height: 1.35;}
        .kdd-step {
            border-left: 5px solid #1f4e79;
            border-radius: 10px;
            padding: 12px 14px;
            background: rgba(31,78,121,0.06);
            min-height: 132px;
        }
        .ok-card {
            border-left: 6px solid #2e7d32;
            background: rgba(46,125,50,0.08);
        }
        .warn-card {
            border-left: 6px solid #ed6c02;
            background: rgba(237,108,2,0.08);
        }
        .danger-card {
            border-left: 6px solid #c62828;
            background: rgba(198,40,40,0.08);
        }
        .info-card {
            border-left: 6px solid #1565c0;
            background: rgba(21,101,192,0.08);
        }
        .big-number {
            font-size: 2rem;
            font-weight: 700;
            margin: 2px 0 4px 0;
        }
        .small-muted {
            opacity: 0.72;
            font-size: 0.92rem;
        }
        .review-box {
            border: 1px dashed #9e9e9e;
            border-radius: 12px;
            padding: 14px 16px;
            margin: 8px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def ler_tabela(conn: sqlite3.Connection, nome: str) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {nome}", conn)


def carregar_dados():
    if not DB_PATH.exists():
        return None

    with sqlite3.connect(DB_PATH) as conn:
        nomes = {
            "bronze": "bronze_qualidade_pecas",
            "prata": "prata_qualidade_pecas",
            "ouro_lotes": "ouro_lotes_qualidade",
            "indicadores": "ouro_indicadores_linha",
            "resultados": "ouro_resultado_teste",
            "importancias": "ouro_importancia_atributos",
            "metricas": "ouro_metricas_modelo",
        }
        return {chave: ler_tabela(conn, tabela) for chave, tabela in nomes.items()}


def card(titulo, valor, explicacao, classe="info-card"):
    st.markdown(
        f"""
        <div class="kdd-card {classe}">
            <h4>{titulo}</h4>
            <div class="big-number">{valor}</div>
            <p>{explicacao}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


dados = carregar_dados()

st.title("📊 Painel KDD — Qualidade Industrial")
st.caption(
    "Visualização das etapas do pipeline desenvolvido no VS Code: "
    "Bronze → Prata → Transformação → Modelo → Interpretação."
)

if dados is None:
    st.error("O banco SQLite ainda não foi encontrado.")
    st.code("python src\\pipeline_qualidade.py", language="bat")
    st.info(
        "Execute primeiro o pipeline. Ele criará o arquivo "
        "`database/aula2_pipeline.db`. Depois recarregue esta página."
    )
    st.stop()

bronze = dados["bronze"]
prata = dados["prata"]
ouro_lotes = dados["ouro_lotes"]
indicadores = dados["indicadores"]
resultados = dados["resultados"]
importancias = dados["importancias"]
metricas_df = dados["metricas"]

metricas = dict(zip(metricas_df["metrica"], metricas_df["valor"]))
acuracia = float(metricas.get("acuracia", 0))
fp = int(metricas.get("falso_positivo", 0))
fn = int(metricas.get("falso_negativo", 0))
tn = int(metricas.get("verdadeiro_negativo", 0))
tp = int(metricas.get("verdadeiro_positivo", 0))

resultados["acertou"] = resultados["acertou"].astype(bool)
acertos = int(resultados["acertou"].sum())
total_teste = len(resultados)

duplicatas = int(bronze.duplicated().sum())
ausencias_bronze = int(bronze.isna().sum().sum())
ausencias_prata = int(prata.isna().sum().sum())

positivas = importancias[importancias["importancia"] > 0].copy()
positivas["importancia_pct"] = positivas["importancia"] * 100

indicadores["taxa_retrabalho_pct"] = indicadores["taxa_retrabalho"] * 100
linha_critica = indicadores.sort_values("taxa_retrabalho", ascending=False).iloc[0]

with st.sidebar:
    st.header("Roteiro da prática")
    st.write(
        "1. Primeiro executamos o pipeline no terminal.\n\n"
        "2. Depois abrimos este painel.\n\n"
        "3. Começamos pela **Visão geral**.\n\n"
        "4. Em seguida analisamos **Modelo e matriz**.\n\n"
        "5. Terminamos em **Decisão de negócio**."
    )
    st.divider()
    st.caption("Banco carregado:")
    st.code(str(DB_PATH), language=None)
    st.caption(
        "Se alterar o CSV ou o pipeline, execute novamente "
        "`python src\\pipeline_qualidade.py` e atualize o navegador."
    )

abas = st.tabs(
    [
        "1️⃣ Visão geral",
        "2️⃣ Bronze → Prata",
        "3️⃣ Modelo e matriz",
        "4️⃣ Onde o modelo errou?",
        "5️⃣ Decisão de negócio",
        "6️⃣ Revisão do fluxo",
    ]
)

with abas[0]:
    st.subheader("O pipeline inteiro em uma única tela")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        card(
            "Camada Bronze",
            f"{len(bronze)} registros",
            "É o dado como chegou. Ainda contém duplicidades, grafias inconsistentes e ausências.",
            "info-card",
        )
    with c2:
        card(
            "Camada Prata",
            f"{len(prata)} registros",
            f"Após limpeza. Foram removidas {duplicatas} duplicidades e os valores foram padronizados.",
            "ok-card",
        )
    with c3:
        card(
            "Conjunto de teste",
            f"{total_teste} registros",
            "Esses registros foram reservados para verificar se o modelo funciona em dados que não usou para aprender.",
            "info-card",
        )
    with c4:
        card(
            "Acurácia",
            f"{acuracia:.1%}",
            f"O modelo acertou {acertos} de {total_teste} classificações no conjunto de teste.",
            "ok-card" if acuracia >= 0.8 else "warn-card",
        )

    st.divider()
    st.subheader("As 5 etapas do KDD neste projeto")

    cols = st.columns(5)
    etapas = [
        ("1. Seleção", "CSV → Bronze", "Carregamos os registros e preservamos o dado bruto."),
        ("2. Pré-processamento", "Bronze → Prata", "Removemos duplicidades, padronizamos e tratamos ausências."),
        ("3. Transformação", "Dados → X e y", "Definimos entradas do modelo e a variável que ele deverá prever."),
        ("4. Mineração", "Árvore de decisão", "O algoritmo aprende divisões para distinguir conforme de retrabalho."),
        ("5. Interpretação", "Métricas → Ouro", "Analisamos acertos, erros, atributos e indicadores para decisão."),
    ]

    for coluna, (titulo_etapa, subtitulo, texto) in zip(cols, etapas):
        with coluna:
            st.markdown(
                f"""
                <div class="kdd-step">
                    <strong>{titulo_etapa}</strong>
                    <p><b>{subtitulo}</b></p>
                    <p class="small-muted">{texto}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.info(
        "Ideia central: o modelo é apenas uma parte do processo. Antes dele existem "
        "seleção, limpeza e transformação; depois dele existe interpretação."
    )

with abas[1]:
    st.subheader("O que mudou do dado bruto para o dado tratado?")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card(
            "Duplicidades no bruto",
            str(duplicatas),
            "Linhas completamente repetidas. Elas foram removidas antes do treinamento.",
            "warn-card" if duplicatas else "ok-card",
        )
    with c2:
        card(
            "Ausências explícitas",
            str(ausencias_bronze),
            "Células vazias já presentes no CSV original.",
            "warn-card" if ausencias_bronze else "ok-card",
        )
    with c3:
        card(
            "Ausências na prata",
            str(ausencias_prata),
            "Depois do tratamento, o conjunto usado no modelo não possui valores ausentes.",
            "ok-card" if ausencias_prata == 0 else "danger-card",
        )
    with c4:
        card(
            "Registros removidos",
            str(len(bronze) - len(prata)),
            "Neste dataset, a redução ocorreu pela retirada das duplicidades completas.",
            "info-card",
        )

    st.divider()
    esq, dire = st.columns(2)

    with esq:
        st.markdown("### 🥉 Bronze — como o dado chegou")
        st.caption("Observe grafias inconsistentes e valores ainda não tratados.")
        st.dataframe(bronze.head(12), use_container_width=True, hide_index=True)

    with dire:
        st.markdown("### 🥈 Prata — depois da limpeza")
        st.caption("Categorias consolidadas, números validados e ausências tratadas.")
        st.dataframe(prata.head(12), use_container_width=True, hide_index=True)

    st.warning(
        "**Ponto importante:** tratar ausente não significa colocar zero. "
        "Neste exercício, valores numéricos ausentes foram imputados pela mediana. "
        "Em um processo real, a estratégia dependeria do contexto."
    )

with abas[2]:
    st.subheader("Como interpretar o resultado do modelo")

    c1, c2, c3 = st.columns(3)
    with c1:
        card(
            "Previsões corretas",
            f"{acertos}/{total_teste}",
            "Classe real e classe prevista coincidiram.",
            "ok-card",
        )
    with c2:
        card(
            "Previsões incorretas",
            str(total_teste - acertos),
            "Precisamos descobrir quais tipos de erro ocorreram.",
            "warn-card",
        )
    with c3:
        card(
            "Atributos usados pela árvore",
            str(len(positivas)),
            f"De {len(importancias)} atributos disponíveis, esta árvore usou efetivamente {len(positivas)}.",
            "info-card",
        )

    st.divider()
    st.markdown("### Matriz de confusão traduzida")
    st.caption("Linhas = situação real | Colunas = previsão do modelo")

    h1, h2 = st.columns(2)
    with h1:
        st.markdown("**Modelo previu: CONFORME**")
    with h2:
        st.markdown("**Modelo previu: RETRABALHO**")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        card(
            "Real: conforme → Previsto: conforme",
            str(tn),
            "✅ Acerto. Era conforme e o modelo reconheceu corretamente.",
            "ok-card",
        )
    with r1c2:
        card(
            "Real: conforme → Previsto: retrabalho",
            str(fp),
            "⚠️ Falso positivo. Uma peça boa seria enviada indevidamente para retrabalho.",
            "warn-card",
        )

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        card(
            "Real: retrabalho → Previsto: conforme",
            str(fn),
            "🚨 Falso negativo. Uma peça que exigia retrabalho seria liberada como conforme.",
            "danger-card",
        )
    with r2c2:
        card(
            "Real: retrabalho → Previsto: retrabalho",
            str(tp),
            "✅ Acerto. O modelo identificou corretamente um caso que precisava de retrabalho.",
            "ok-card",
        )

    st.markdown("### O que mais influenciou a árvore?")
    if len(positivas):
        chart_imp = positivas[["atributo", "importancia_pct"]].sort_values("importancia_pct")
        st.bar_chart(
            chart_imp,
            x="atributo",
            y="importancia_pct",
            horizontal=True,
            x_label="Importância (%)",
            y_label="Atributo",
        )

        top = positivas.sort_values("importancia", ascending=False).reset_index(drop=True)
        cols = st.columns(min(3, len(top)))
        for i, col in enumerate(cols):
            with col:
                row = top.iloc[i]
                card(
                    f"{i+1}º — {row['atributo']}",
                    f"{row['importancia']:.1%}",
                    "Participação relativa deste atributo nas divisões realizadas pela árvore.",
                    "info-card",
                )

    st.info(
        "Importância zero não significa que a variável seja inútil no mundo real. "
        "Significa apenas que esta árvore, neste dataset e neste split, não precisou dela."
    )

with abas[3]:
    st.subheader("Os erros ficam mais fáceis de entender olhando os registros")

    erros = resultados[~resultados["acertou"]].merge(
        prata,
        on="id_lote",
        how="left",
        suffixes=("_teste", ""),
    )

    if erros.empty:
        st.success("Nenhum erro foi encontrado no conjunto de teste.")
    else:
        for _, row in erros.iterrows():
            falso_negativo = row["real"] == "retrabalho" and row["previsto"] == "conforme"

            if falso_negativo:
                st.markdown(
                    f"""
                    <div class="kdd-card danger-card">
                        <h4>🚨 Lote {int(row['id_lote'])} — FALSO NEGATIVO</h4>
                        <p><b>Real:</b> retrabalho → <b>Modelo:</b> conforme</p>
                        <p>O modelo liberaria como conforme um registro que exigia retrabalho.</p>
                        <p><b>O que isso representa:</b> neste cenário, pode ser mais crítico porque
                        um problema pode seguir adiante no processo.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="kdd-card warn-card">
                        <h4>⚠️ Lote {int(row['id_lote'])} — FALSO POSITIVO</h4>
                        <p><b>Real:</b> conforme → <b>Modelo:</b> retrabalho</p>
                        <p>O modelo enviaria para retrabalho um registro que estava conforme.</p>
                        <p><b>O que isso representa:</b> tende a gerar custo ou inspeção desnecessária.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            detalhes = pd.DataFrame(
                {
                    "Variável": [
                        "linha",
                        "turno",
                        "fornecedor",
                        "temp_forno_c",
                        "tempo_ciclo_s",
                        "pressao_bar",
                        "umidade_pct",
                    ],
                    "Valor": [
                        row["linha"],
                        row["turno"],
                        row["fornecedor"],
                        row["temp_forno_c"],
                        row["tempo_ciclo_s"],
                        row["pressao_bar"],
                        row["umidade_pct"],
                    ],
                }
            )
            st.dataframe(detalhes, use_container_width=True, hide_index=True)
            st.divider()

    st.markdown(
        """
        <div class="review-box">
        <b>Vamos pensar:</b><br>
        “Se dois modelos tiverem a mesma acurácia, mas um tiver mais falsos negativos
        e o outro mais falsos positivos, eles são igualmente bons para a indústria?”
        </div>
        """,
        unsafe_allow_html=True,
    )

with abas[4]:
    st.subheader("Transformando resultado técnico em informação para decisão")

    c1, c2, c3 = st.columns(3)
    with c1:
        card(
            "Linha com maior retrabalho",
            str(linha_critica["linha"]).upper(),
            f"Taxa observada no dataset: {linha_critica['taxa_retrabalho']:.1%}.",
            "danger-card",
        )
    with c2:
        card(
            "Lotes na camada ouro",
            str(len(ouro_lotes)),
            "Base refinada que pode alimentar relatórios e aplicações.",
            "info-card",
        )
    with c3:
        card(
            "Mensagem principal",
            "Investigar",
            "O painel aponta onde olhar primeiro; não prova sozinho a causa do retrabalho.",
            "warn-card",
        )

    st.markdown("### Taxa de retrabalho por linha")
    chart_linhas = indicadores[["linha", "taxa_retrabalho_pct"]].sort_values(
        "taxa_retrabalho_pct", ascending=False
    )
    st.bar_chart(
        chart_linhas,
        x="linha",
        y="taxa_retrabalho_pct",
        x_label="Linha",
        y_label="Retrabalho (%)",
    )

    tabela_ind = indicadores[["linha", "taxa_retrabalho"]].copy()
    tabela_ind["taxa_retrabalho"] = tabela_ind["taxa_retrabalho"].map(lambda x: f"{x:.1%}")
    tabela_ind.columns = ["Linha", "Taxa de retrabalho"]
    st.dataframe(tabela_ind, use_container_width=True, hide_index=True)

    st.warning(
        "**Associação não é causalidade.** A Linha C apresenta maior taxa de retrabalho "
        "neste conjunto, mas isso não demonstra que a linha seja a causa. "
        "A próxima ação seria investigar processo, parâmetros, fornecedor e turno."
    )

    st.markdown("### Camada Ouro")
    st.caption("Dados refinados prontos para consumo por uma aplicação ou dashboard.")
    st.dataframe(ouro_lotes.head(30), use_container_width=True, hide_index=True)

with abas[5]:
    st.subheader("Vamos retomar o que aconteceu no projeto")

    blocos = [
        (
            "1. Começamos pelo problema, não pelo algoritmo",
            "“Temos registros de inspeção. Alguns estão duplicados, outros têm valores "
            "ausentes ou escritos de formas diferentes. Antes da IA, precisamos tornar esses dados confiáveis.”",
        ),
        (
            "2. Passamos da Bronze para a Prata",
            "“Bronze é uma fotografia do que chegou. Prata é o dado depois de decisões de limpeza. "
            "Preservar Bronze ajuda a manter rastreabilidade.”",
        ),
        (
            "3. Separamos treino e teste",
            "“O modelo aprende com 72 registros e é avaliado em outros 24. "
            "Não queremos avaliar o modelo apenas nos exemplos que ele já viu.”",
        ),
        (
            "4. Interpretamos a acurácia de 91,67%",
            f"“De {total_teste} registros separados para testar, ele acertou {acertos}. "
            f"Isso corresponde a {acuracia:.1%}. Mas ainda precisamos descobrir COMO ele errou.”",
        ),
        (
            "5. Usamos a matriz para entender os erros",
            f"“Tivemos {fp} falso positivo e {fn} falso negativo. O falso positivo gera custo "
            "desnecessário. O falso negativo pode deixar passar um problema real.”",
        ),
        (
            "6. Finalizamos com a camada Ouro",
            f"“A análise mostra que {linha_critica['linha']} teve {linha_critica['taxa_retrabalho']:.1%} "
            "de retrabalho neste conjunto. Isso não encerra a investigação; mostra onde começar.”",
        ),
    ]

    for titulo_bloco, fala in blocos:
        st.markdown(f"### {titulo_bloco}")
        st.markdown(
            f"""
            <div class="review-box">
                {fala}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Agora vamos conferir se o raciocínio ficou claro")
    st.markdown(
        """
        1. **Por que o número de registros caiu de Bronze para Prata?**
        2. **Por que um valor ausente não deve ser automaticamente substituído por zero?**
        3. **Por que precisamos separar dados de treino e teste?**
        4. **O que a acurácia de 91,67% não consegue nos contar sozinha?**
        5. **Qual erro é mais preocupante neste cenário: falso positivo ou falso negativo? Por quê?**
        """
    )

    st.success(
        "Nosso percurso foi: terminal → Visão geral → Bronze/Prata → "
        "Matriz de confusão → Erros → Decisão de negócio."
    )
