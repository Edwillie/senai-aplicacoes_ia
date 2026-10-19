"""
app_streamlit.py
================
Interface da aula Deep Voice: Central de Voz da Linha de Usinagem.

Rode com o (.venv) ativo, na pasta raiz do projeto:
    python -m streamlit run app_streamlit.py

Toda a IA está em pipeline_voz.py. Este arquivo só monta as telas.
"""

import html
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import pipeline_voz as pv

st.set_page_config(page_title="Central de Voz - Usinagem", layout="wide")
pv.preparar_pastas()


# ---------------------------------------------------------------------------
# Modelos (carregados uma única vez e reaproveitados entre cliques)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando o modelo de reconhecimento de fala...")
def obter_whisper(nome: str):
    return pv.carregar_whisper(nome)


@st.cache_resource(show_spinner="Carregando a voz sintética...")
def obter_voz(caminho: str):
    return pv.carregar_voz(caminho)


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
vozes = pv.listar_vozes()
modelos_whisper = pv.listar_modelos_whisper()

st.sidebar.title("Configuração")
if not vozes or not modelos_whisper:
    st.sidebar.error("Modelos não encontrados.")
    st.error("Os modelos de IA ainda não foram baixados. No terminal, com o (.venv) ativo, rode:")
    st.code("python baixar_modelos.py")
    st.stop()

nome_whisper = st.sidebar.selectbox(
    "Ouvido da central (Whisper)", modelos_whisper,
    index=modelos_whisper.index("base") if "base" in modelos_whisper else 0,
    help="Modelos maiores erram menos, mas demoram mais. 'base' ≈ 74 milhões de "
         "parâmetros; 'small' ≈ 244 milhões.",
)
lista_vozes = list(vozes)
nome_voz = st.sidebar.selectbox(
    "Voz da central (Piper)", lista_vozes,
    index=lista_vozes.index("pt_BR-faber-medium") if "pt_BR-faber-medium" in lista_vozes else 0,
)
usar_vocabulario = st.sidebar.toggle(
    "Usar vocabulário da fábrica", value=True,
    help="Envia ao Whisper um texto inicial com termos técnicos (torno, fresadora...). "
         "Desligue e compare os resultados.",
)
st.sidebar.caption(f"Vozes instaladas: {len(vozes)} | Whisper: {', '.join(modelos_whisper)}")
st.sidebar.divider()
st.sidebar.caption("Toda a IA roda localmente, na CPU deste computador. "
                   "Nenhum áudio é enviado para a internet.")


def voz_atual():
    return obter_voz(str(vozes[nome_voz]))


# ---------------------------------------------------------------------------
# Componente reutilizado: escolher de onde vem o áudio
# ---------------------------------------------------------------------------
def entrada_de_audio(chave: str):
    """Retorna (bytes_do_audio, nome) ou (None, None)."""
    origem = st.radio(
        "De onde vem o áudio?",
        ["Amostra pronta", "Gravar no microfone", "Enviar arquivo"],
        horizontal=True, key=f"origem_{chave}",
    )

    if origem == "Gravar no microfone":
        gravacao = st.audio_input("Clique no microfone, fale e clique de novo para parar",
                                  key=f"mic_{chave}")
        if gravacao is not None:
            return gravacao.getvalue(), "gravação do microfone"

    elif origem == "Enviar arquivo":
        arquivo = st.file_uploader(
            "Envie um áudio (wav, mp3, m4a, ogg, opus, webm, flac)",
            type=["wav", "mp3", "m4a", "ogg", "opus", "webm", "flac"], key=f"arq_{chave}",
        )
        if arquivo is not None:
            st.audio(arquivo)
            return arquivo.getvalue(), arquivo.name

    else:
        amostras = sorted(pv.PASTA_AMOSTRAS.glob("0*.wav"))
        if not amostras:
            st.info("Ainda não há amostras. Elas são criadas pela própria voz sintética.")
            if st.button("Gerar amostras de relatos", key=f"gerar_{chave}"):
                with st.spinner("Gerando 5 relatos com a voz sintética..."):
                    pv.gerar_amostras(voz_atual())
                st.rerun()
            return None, None
        frases = dict(pv.FRASES_AMOSTRA)
        escolhida = st.selectbox("Escolha um relato", amostras,
                                 format_func=lambda p: p.stem, key=f"amostra_{chave}")
        st.caption(f"Texto original: \"{frases.get(escolhida.stem, '')}\"")
        st.audio(str(escolhida))
        return escolhida.read_bytes(), escolhida.stem

    return None, None


def palavras_coloridas(palavras: list[dict]) -> str:
    """Pinta cada palavra conforme a confiança do Whisper."""
    partes = []
    for p in palavras:
        prob = p["probabilidade"]
        cor = "#1b7f3b" if prob >= 0.8 else ("#c77700" if prob >= 0.5 else "#c62828")
        partes.append(
            f"<span title='{prob:.0%}' style='color:{cor};font-weight:600'>"
            f"{html.escape(p['palavra'])}</span>"
        )
    legenda = ("<br><small><span style='color:#1b7f3b'>■ ≥ 80%</span> &nbsp; "
               "<span style='color:#c77700'>■ 50–79%</span> &nbsp; "
               "<span style='color:#c62828'>■ &lt; 50%</span> &nbsp; "
               "(passe o mouse sobre a palavra)</small>")
    return "<div style='font-size:1.15rem;line-height:1.9'>" + " ".join(partes) + legenda + "</div>"


# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
st.title("Central de Voz — Linha de Usinagem")
st.caption("Aplicações de Inteligência Artificial · Aula Deep Voice · "
           "Reconhecimento de fala (Whisper) + Síntese de voz (Piper/VITS)")

aba_central, aba_estudio, aba_ouvido, aba_painel, aba_como = st.tabs([
    "1. Central de Ocorrências", "2. Estúdio de Voz (TTS)",
    "3. Ouvido da IA (ASR)", "4. Painel de Ocorrências", "5. Como funciona",
])

# ===========================================================================
# ABA 1 - CENTRAL DE OCORRÊNCIAS
# ===========================================================================
with aba_central:
    st.markdown(
        "**Cenário:** o operador está com luvas e as mãos ocupadas. Em vez de "
        "parar e digitar, ele **fala** o problema. A central entende, registra "
        "e **responde em voz**, confirmando o que foi acionado."
    )
    col_entrada, col_saida = st.columns([1, 2], gap="large")

    with col_entrada:
        st.markdown("##### Relato do operador")
        audio_bytes, nome_audio = entrada_de_audio("central")
        processar = st.button("Processar ocorrência", type="primary",
                              disabled=audio_bytes is None, width="stretch")

    if processar and audio_bytes is not None:
        try:
            inicio_total = time.perf_counter()
            with st.spinner("Ouvindo..."):
                audio = pv.ler_audio(audio_bytes)
                tr = pv.transcrever(obter_whisper(nome_whisper), audio,
                                    usar_vocabulario=usar_vocabulario)
            dados = pv.interpretar(tr["texto"])
            id_oc = pv.registrar_ocorrencia(audio, tr, dados) if dados["completo"] else None
            resposta = pv.montar_resposta(dados, id_oc)
            with st.spinner("Falando..."):
                fala = pv.sintetizar(voz_atual(), resposta)
            st.session_state["central"] = {
                "tr": tr, "dados": dados, "id": id_oc, "resposta": resposta,
                "fala": pv.audio_para_wav(fala["audio"], fala["taxa"]),
                "tempo_tts": fala["tempo_s"],
                "tempo_total": time.perf_counter() - inicio_total,
            }
        except Exception as e:
            st.session_state.pop("central", None)
            st.error(f"Não foi possível processar o áudio: {e}")

    resultado = st.session_state.get("central")
    with col_saida:
        if not resultado:
            st.info("Escolha um áudio e clique em **Processar ocorrência**.")
        else:
            tr, dados = resultado["tr"], resultado["dados"]
            c1, c2 = st.columns(2)
            with c1.container(border=True):
                st.markdown("**1 · OUVIR** — fala → texto (Whisper)")
                st.write(f"\"{tr['texto'] or '(nenhuma fala reconhecida)'}\"")
                st.caption(f"Confiança média {tr['confianca']:.0%} · {tr['tempo_s']:.1f} s")
            with c2.container(border=True):
                st.markdown("**2 · ENTENDER** — texto → campos (regras)")
                equip = (f"{dados['equipamento']} {dados['numero'] or '?'}"
                         if dados["equipamento"] else "não identificado")
                st.write(f"Equipamento: **{equip}**  \n"
                         f"Problema: **{dados['categoria'] or 'não identificado'}**  \n"
                         f"Prioridade: **{dados['prioridade']}** · Equipe: {dados['equipe']}")
                if dados["palavras_chave"]:
                    st.caption("Palavras-chave: " + ", ".join(dados["palavras_chave"]))
            c3, c4 = st.columns(2)
            with c3.container(border=True):
                st.markdown("**3 · REGISTRAR** — Bronze → Prata → Ouro")
                if resultado["id"]:
                    st.success(f"Ocorrência nº {resultado['id']} gravada na base.")
                else:
                    st.warning("Relato incompleto: nada foi gravado. Faltou: "
                               + ", ".join(dados["faltando"]))
            with c4.container(border=True):
                st.markdown("**4 · RESPONDER** — texto → fala (Piper)")
                st.write(f"\"{resultado['resposta']}\"")
                st.audio(resultado["fala"], format="audio/wav", autoplay=processar)
                st.caption(f"Síntese em {resultado['tempo_tts']:.2f} s")
            st.caption(f"Tempo total do ciclo: {resultado['tempo_total']:.1f} s "
                       f"(modelo Whisper: {nome_whisper}, voz: {nome_voz})")
            with st.expander("Detalhes técnicos da transcrição"):
                st.markdown(palavras_coloridas(tr["palavras"]), unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(tr["segmentos"]), hide_index=True)

# ===========================================================================
# ABA 2 - ESTÚDIO DE VOZ (TTS)
# ===========================================================================
TEXTOS_PRONTOS = {
    "Aviso de segurança": "Atenção. A prensa dois está em manutenção. Utilize protetor "
                          "auricular e óculos de segurança nesta área.",
    "Troca de turno": "Atenção, equipe. A troca de turno será às catorze horas. "
                      "Registrem as ocorrências pendentes antes de sair.",
    "Instrução de operação": "Passo um: confirme o zero peça. Passo dois: feche a porta "
                             "da máquina. Passo três: pressione início de ciclo.",
    "Acessibilidade": "Olá. Este terminal lê em voz alta as instruções de trabalho "
                      "para colaboradores com deficiência visual.",
    "Texto livre": "",
}

with aba_estudio:
    st.markdown("Digite um texto e ouça a IA falar. Mude os controles e compare. "
                "**Pergunta para a turma:** o que muda no som quando cada controle muda?")
    col_a, col_b = st.columns([3, 2], gap="large")
    with col_a:
        modelo_texto = st.selectbox("Texto pronto", list(TEXTOS_PRONTOS))
        texto = st.text_area("Texto para falar", TEXTOS_PRONTOS[modelo_texto], height=110,
                             key=f"texto_{modelo_texto}")
    with col_b:
        voz_estudio = st.selectbox("Voz", lista_vozes, index=lista_vozes.index(nome_voz))
        velocidade = st.slider("Velocidade", 0.5, 2.0, 1.0, 0.1)
        expressividade = st.slider("Expressividade (variação de entonação)", 0.0, 1.0, 0.667, 0.05)
        ritmo = st.slider("Variação de ritmo", 0.0, 1.5, 0.8, 0.05)

    b1, b2 = st.columns(2)
    gerar = b1.button("Gerar voz", type="primary", width="stretch",
                      disabled=not texto.strip())
    comparar = b2.button("Comparar todas as vozes", width="stretch",
                         disabled=not texto.strip())

    if gerar:
        try:
            r = pv.sintetizar(obter_voz(str(vozes[voz_estudio])), texto,
                              velocidade, expressividade, ritmo)
            st.markdown("**Etapa 1 — Texto → fonemas (G2P).** É assim que a IA \"lê\" o texto:")
            st.code(r["fonemas"], language=None)
            st.markdown("**Etapa 2 — Fonemas → áudio (rede neural VITS):**")
            wav = pv.audio_para_wav(r["audio"], r["taxa"])
            st.audio(wav, format="audio/wav", autoplay=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Duração da fala", f"{r['duracao_s']:.1f} s")
            m2.metric("Tempo para gerar", f"{r['tempo_s']:.2f} s")
            m3.metric("RTF", f"{r['rtf']:.2f}", help="Real-Time Factor = tempo para gerar ÷ "
                      "duração. Abaixo de 1 = mais rápido que tempo real.")
            m4.metric("Amostras por segundo", f"{r['taxa']:,}".replace(",", "."))
            st.pyplot(pv.figura_audio(r["audio"], r["taxa"], f"Voz {voz_estudio}"))
            st.download_button("Baixar áudio (.wav)", wav, file_name="voz_sintetica.wav",
                               mime="audio/wav")
        except Exception as e:
            st.error(f"Erro na síntese: {e}")

    if comparar:
        colunas = st.columns(min(len(vozes), 4))
        for i, (nome, caminho) in enumerate(vozes.items()):
            with colunas[i % len(colunas)]:
                try:
                    r = pv.sintetizar(obter_voz(str(caminho)), texto,
                                      velocidade, expressividade, ritmo)
                    st.markdown(f"**{nome}**")
                    st.audio(pv.audio_para_wav(r["audio"], r["taxa"]), format="audio/wav")
                    st.caption(f"{r['taxa']} Hz · gerado em {r['tempo_s']:.2f} s")
                except Exception as e:
                    st.error(f"{nome}: {e}")

    st.info("Uma voz sintética convincente também pode ser usada para enganar pessoas. "
            "Em sistemas reais, informe sempre que a voz é gerada por IA.")

# ===========================================================================
# ABA 3 - OUVIDO DA IA (ASR)
# ===========================================================================
with aba_ouvido:
    st.markdown("Teste o reconhecimento de fala em condições diferentes: **ruído de fábrica**, "
                "**tamanho do modelo** e **vocabulário**. Cada teste entra na tabela de "
                "experimentos no fim da página.")
    col_a, col_b = st.columns([1, 1], gap="large")
    with col_a:
        audio_bytes_lab, nome_lab = entrada_de_audio("ouvido")
    with col_b:
        nivel_ruido = st.select_slider(
            "Ruído de fábrica (relação sinal-ruído)",
            options=["Sem ruído", "30 dB", "20 dB", "10 dB", "5 dB", "0 dB"],
            help="Quanto MENOR o número em dB, MAIS alto é o ruído em relação à voz.",
        )
        modelo_lab = st.selectbox("Modelo Whisper", modelos_whisper,
                                  index=modelos_whisper.index(nome_whisper), key="modelo_lab")
        vocab_lab = st.toggle("Usar vocabulário da fábrica", value=usar_vocabulario, key="vocab_lab")
        idioma_auto = st.toggle("Deixar o modelo detectar o idioma", value=False)
        vad = st.toggle("Filtrar trechos sem voz (VAD)", value=False,
                        help="Remove silêncios antes de transcrever. Ajuda a evitar "
                             "frases inventadas em áudios com muito silêncio.")
        transcrever_btn = st.button("Transcrever", type="primary", width="stretch",
                                    disabled=audio_bytes_lab is None)

    if transcrever_btn and audio_bytes_lab is not None:
        try:
            audio = pv.ler_audio(audio_bytes_lab)
            snr = None if nivel_ruido == "Sem ruído" else float(nivel_ruido.split()[0])
            audio_teste = pv.adicionar_ruido(audio, snr)
            tr = pv.transcrever(obter_whisper(modelo_lab), audio_teste,
                                usar_vocabulario=vocab_lab,
                                idioma=None if idioma_auto else "pt", filtro_vad=vad)

            st.markdown("**Áudio que o modelo recebeu:**")
            st.audio(pv.audio_para_wav(audio_teste, pv.TAXA_WHISPER), format="audio/wav")
            st.markdown("**Transcrição** (cor = confiança de cada palavra):")
            if tr["palavras"]:
                st.markdown(palavras_coloridas(tr["palavras"]), unsafe_allow_html=True)
            else:
                st.warning("Nenhuma fala reconhecida.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Confiança média", f"{tr['confianca']:.0%}")
            m2.metric("Tempo", f"{tr['tempo_s']:.1f} s")
            m3.metric("RTF", f"{tr['rtf']:.2f}")
            m4.metric("Idioma", f"{tr['idioma']} ({tr['prob_idioma']:.0%})")
            st.pyplot(pv.figura_audio(audio_teste, pv.TAXA_WHISPER,
                                      f"O que o Whisper \"vê\" ({nivel_ruido})"))
            with st.expander("Segmentos com tempo de início e fim"):
                st.dataframe(pd.DataFrame(tr["segmentos"]), hide_index=True)

            st.session_state.setdefault("experimentos", []).append({
                "áudio": nome_lab, "ruído": nivel_ruido, "modelo": modelo_lab,
                "vocabulário": "sim" if vocab_lab else "não", "VAD": "sim" if vad else "não",
                "transcrição": tr["texto"], "confiança": f"{tr['confianca']:.0%}",
                "tempo (s)": round(tr["tempo_s"], 1),
            })
        except Exception as e:
            st.error(f"Erro na transcrição: {e}")

    if st.session_state.get("experimentos"):
        st.markdown("##### Tabela de experimentos")
        st.dataframe(pd.DataFrame(st.session_state["experimentos"]), hide_index=True)
        if st.button("Limpar experimentos"):
            st.session_state["experimentos"] = []
            st.rerun()

# ===========================================================================
# ABA 4 - PAINEL DE OCORRÊNCIAS (camada Ouro)
# ===========================================================================
with aba_painel:
    df = pv.listar_ocorrencias()
    if df.empty:
        st.info("Nenhuma ocorrência registrada ainda. Use a aba 1.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Ocorrências", len(df))
        m2.metric("Prioridade alta", int((df["prioridade"] == "Alta").sum()))
        m3.metric("Problema mais frequente", df["categoria"].mode().iat[0])
        st.dataframe(df.drop(columns=["arquivo_audio"]), hide_index=True)
        st.bar_chart(df["categoria"].value_counts(), horizontal=True)

        c1, c2 = st.columns(2)
        with c1:
            escolhido = st.selectbox("Ouvir o áudio original (Bronze) da ocorrência nº",
                                     df["id"].tolist())
            caminho = pv.RAIZ / df.loc[df["id"] == escolhido, "arquivo_audio"].iat[0]
            if Path(caminho).exists():
                st.audio(str(caminho))
        with c2:
            st.download_button("Baixar base (.csv para Excel)",
                               df.to_csv(index=False, sep=";").encode("utf-8-sig"),
                               file_name="ocorrencias.csv", mime="text/csv")
            confirmar = st.checkbox("Confirmo que quero apagar todas as ocorrências")
            if st.button("Apagar ocorrências", disabled=not confirmar):
                pv.limpar_ocorrencias()
                st.rerun()

# ===========================================================================
# ABA 5 - COMO FUNCIONA
# ===========================================================================
with aba_como:
    st.markdown("""
#### O caminho de uma ocorrência dentro do código

| Etapa | O que acontece | Tecnologia | Função em `pipeline_voz.py` |
|---|---|---|---|
| Ouvir | O áudio vira um vetor de números (16 mil por segundo) e o Whisper gera o texto | Whisper (Transformer encoder-decoder) via faster-whisper | `ler_audio()`, `transcrever()` |
| Entender | Regras procuram equipamento, número, tipo de problema e urgência | Python + expressões regulares | `interpretar()` |
| Registrar | Bronze: `.wav` bruto · Prata: `.json` com a transcrição · Ouro: linha no SQLite | sqlite3, json | `registrar_ocorrencia()` |
| Responder | Texto → fonemas → áudio | Piper (rede VITS em ONNX) | `montar_resposta()`, `sintetizar()` |

#### Onde está o "Deep Voice" aqui?
O Deep Voice (Baidu, 2017) mostrou que **todas** as etapas de um sintetizador de voz podiam ser
redes neurais aprendidas a partir de dados. O Piper usa um descendente direto dessa ideia (VITS):
o texto vira fonemas e uma única rede neural gera o áudio. O Whisper faz o caminho inverso.

#### Cuidados
- Voz é **dado biométrico**: pela LGPD é dado pessoal sensível. Gravar a voz de alguém exige finalidade clara e consentimento.
- Vozes sintéticas podem ser usadas em golpes. Sistemas sérios avisam quando a voz é artificial.
- A etapa "Entender" usa regras simples. Ela erra quando o operador fala de um jeito não previsto — teste isso!
""")
