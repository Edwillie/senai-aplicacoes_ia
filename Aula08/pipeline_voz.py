"""
pipeline_voz.py
===============
Aula: Deep Voice - IA aplicada a voz (Aplicações de Inteligência Artificial)

Este arquivo concentra TODA a lógica de IA da aula. O app (app_streamlit.py)
apenas chama as funções daqui.

Fluxo da "Central de Ocorrências por Voz" da linha de usinagem:

    OUVIR      -> o operador fala; o Whisper converte fala em texto (ASR)
    ENTENDER   -> regras simples extraem máquina, tipo de falha e prioridade
    REGISTRAR  -> Bronze (áudio bruto) -> Prata (transcrição JSON) -> Ouro (SQLite)
    RESPONDER  -> o Piper converte o texto de resposta em fala (TTS)

Para testar só pelo terminal (sem interface):
    python pipeline_voz.py
"""

from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import time
import unicodedata
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

# Evita avisos desnecessários do Hugging Face no Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ---------------------------------------------------------------------------
# 0. PASTAS DO PROJETO
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent

PASTA_MODELOS = RAIZ / "modelos"
PASTA_WHISPER = PASTA_MODELOS / "whisper"   # modelos de reconhecimento de fala
PASTA_VOZES = PASTA_MODELOS / "vozes"       # modelos de síntese de voz (Piper)

PASTA_BRONZE = RAIZ / "data" / "bronze" / "audios"        # áudio bruto
PASTA_PRATA = RAIZ / "data" / "prata" / "transcricoes"    # texto + metadados
PASTA_OURO = RAIZ / "data" / "ouro"                       # base final
BANCO = PASTA_OURO / "ocorrencias.db"

PASTA_AMOSTRAS = RAIZ / "amostras"   # áudios de exemplo para testar sem microfone

TAXA_WHISPER = 16000   # o Whisper trabalha com áudio a 16 mil amostras por segundo


def preparar_pastas() -> None:
    """Cria todas as pastas do projeto (não dá erro se já existirem)."""
    for pasta in (PASTA_WHISPER, PASTA_VOZES, PASTA_BRONZE, PASTA_PRATA,
                  PASTA_OURO, PASTA_AMOSTRAS):
        pasta.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. MODELOS DISPONÍVEIS NO COMPUTADOR
# ---------------------------------------------------------------------------
def listar_vozes() -> dict[str, Path]:
    """Retorna {nome_da_voz: caminho_do_.onnx} das vozes baixadas."""
    vozes = {}
    if PASTA_VOZES.exists():
        for onnx in sorted(PASTA_VOZES.glob("*.onnx")):
            config = onnx.with_suffix(".onnx.json")
            if config.exists() and onnx.stat().st_size > 0:
                vozes[onnx.stem] = onnx
    return vozes


def listar_modelos_whisper() -> list[str]:
    """Retorna os tamanhos de Whisper baixados (ex.: ['base', 'small'])."""
    modelos = []
    if PASTA_WHISPER.exists():
        for pasta in sorted(PASTA_WHISPER.iterdir()):
            if (pasta / "model.bin").exists():
                modelos.append(pasta.name)
    return modelos


# ---------------------------------------------------------------------------
# 2. FALAR - síntese de voz (Text-to-Speech) com Piper
# ---------------------------------------------------------------------------
def carregar_voz(caminho_onnx: str | Path):
    """Carrega uma voz do Piper (rede neural VITS exportada em ONNX)."""
    from piper import PiperVoice
    return PiperVoice.load(str(caminho_onnx))


def sintetizar(voz, texto: str, velocidade: float = 1.0,
               expressividade: float = 0.667, variacao_ritmo: float = 0.8) -> dict:
    """
    Converte texto em fala.

    velocidade     -> 1.0 normal | 1.5 mais rápido | 0.7 mais lento
    expressividade -> variação de entonação (noise_scale do VITS)
    variacao_ritmo -> variação na duração dos fonemas (noise_w do VITS)
    """
    from piper import SynthesisConfig

    config = SynthesisConfig(
        length_scale=1.0 / max(velocidade, 0.1),  # no Piper, maior = mais lento
        noise_scale=expressividade,
        noise_w_scale=variacao_ritmo,
        normalize_audio=True,
    )

    inicio = time.perf_counter()
    pedacos = list(voz.synthesize(texto, syn_config=config))
    tempo = time.perf_counter() - inicio

    if not pedacos:
        raise ValueError("Nenhum áudio foi gerado. O texto está vazio?")

    taxa = pedacos[0].sample_rate
    audio = np.concatenate([p.audio_float_array for p in pedacos]).astype(np.float32)
    fonemas = " | ".join(
        unicodedata.normalize("NFC", "".join(p.phonemes)) for p in pedacos
    )
    duracao = len(audio) / taxa

    return {
        "audio": audio,
        "taxa": taxa,
        "fonemas": fonemas,
        "tempo_s": tempo,
        "duracao_s": duracao,
        # RTF < 1 significa que a IA gera a fala mais rápido do que o tempo da fala
        "rtf": tempo / duracao if duracao else 0.0,
    }


def audio_para_wav(audio: np.ndarray, taxa: int) -> bytes:
    """Converte um vetor de áudio (-1 a 1) em bytes de um arquivo .wav."""
    audio16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as arquivo:
        arquivo.setnchannels(1)
        arquivo.setsampwidth(2)
        arquivo.setframerate(int(taxa))
        arquivo.writeframes(audio16.tobytes())
    return buffer.getvalue()


def salvar_wav(audio: np.ndarray, taxa: int, caminho: str | Path) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_bytes(audio_para_wav(audio, taxa))
    return caminho


# ---------------------------------------------------------------------------
# 3. OUVIR - reconhecimento de fala (Speech-to-Text) com Whisper
# ---------------------------------------------------------------------------
VOCABULARIO_FABRICA = (
    "Relato de ocorrência na linha de usinagem: máquina, torno CNC, fresadora, "
    "prensa, superaquecimento, vibração, vazamento de óleo, ferramenta quebrada."
)


def carregar_whisper(nome: str = "base"):
    """Carrega um modelo Whisper já baixado (modelos/whisper/<nome>)."""
    from faster_whisper import WhisperModel
    pasta = PASTA_WHISPER / nome
    if not (pasta / "model.bin").exists():
        raise FileNotFoundError(
            f"Modelo Whisper '{nome}' não encontrado em {pasta}. "
            "Rode: python baixar_modelos.py"
        )
    # int8 = pesos compactados: mais rápido e mais leve em CPU
    return WhisperModel(str(pasta), device="cpu", compute_type="int8")


def ler_audio(fonte) -> np.ndarray:
    """
    Lê qualquer áudio (wav, mp3, m4a, ogg do WhatsApp...) e devolve
    um vetor numpy mono, 16 kHz, valores entre -1 e 1.
    'fonte' pode ser um caminho de arquivo ou bytes/BytesIO.
    """
    from faster_whisper import decode_audio
    if isinstance(fonte, (bytes, bytearray)):
        fonte = io.BytesIO(fonte)
    if hasattr(fonte, "seek"):
        fonte.seek(0)
    if isinstance(fonte, Path):
        fonte = str(fonte)
    return decode_audio(fonte, sampling_rate=TAXA_WHISPER).astype(np.float32)


def adicionar_ruido(audio: np.ndarray, snr_db: float | None,
                    taxa: int = TAXA_WHISPER, semente: int = 7) -> np.ndarray:
    """
    Simula ruído de chão de fábrica: chiado (ruído branco) + zumbido
    elétrico de 60 Hz e harmônicos (rede elétrica brasileira).

    snr_db = relação sinal-ruído em decibéis.
       30 dB -> ruído quase imperceptível
       10 dB -> ruído forte
        0 dB -> ruído com a mesma energia da voz
    """
    if snr_db is None or len(audio) == 0:
        return audio
    gerador = np.random.default_rng(semente)
    t = np.arange(len(audio)) / taxa
    zumbido = sum(np.sin(2 * np.pi * f * t) / (i + 1)
                  for i, f in enumerate((60, 120, 180, 240)))
    ruido = gerador.normal(0, 1, len(audio)) + 0.8 * zumbido

    potencia_voz = np.mean(audio ** 2) + 1e-12
    potencia_ruido = np.mean(ruido ** 2) + 1e-12
    fator = np.sqrt(potencia_voz / (potencia_ruido * 10 ** (snr_db / 10)))
    misturado = audio + fator * ruido
    pico = np.max(np.abs(misturado)) + 1e-9
    return (misturado / max(pico, 1.0)).astype(np.float32)


def transcrever(modelo, audio: np.ndarray, usar_vocabulario: bool = True,
                idioma: str | None = "pt", filtro_vad: bool = False) -> dict:
    """
    Converte fala em texto.

    usar_vocabulario -> envia um "prompt inicial" com termos da fábrica,
                        o que ajuda o modelo a acertar palavras técnicas.
    idioma           -> "pt" fixo, ou None para o modelo detectar sozinho.
    filtro_vad       -> remove trechos sem voz antes de transcrever.
    """
    inicio = time.perf_counter()
    segmentos_gen, info = modelo.transcribe(
        audio,
        language=idioma,
        beam_size=5,
        initial_prompt=VOCABULARIO_FABRICA if usar_vocabulario else None,
        word_timestamps=True,
        vad_filter=filtro_vad,
        condition_on_previous_text=False,
    )
    segmentos = list(segmentos_gen)   # a transcrição acontece de fato aqui
    tempo = time.perf_counter() - inicio

    palavras = []
    for seg in segmentos:
        for p in (seg.words or []):
            palavras.append({
                "palavra": p.word.strip(),
                "inicio": round(p.start, 2),
                "fim": round(p.end, 2),
                "probabilidade": round(float(p.probability), 3),
            })

    texto = " ".join(s.text.strip() for s in segmentos).strip()
    confianca = float(np.mean([p["probabilidade"] for p in palavras])) if palavras else 0.0
    duracao = len(audio) / TAXA_WHISPER

    return {
        "texto": texto,
        "idioma": info.language,
        "prob_idioma": round(float(info.language_probability), 3),
        "segmentos": [{"inicio": round(s.start, 2), "fim": round(s.end, 2),
                       "texto": s.text.strip()} for s in segmentos],
        "palavras": palavras,
        "confianca": round(confianca, 3),
        "tempo_s": tempo,
        "duracao_s": duracao,
        "rtf": tempo / duracao if duracao else 0.0,
    }


# ---------------------------------------------------------------------------
# 4. ENTENDER - extrair informações do texto com regras simples
# ---------------------------------------------------------------------------
# Em um sistema maior, esta etapa poderia usar um modelo de linguagem (LLM).
# Aqui usamos regras para deixar o funcionamento 100% visível.

NUMEROS_POR_EXTENSO = {
    "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4,
    "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10,
    "onze": 11, "doze": 12, "treze": 13, "catorze": 14, "quatorze": 14,
    "quinze": 15, "vinte": 20,
}

EQUIPAMENTOS = ["centro de usinagem", "maquina", "torno", "fresadora", "prensa",
                "esteira", "robo", "compressor", "injetora", "furadeira"]

# A ordem importa: a primeira categoria encontrada é a escolhida.
CATEGORIAS = [
    {"nome": "Segurança", "prioridade": "Alta",
     "equipe": "Segurança do Trabalho e Brigada",
     "termos": ["fumaca", "fogo", "incendio", "faisca", "choque", "acidente",
                "ferido", "machucado", "cheiro de queimado"]},
    {"nome": "Parada de máquina", "prioridade": "Alta",
     "equipe": "Manutenção elétrica",
     "termos": ["parou", "parada", "travou", "travada", "nao liga", "desligou",
                "desarmou", "nao funciona"]},
    {"nome": "Superaquecimento", "prioridade": "Média",
     "equipe": "Manutenção mecânica",
     "termos": ["superaquec", "esquent", "aquecend", "quente", "temperatura"]},
    {"nome": "Vazamento", "prioridade": "Média",
     "equipe": "Manutenção mecânica",
     "termos": ["vazamento", "vazando", "oleo", "fluido", "refrigerante"]},
    {"nome": "Vibração ou ruído", "prioridade": "Média",
     "equipe": "Manutenção mecânica",
     "termos": ["vibra", "barulho", "ruido", "estalo", "batendo", "rangendo"]},
    {"nome": "Ferramenta", "prioridade": "Baixa",
     "equipe": "Ferramentaria",
     "termos": ["ferramenta", "broca", "fresa ", "pastilha", "quebrou", "desgast"]},
]

TERMOS_URGENCIA = ["urgente", "urgencia", "emergencia", "imediato", "agora"]


def normalizar(texto: str) -> str:
    """minúsculas e sem acentos: 'Máquina SETE' -> 'maquina sete'"""
    sem_acento = unicodedata.normalize("NFKD", texto)
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return " " + re.sub(r"[^\w\s]", " ", sem_acento.lower()) + " "


def interpretar(texto: str) -> dict:
    """Extrai equipamento, número, categoria, prioridade e equipe do texto."""
    t = normalizar(texto)

    # Equipamento + número (ex.: "maquina 7", "torno numero tres")
    equipamento, numero = None, None
    padrao_num = r"(\d+|" + "|".join(NUMEROS_POR_EXTENSO) + r")"
    for equip in EQUIPAMENTOS:
        achou = re.search(rf"{equip}\s+(?:numero\s+|n\s+|no\s+)?{padrao_num}\b", t)
        if achou:
            equipamento = equip
            valor = achou.group(1)
            numero = int(valor) if valor.isdigit() else NUMEROS_POR_EXTENSO[valor]
            break
    if equipamento is None:
        for equip in EQUIPAMENTOS:
            if f" {equip} " in t:
                equipamento = equip
                break

    # Categoria da ocorrência
    categoria, termos_achados = None, []
    for cat in CATEGORIAS:
        achados = [termo.strip() for termo in cat["termos"] if termo in t]
        if achados:
            categoria = cat
            termos_achados = achados
            break

    prioridade = categoria["prioridade"] if categoria else "Baixa"
    urgente = [u for u in TERMOS_URGENCIA if f" {u} " in t]
    if urgente:
        prioridade = "Alta"
        termos_achados += urgente

    faltando = []
    if numero is None:
        faltando.append("número do equipamento")
    if categoria is None:
        faltando.append("tipo de problema")

    nomes_bonitos = {"maquina": "máquina", "robo": "robô"}
    return {
        "equipamento": nomes_bonitos.get(equipamento, equipamento) if equipamento else None,
        "numero": numero,
        "categoria": categoria["nome"] if categoria else None,
        "prioridade": prioridade,
        "equipe": categoria["equipe"] if categoria else "Supervisão do turno",
        "palavras_chave": termos_achados,
        "completo": not faltando,
        "faltando": faltando,
    }


def montar_resposta(dados: dict, id_ocorrencia: int | None = None) -> str:
    """Monta a frase que a central vai FALAR de volta para o operador."""
    if not dados["completo"]:
        falta = " e ".join(dados["faltando"])
        return (f"Não consegui registrar. Faltou informar o {falta}. "
                "Por favor, repita o relato.")
    equip = f"{dados['equipamento']} {dados['numero']}"
    texto = (f"Ocorrência {id_ocorrencia} registrada. {equip.capitalize()}, "
             f"{dados['categoria'].lower()}, prioridade {dados['prioridade'].lower()}. "
             f"Equipe acionada: {dados['equipe'].lower()}.")
    if dados["categoria"] == "Segurança":
        texto += " Afaste-se do equipamento e aguarde a brigada."
    return texto


# ---------------------------------------------------------------------------
# 5. REGISTRAR - Bronze -> Prata -> Ouro
# ---------------------------------------------------------------------------
def iniciar_banco() -> None:
    preparar_pastas()
    with sqlite3.connect(BANCO) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS ocorrencias (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_hora TEXT,
                equipamento TEXT,
                numero INTEGER,
                categoria TEXT,
                prioridade TEXT,
                equipe TEXT,
                transcricao TEXT,
                confianca REAL,
                arquivo_audio TEXT
            )""")


def registrar_ocorrencia(audio: np.ndarray, transcricao: dict, dados: dict) -> int:
    """Salva o áudio (Bronze), a transcrição (Prata) e a ocorrência (Ouro)."""
    iniciar_banco()
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    # BRONZE: áudio exatamente como chegou (já em 16 kHz)
    caminho_audio = salvar_wav(audio, TAXA_WHISPER, PASTA_BRONZE / f"ocorrencia_{carimbo}.wav")

    # PRATA: texto + metadados da transcrição
    (PASTA_PRATA / f"ocorrencia_{carimbo}.json").write_text(
        json.dumps({"transcricao": transcricao, "interpretacao": dados},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # OURO: uma linha limpa na base de ocorrências
    with sqlite3.connect(BANCO) as con:
        cursor = con.execute(
            """INSERT INTO ocorrencias (data_hora, equipamento, numero, categoria,
               prioridade, equipe, transcricao, confianca, arquivo_audio)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().strftime("%d/%m/%Y %H:%M:%S"), dados["equipamento"],
             dados["numero"], dados["categoria"], dados["prioridade"], dados["equipe"],
             transcricao["texto"], transcricao["confianca"],
             caminho_audio.relative_to(RAIZ).as_posix()),
        )
        return int(cursor.lastrowid)


def listar_ocorrencias():
    import pandas as pd
    iniciar_banco()
    with sqlite3.connect(BANCO) as con:
        return pd.read_sql_query("SELECT * FROM ocorrencias ORDER BY id DESC", con)


def limpar_ocorrencias() -> None:
    iniciar_banco()
    with sqlite3.connect(BANCO) as con:
        con.execute("DELETE FROM ocorrencias")
        con.execute("DELETE FROM sqlite_sequence WHERE name='ocorrencias'")


# ---------------------------------------------------------------------------
# 6. VISUALIZAR - forma de onda e espectrograma mel
# ---------------------------------------------------------------------------
def _hz_para_mel(f):
    return 2595 * np.log10(1 + np.asarray(f) / 700)


def _mel_para_hz(m):
    return 700 * (10 ** (np.asarray(m) / 2595) - 1)


def espectrograma_mel(audio: np.ndarray, taxa: int, n_mels: int = 80):
    """
    Calcula o espectrograma mel: a "imagem" do som que modelos como
    Tacotron 2, Deep Voice 3 e Whisper usam como representação intermediária.
    """
    n_fft = 1024 if taxa > 16000 else 512
    salto = n_fft // 4
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    janela = np.hanning(n_fft)
    quadros = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::salto] * janela
    espectro = np.abs(np.fft.rfft(quadros, axis=1)) ** 2          # (tempo, freq)

    freqs = np.linspace(0, taxa / 2, espectro.shape[1])
    mels = np.linspace(_hz_para_mel(0), _hz_para_mel(taxa / 2), n_mels + 2)
    bordas = _mel_para_hz(mels)
    filtros = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        esq, centro, dir_ = bordas[i], bordas[i + 1], bordas[i + 2]
        subida = (freqs - esq) / (centro - esq + 1e-9)
        descida = (dir_ - freqs) / (dir_ - centro + 1e-9)
        filtros[i] = np.maximum(0, np.minimum(subida, descida))

    mel = espectro @ filtros.T                                      # (tempo, n_mels)
    mel_db = 10 * np.log10(mel + 1e-10)
    mel_db = np.maximum(mel_db, mel_db.max() - 80)
    duracao = len(audio) / taxa
    return mel_db.T, duracao


def figura_audio(audio: np.ndarray, taxa: int, titulo: str = ""):
    """Figura com a forma de onda (em cima) e o espectrograma mel (embaixo)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mel_db, duracao = espectrograma_mel(audio, taxa)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.6]})
    t = np.arange(len(audio)) / taxa
    ax1.plot(t, audio, linewidth=0.5, color="#1f4e79")
    ax1.set_ylabel("Amplitude")
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_title(titulo or "Forma de onda e espectrograma mel", fontsize=11)
    ax2.imshow(mel_db, origin="lower", aspect="auto", cmap="magma",
               extent=[0, duracao, 0, mel_db.shape[0]])
    ax2.set_ylabel("Faixas mel (grave → agudo)")
    ax2.set_xlabel("Tempo (s)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. AMOSTRAS - relatos de exemplo gerados pela própria IA
# ---------------------------------------------------------------------------
FRASES_AMOSTRA = [
    ("01_superaquecimento",
     "Aqui é o operador do turno da manhã. A máquina sete está superaquecendo, "
     "o painel mostra temperatura alta."),
    ("02_vazamento",
     "Torno três com vazamento de óleo embaixo do barramento. Preciso de manutenção."),
    ("03_seguranca",
     "Urgente! Está saindo fumaça da prensa dois. Já isolamos a área."),
    ("04_ferramenta",
     "A broca da furadeira cinco quebrou durante a operação."),
    ("05_incompleto",
     "Tem um barulho estranho vindo da máquina perto da janela."),
]


def gerar_amostras(voz, pasta: Path = PASTA_AMOSTRAS) -> list[Path]:
    """Usa o TTS para criar áudios de relatos (útil se não houver microfone)."""
    pasta.mkdir(parents=True, exist_ok=True)
    arquivos = []
    for nome, frase in FRASES_AMOSTRA:
        r = sintetizar(voz, frase)
        arquivos.append(salvar_wav(r["audio"], r["taxa"], pasta / f"{nome}.wav"))
    return arquivos


# ---------------------------------------------------------------------------
# 8. DEMONSTRAÇÃO NO TERMINAL
# ---------------------------------------------------------------------------
def demonstracao() -> None:
    preparar_pastas()
    vozes = listar_vozes()
    modelos = listar_modelos_whisper()
    if not vozes or not modelos:
        print("Modelos não encontrados. Rode antes: python baixar_modelos.py")
        return

    nome_voz = "pt_BR-faber-medium" if "pt_BR-faber-medium" in vozes else next(iter(vozes))
    nome_whisper = "base" if "base" in modelos else modelos[0]

    print(f"\n[1/5] Carregando voz '{nome_voz}' e Whisper '{nome_whisper}'...")
    voz = carregar_voz(vozes[nome_voz])
    whisper = carregar_whisper(nome_whisper)

    frase = FRASES_AMOSTRA[0][1]
    print(f"\n[2/5] Simulando o operador (TTS gera a fala):\n      \"{frase}\"")
    fala = sintetizar(voz, frase)
    print(f"      fonemas: {fala['fonemas'][:90]}...")
    print(f"      {fala['duracao_s']:.1f} s de áudio gerados em {fala['tempo_s']:.2f} s "
          f"(RTF = {fala['rtf']:.2f})")
    salvar_wav(fala["audio"], fala["taxa"], PASTA_AMOSTRAS / "demo_terminal.wav")

    print("\n[3/5] OUVIR - Whisper transcreve o áudio:")
    audio = ler_audio(PASTA_AMOSTRAS / "demo_terminal.wav")
    tr = transcrever(whisper, audio)
    print(f"      \"{tr['texto']}\"")
    print(f"      confiança média: {tr['confianca']:.0%} | tempo: {tr['tempo_s']:.2f} s")

    print("\n[4/5] ENTENDER - regras extraem os campos:")
    dados = interpretar(tr["texto"])
    for chave in ("equipamento", "numero", "categoria", "prioridade", "equipe"):
        print(f"      {chave:<12}: {dados[chave]}")

    id_oc = registrar_ocorrencia(audio, tr, dados) if dados["completo"] else None
    resposta = montar_resposta(dados, id_oc)
    print(f"\n[5/5] REGISTRAR + RESPONDER:\n      \"{resposta}\"")
    r = sintetizar(voz, resposta)
    destino = salvar_wav(r["audio"], r["taxa"], PASTA_AMOSTRAS / "demo_resposta.wav")
    print(f"      resposta falada salva em: {destino.relative_to(RAIZ)}")
    print("\nPronto. Abra a interface com:  python -m streamlit run app_streamlit.py\n")


if __name__ == "__main__":
    demonstracao()
