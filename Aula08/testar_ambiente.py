"""
testar_ambiente.py
==================
Confere, em ordem, tudo o que a aula precisa. Rode com o (.venv) ativo:

    python testar_ambiente.py

Cada linha mostra OK ou ERRO com a orientação do que fazer.
"""

import importlib
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
erros = 0


def ok(msg):
    print(f"  [OK]   {msg}")


def erro(msg, dica):
    global erros
    erros += 1
    print(f"  [ERRO] {msg}\n         -> {dica}")


print("\n1) Python e ambiente virtual")
versao = sys.version_info
if (versao.major, versao.minor) == (3, 12):
    ok(f"Python {versao.major}.{versao.minor}.{versao.micro}")
else:
    erro(f"Python {versao.major}.{versao.minor} em uso (precisa ser 3.12)",
         "Apague a pasta .venv e recrie com:  py -3.12 -m venv .venv")

if sys.prefix != sys.base_prefix:
    ok(f"ambiente virtual ativo: {Path(sys.prefix).name}")
    if Path(sys.prefix).resolve().parent != RAIZ:
        erro("o .venv ativo NÃO está na pasta raiz deste projeto",
             "Feche o terminal, entre na pasta aula_deepvoice e ative o .venv dela.")
else:
    erro("ambiente virtual NÃO está ativo",
         "Ative com:  .venv\\Scripts\\activate   (deve aparecer (.venv) no terminal)")

if str(RAIZ).isascii() and " " not in str(RAIZ):
    ok(f"pasta do projeto: {RAIZ}")
else:
    erro(f"o caminho da pasta tem acento ou espaço: {RAIZ}",
         r"Mova a pasta para C:\aulas\aula_deepvoice (alguns modelos falham com acentos).")

print("\n2) Bibliotecas")
for pacote in ("numpy", "pandas", "matplotlib", "streamlit", "onnxruntime",
               "piper", "ctranslate2", "faster_whisper", "av", "huggingface_hub"):
    try:
        modulo = importlib.import_module(pacote)
        ok(f"{pacote:<16} {getattr(modulo, '__version__', '')}")
    except Exception as e:
        erro(f"{pacote} não importou ({e.__class__.__name__}: {e})",
             "Rode:  python -m pip install -r requirements.txt")

if erros:
    print(f"\n{erros} problema(s) encontrado(s). Corrija antes de continuar.\n")
    sys.exit(1)

print("\n3) Modelos baixados")
import pipeline_voz as pv  # noqa: E402

pv.preparar_pastas()
vozes = pv.listar_vozes()
modelos = pv.listar_modelos_whisper()
if vozes:
    ok("vozes: " + ", ".join(vozes))
else:
    erro("nenhuma voz encontrada em modelos/vozes", "Rode:  python baixar_modelos.py")
if modelos:
    ok("whisper: " + ", ".join(modelos))
else:
    erro("nenhum Whisper encontrado em modelos/whisper", "Rode:  python baixar_modelos.py")

if erros:
    print(f"\n{erros} problema(s) encontrado(s). Corrija antes de continuar.\n")
    sys.exit(1)

print("\n4) Teste funcional (fala -> texto)")
try:
    nome_voz = "pt_BR-faber-medium" if "pt_BR-faber-medium" in vozes else next(iter(vozes))
    voz = pv.carregar_voz(vozes[nome_voz])
    r = pv.sintetizar(voz, "Teste da central de voz. Máquina sete com vibração.")
    ok(f"TTS gerou {r['duracao_s']:.1f} s de áudio em {r['tempo_s']:.2f} s")
    caminho = pv.salvar_wav(r["audio"], r["taxa"], pv.PASTA_AMOSTRAS / "_teste.wav")
except Exception as e:
    erro(f"falha no TTS: {e}", "Confira se os arquivos .onnx e .onnx.json da voz estão completos.")
    sys.exit(1)

try:
    inicio = time.perf_counter()
    whisper = pv.carregar_whisper(modelos[0])
    ok(f"Whisper '{modelos[0]}' carregado em {time.perf_counter() - inicio:.1f} s")
    tr = pv.transcrever(whisper, pv.ler_audio(caminho))
    ok(f"transcrição: \"{tr['texto']}\" ({tr['tempo_s']:.1f} s)")
    caminho.unlink(missing_ok=True)
except Exception as e:
    erro(f"falha no Whisper: {e}", "Rode de novo:  python baixar_modelos.py")
    sys.exit(1)

print("\nAmbiente OK. Abra o app com:  python -m streamlit run app_streamlit.py\n")
