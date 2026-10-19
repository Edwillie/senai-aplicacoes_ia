"""
baixar_modelos.py
=================
Baixa os modelos de IA usados na aula para a pasta  modelos/  do projeto.
Precisa de internet SOMENTE nesta etapa. Depois, tudo roda offline.

Uso:
    python baixar_modelos.py            -> Whisper "base" + vozes em português


"""

import os
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

RAIZ = Path(__file__).resolve().parent
PASTA_WHISPER = RAIZ / "modelos" / "whisper"
PASTA_VOZES = RAIZ / "modelos" / "vozes"

URL_VOZ = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
           "pt/pt_BR/{nome}/{qualidade}/pt_BR-{nome}-{qualidade}{ext}?download=true")

# (nome, qualidade, obrigatória?)
VOZES = [
    ("faber", "medium", True),      # voz masculina - usada como padrão
    ("edresson", "low", False),     # voz masculina, qualidade menor e mais rápida
    ("cadu", "medium", False),      # voz masculina
    ("jeff", "medium", False),      # voz masculina
]

ARQUIVOS_WHISPER = ["config.json", "model.bin", "tokenizer.json", "vocabulary.*",
                    "preprocessor_config.json"]


def baixar_arquivo(url: str, destino: Path) -> None:
    if destino.exists() and destino.stat().st_size > 0:
        print(f"   já existe: {destino.name}")
        return
    temporario = destino.with_suffix(destino.suffix + ".parcial")
    with urllib.request.urlopen(url, timeout=60) as resposta:
        total = int(resposta.headers.get("Content-Length", 0))
        baixado = 0
        with open(temporario, "wb") as arquivo:
            while True:
                bloco = resposta.read(1024 * 256)
                if not bloco:
                    break
                arquivo.write(bloco)
                baixado += len(bloco)
                if total:
                    print(f"\r   {destino.name}: {baixado / total:5.0%} "
                          f"de {total / 1e6:.0f} MB", end="", flush=True)
    print()
    temporario.replace(destino)


def baixar_voz(nome: str, qualidade: str) -> bool:
    PASTA_VOZES.mkdir(parents=True, exist_ok=True)
    try:
        for ext in (".onnx.json", ".onnx"):
            url = URL_VOZ.format(nome=nome, qualidade=qualidade, ext=ext)
            baixar_arquivo(url, PASTA_VOZES / f"pt_BR-{nome}-{qualidade}{ext}")
        return True
    except Exception as erro:  # voz opcional pode não existir/estar indisponível
        print(f"\n   não foi possível baixar pt_BR-{nome}-{qualidade}: {erro}")
        # remove restos do download para não deixar uma voz "pela metade"
        for arquivo in PASTA_VOZES.glob(f"pt_BR-{nome}-{qualidade}.onnx*"):
            arquivo.unlink()
        return False


def baixar_whisper(tamanho: str) -> bool:
    from huggingface_hub import snapshot_download
    destino = PASTA_WHISPER / tamanho
    if (destino / "model.bin").exists():
        print(f"   já existe: whisper/{tamanho}")
        return True
    print(f"   baixando Systran/faster-whisper-{tamanho} (aguarde, sem barra de progresso)...")
    inicio = time.time()
    try:
        snapshot_download(repo_id=f"Systran/faster-whisper-{tamanho}",
                          local_dir=str(destino), allow_patterns=ARQUIVOS_WHISPER)
    except Exception as erro:
        print(f"   ERRO ao baixar o Whisper '{tamanho}': {erro}")
        return False
    print(f"   ok: whisper/{tamanho} ({time.time() - inicio:.0f} s)")
    return True


def main() -> None:
    print("=" * 60)
    print("Download dos modelos da aula Deep Voice")
    print("=" * 60)

    print("\n1) Vozes do Piper (síntese de voz, ~60 MB cada)")
    resultado_vozes = {}
    for nome, qualidade, obrigatoria in VOZES:
        resultado_vozes[f"{nome}-{qualidade}"] = (baixar_voz(nome, qualidade), obrigatoria)

    print("\n2) Whisper (reconhecimento de fala)")
    tamanhos = ["base"] + (["small"] if "--small" in sys.argv else [])
    resultado_whisper = {t: baixar_whisper(t) for t in tamanhos}

    print("\n" + "=" * 60)
    problemas = [n for n, (ok, obrig) in resultado_vozes.items() if obrig and not ok]
    problemas += [f"whisper-{t}" for t, ok in resultado_whisper.items() if not ok]
    for nome, (ok, obrig) in resultado_vozes.items():
        print(f"   voz {nome:<16} {'OK' if ok else ('FALHOU' if obrig else 'não baixada (opcional)')}")
    for t, ok in resultado_whisper.items():
        print(f"   whisper {t:<12} {'OK' if ok else 'FALHOU'}")
    if problemas:
        print("\nATENÇÃO: faltam modelos obrigatórios. Verifique a internet e rode de novo.")
        sys.exit(1)
    print("\nTudo pronto. Próximo passo:  python testar_ambiente.py")


if __name__ == "__main__":
    main()
