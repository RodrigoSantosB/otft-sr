"""Gera ``dist/otft-sr-<commit>.zip`` para o notebook 02 no Colab.

Uso: ``<venv>/bin/python scripts/empacotar_para_colab.py`` e copie o zip para
``MyDrive/otft-sr/`` (o notebook extrai o mais recente).

Conteúdo: ``src``, ``prompts``, ``data``, ``references``, ``notebooks``, ``README.md``,
``requirements-colab.txt`` e ``VERSAO_PACOTE.json`` (commit, data, alterações pendentes).
Excluídos: ``results/``, ``.cache/``, ``.git/``, ``__pycache__``, ``.ipynb_checkpoints`` e
arquivos de credencial (``.env``, ``*.key``, ``*.pem``...). Todo arquivo de texto é varrido
com os padrões de chave de API de ``otft_sr.persistencia``: um achado aborta o pacote.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from otft_sr.persistencia import _PADROES_SEGREDO  # noqa: E402

DIRETORIOS = ("src", "prompts", "data", "references", "notebooks")
ARQUIVOS = ("README.md", "requirements-colab.txt", "pytest.ini")
EXCLUIR_DIRS = frozenset({"results", ".cache", ".git", "__pycache__", ".ipynb_checkpoints",
                          ".pytest_cache", "dist"})
EXCLUIR_NOMES = frozenset({".env", ".netrc", "credentials.json", "token.json"})
EXCLUIR_SUFIXOS = (".pyc", ".key", ".pem", ".p12", ".env")


class SegredoEncontrado(RuntimeError):
    """Arquivo com padrão de chave de API: o pacote não é gerado."""


def _git(raiz: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(raiz), *args], capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def arquivos_do_pacote(raiz: Path = RAIZ) -> list[Path]:
    """Arquivos incluídos (caminhos relativos a ``raiz``), em ordem."""
    saida = [Path(a) for a in ARQUIVOS if (raiz / a).is_file()]
    for d in DIRETORIOS:
        for p in sorted((raiz / d).rglob("*")):
            rel = p.relative_to(raiz)
            if not p.is_file() or any(parte in EXCLUIR_DIRS for parte in rel.parts):
                continue
            if p.name in EXCLUIR_NOMES or p.name.endswith(EXCLUIR_SUFIXOS):
                continue
            saida.append(rel)
    return saida


def _verificar_segredos(raiz: Path, arquivos: list[Path]) -> None:
    for rel in arquivos:
        try:
            texto = (raiz / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binário: não contém texto de chave
        if _PADROES_SEGREDO.search(texto):
            raise SegredoEncontrado(f"padrão de chave de API em {rel}: pacote não gerado")


def empacotar(raiz: Path = RAIZ, destino: Path | None = None) -> Path:
    """Gera o zip e devolve o caminho."""
    raiz = Path(raiz)
    destino = Path(destino) if destino is not None else raiz / "dist"
    commit = _git(raiz, "rev-parse", "--short=12", "HEAD") or "sem-git"
    pendentes = bool(_git(raiz, "status", "--porcelain", "--untracked-files=no"))
    arquivos = arquivos_do_pacote(raiz)
    _verificar_segredos(raiz, arquivos)
    versao = {"commit": _git(raiz, "rev-parse", "HEAD") or None, "commit_curto": commit,
              "ramo": _git(raiz, "rev-parse", "--abbrev-ref", "HEAD") or None,
              "alteracoes_pendentes": pendentes,
              "gerado_em": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
              "n_arquivos": len(arquivos)}
    destino.mkdir(parents=True, exist_ok=True)
    zip_path = destino / f"otft-sr-{commit}{'-com-alteracoes' if pendentes else ''}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for rel in arquivos:
            z.write(raiz / rel, rel.as_posix())
        z.writestr("VERSAO_PACOTE.json", json.dumps(versao, ensure_ascii=False, indent=1))
    return zip_path


def main() -> None:
    zip_path = empacotar()
    with zipfile.ZipFile(zip_path) as z:
        n = len(z.namelist())
    print(f"pacote: {zip_path} ({n} arquivos, {zip_path.stat().st_size / 1e6:.2f} MB)")
    print("copie para o Google Drive em MyDrive/otft-sr/ e abra notebooks/"
          "02_multi_llm_descoberta_otft_colab.ipynb no Colab")


if __name__ == "__main__":
    main()
