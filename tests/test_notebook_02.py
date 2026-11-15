"""Notebook 02: sincronia com o builder e execução de verificação de ponta a ponta."""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import nbformat
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOKS = RAIZ / "notebooks"
IPYNB = NOTEBOOKS / "02_multi_llm_descoberta_otft_colab.ipynb"


def _carregar_builder():
    spec = importlib.util.spec_from_file_location(
        "build_02_multi_llm_colab", NOTEBOOKS / "build_02_multi_llm_colab.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _fontes(nb):
    return [(c.cell_type, c.source) for c in nb.cells]


def test_notebook_02_sincronizado_com_builder(tmp_path):
    destino = tmp_path / "gerado.ipynb"
    _carregar_builder().main(destino)
    assert _fontes(nbformat.read(destino, as_version=4)) == _fontes(nbformat.read(IPYNB, as_version=4))


def test_notebook_02_secoes_e_sem_saidas():
    nb = nbformat.read(IPYNB, as_version=4)
    titulos = [re.search(r"^## (\d+)\.", c.source, flags=re.M) for c in nb.cells if c.cell_type == "markdown"]
    assert [int(m.group(1)) for m in titulos if m] == list(range(1, 17))
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")


def preparar(nb, backend: str = "simulado", extra: dict[str, str] | None = None):
    """Força EXECUCAO_DE_VERIFICACAO = True e o backend de verificação no bloco de configuração."""
    alvo = next(c for c in nb.cells if c.cell_type == "code" and "EXECUCAO_DE_VERIFICACAO =" in c.source)
    fonte = re.sub(r"^EXECUCAO_DE_VERIFICACAO = \w+", "EXECUCAO_DE_VERIFICACAO = True", alvo.source, flags=re.M)
    fonte = re.sub(r'^BACKEND_VERIFICACAO = "[^"]*"', f'BACKEND_VERIFICACAO = "{backend}"', fonte, flags=re.M)
    for antigo, novo in (extra or {}).items():
        assert antigo in fonte, antigo
        fonte = fonte.replace(antigo, novo)
    assert re.search(r"^EXECUCAO_DE_VERIFICACAO = True", fonte, flags=re.M)
    alvo.source = fonte
    return nb


def executar(nb, dir_saida: Path, timeout: int = 3600, env_extra: dict[str, str] | None = None):
    from nbclient import NotebookClient

    env = {"OTFT_SR_DIR_RESULTADOS": str(dir_saida / "results"),
           "OTFT_SR_DIR_CACHE_LLM": str(dir_saida / "cache_llm"), **(env_extra or {})}
    antigo = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        NotebookClient(nb, timeout=timeout, kernel_name="python3",
                       resources={"metadata": {"path": str(NOTEBOOKS)}}).execute()
    finally:
        for k, v in antigo.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return [s for c in nb.cells if c.cell_type == "code" for s in c.get("outputs", [])
            if s.get("output_type") == "error"]


@pytest.mark.lento
def test_notebook_02_executa_em_verificacao_com_backend_simulado(tmp_path):
    """Configuração mínima, backend simulado, Ollama desligado; o código do Colab é pulado."""
    nb = preparar(nbformat.read(IPYNB, as_version=4), "simulado")
    erros = executar(nb, tmp_path, env_extra={"OLLAMA_HOST": "http://127.0.0.1:9"})
    assert erros == []
    d = tmp_path / "results" / "otft_multi_llm_v1_verificacao"
    m = pd.read_csv(d / "metrics.csv")
    # A (1) + M2 G0/G1/G2 (3) + M1 B/C (2) com um modelo simulado
    assert len(m) == 6 and set(m["modo"]) == {"M1", "M2"}
    assert (m["status"] == "ok").all(), m[["modo", "condicao", "erro"]]
    for arq in ("config.json", "environment.json", "metrics_summary.csv", "candidates.csv",
                "auditoria_avaliador.csv", "benchmark_recuperacao.csv", "uso_llm.json"):
        assert (d / arq).is_file(), arq
    texto = "\n".join(str(s.get("text", "")) + str(s.get("data", {}).get("text/markdown", ""))
                      for c in nb.cells if c.cell_type == "code" for s in c.get("outputs", []))
    assert "não constitui resultado" in texto
    assert "Colab: False" in texto

    # retomada: segunda execução não refaz ensaios
    nb2 = preparar(nbformat.read(IPYNB, as_version=4), "simulado")
    assert executar(nb2, tmp_path, env_extra={"OLLAMA_HOST": "http://127.0.0.1:9"}) == []
    log = (d / "execution_log.jsonl").read_text()
    depois = log.split('"experimento_retomado"')[-1]
    assert '"ensaio_registrado"' not in depois
    assert len(pd.read_csv(d / "metrics.csv")) == 6
