"""Notebook 01: sincronia com o builder e execução de ponta a ponta em verificação."""

import importlib.util
from pathlib import Path

import nbformat
import pytest

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOKS = RAIZ / "notebooks"
IPYNB = NOTEBOOKS / "01_rs_otft.ipynb"


def _carregar_builder():
    spec = importlib.util.spec_from_file_location("build_01_rs_otft", NOTEBOOKS / "build_01_rs_otft.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _fontes(nb):
    return [(c.cell_type, c.source) for c in nb.cells]


def test_notebook_sincronizado_com_builder(tmp_path):
    destino = tmp_path / "gerado.ipynb"
    _carregar_builder().main(destino)
    gerado = nbformat.read(destino, as_version=4)
    versionado = nbformat.read(IPYNB, as_version=4)
    assert _fontes(gerado) == _fontes(versionado)


@pytest.mark.lento
def test_notebook_executa_em_verificacao():
    """Executa o notebook (cópia em tmp) com configuração de verificação e sem LLM."""
    import re

    from nbclient import NotebookClient

    nb = nbformat.read(IPYNB, as_version=4)
    alvo = next(c for c in nb.cells if c.cell_type == "code" and "BACKEND_LLM =" in c.source)
    fonte = re.sub(r'^CONFIGURACAO_BUSCA = "[^"]*"', 'CONFIGURACAO_BUSCA = "verificacao"', alvo.source, flags=re.M)
    fonte = re.sub(r'^BACKEND_LLM = "[^"]*"', 'BACKEND_LLM = "nenhum"', fonte, flags=re.M)
    assert re.search(r'^CONFIGURACAO_BUSCA = "verificacao"', fonte, flags=re.M)
    assert re.search(r'^BACKEND_LLM = "nenhum"', fonte, flags=re.M)
    alvo.source = fonte

    cliente = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOKS)}})
    cliente.execute()  # levanta CellExecutionError se a execução falhar
    erros = [
        s for c in nb.cells if c.cell_type == "code" for s in c.get("outputs", [])
        if s.get("output_type") == "error"
    ]
    assert erros == []
