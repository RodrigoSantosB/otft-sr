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
