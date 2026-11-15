"""Terminologia técnico-científica (R1): paridade numérica e varredura de termos vetados."""

import json
from pathlib import Path

import pytest

from otft_sr.experimento import ALGORITMOS_RS, construir_config

SNAPSHOT = Path(__file__).parent / "fixtures" / "configuracoes_busca_snapshot.json"


@pytest.mark.parametrize("algoritmo", ["pysr", "deap", "gplearn"])
@pytest.mark.parametrize("configuracao_busca", ["verificacao", "calibrada"])
def test_paridade_configuracoes_busca(algoritmo, configuracao_busca):
    """A renomeação não altera nenhum valor: config nova == snapshot gerado antes."""
    assert algoritmo in ALGORITMOS_RS
    esperado = json.loads(SNAPSHOT.read_text(encoding="utf-8"))[f"{algoritmo}/{configuracao_busca}"]
    atual = construir_config(algoritmo, configuracao_busca, semente=0).model_dump(mode="json")
    assert atual == esperado
