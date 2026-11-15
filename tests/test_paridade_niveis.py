"""Nomenclatura dos níveis (N<k> = nível k da proposta, Seção 5.1) e paridade das expressões."""

import pytest

from otft_sr.niveis import escada, nivel

# id atual -> (nome científico, id antes do alinhamento com a proposta, expressão capturada
# do código anterior). N2 e N3 trocaram de posição; N1 e N4 não mudaram.
ESPERADO = {
    "N1": ("Saturação quadrática ideal", "N1", "0.008293 * (V_GS - 20.24) ** 2"),
    "N2": (
        "Regime linear (triodo) com termo quadrático em V_DS",
        "N3",
        "0.01173 * (2 * (V_GS - 25.52) * V_DS - V_DS ** 2)",
    ),
    "N3": ("Saturação com mobilidade em lei de potência", "N2", "4.119e-05 * (V_GS - 8.57) ** 3.18"),
    "N4": ("Corrente sublimiar exponencial", "N4", "0.0006386 * exp(1.423 * V_GS)"),
}


def test_escada_na_ordem_da_proposta():
    assert [n.nivel_proposta for n in escada()] == [1, 2, 3, 4]
    assert [n.id for n in escada()] == ["N1", "N2", "N3", "N4"]
    assert all(n.id == f"N{n.nivel_proposta}" for n in escada())


@pytest.mark.parametrize("id_", ESPERADO)
def test_nome_e_expressao_identica_a_do_nivel_antigo(id_):
    nome, _antigo, expressao = ESPERADO[id_]
    assert nivel(id_).nome == nome
    assert nivel(id_).expressao == expressao


def test_n5_nao_e_executavel():
    with pytest.raises(KeyError):
        nivel("N5")
