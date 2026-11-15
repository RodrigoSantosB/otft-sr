"""Caracterização: as expressões de referência dos níveis não mudam com a renomeação."""

from otft_sr.niveis import nivel

# Capturadas do código anterior à renomeação (ids N1-N4).
EXPRESSOES_REFERENCIA = {
    "N1": "0.008293 * (V_GS - 20.24) ** 2",
    "N2": "4.119e-05 * (V_GS - 8.57) ** 3.18",
    "N3": "0.01173 * (2 * (V_GS - 25.52) * V_DS - V_DS ** 2)",
    "N4": "0.0006386 * exp(1.423 * V_GS)",
}


def test_expressoes_identicas_as_capturadas():
    for id_, expr in EXPRESSOES_REFERENCIA.items():
        assert nivel(id_).expressao == expr
