"""Nomenclatura dos níveis (N<k> = nível k da proposta, Seção 5.1) e paridade das expressões.

N1 e N2 têm a expressão idêntica à do código anterior ao alinhamento com a proposta. N3 e N4
mudaram deliberadamente de região (H-OTFT-17, H-OTFT-18): as expressões novas ficam
registradas aqui ao lado das anteriores, e o critério que justifica cada mudança é testado com
valores fixados (obtidos na análise de escolha das regiões, não recalculados pelo código).
"""

import dataclasses

import numpy as np
import pytest
from scipy.optimize import least_squares

from otft_sr.dados import construir_datasets
from otft_sr.niveis import (
    _grade_regiao,
    discriminabilidade_quadratica,
    erro_vs_medida_org1,
    erro_vs_ovsed,
    escada,
    nivel,
)

# id atual -> (nome científico, id antes do alinhamento com a proposta, expressão atual)
ESPERADO = {
    "N1": ("Saturação quadrática ideal", "N1", "0.008293 * (V_GS - 20.24) ** 2"),
    "N2": (
        "Regime linear (triodo) com termo quadrático em V_DS",
        "N3",
        "0.01173 * (2 * (V_GS - 25.52) * V_DS - V_DS ** 2)",
    ),
    "N3": ("Saturação com mobilidade em lei de potência", "N2", "2.2e-05 * (V_GS - 8.57) ** 3.18"),
    "N4": ("Corrente sublimiar exponencial", "N4", "65.0 + 0.009502 * exp(0.9882 * V_GS)"),
}

#: Expressões e regiões anteriores de N3 e N4 (código antes da revisão das regiões).
ANTERIOR = {
    "N3": ("4.119e-05 * (V_GS - 8.57) ** 3.18", (20.0, 50.0)),
    "N4": ("0.0006386 * exp(1.423 * V_GS)", (0.0, 4.0)),
}


def test_escada_na_ordem_da_proposta():
    assert [n.nivel_proposta for n in escada()] == [1, 2, 3, 4]
    assert [n.id for n in escada()] == ["N1", "N2", "N3", "N4"]
    assert all(n.id == f"N{n.nivel_proposta}" for n in escada())


@pytest.mark.parametrize("id_", ESPERADO)
def test_nome_e_expressao_registrados(id_):
    nome, _antigo, expressao = ESPERADO[id_]
    assert nivel(id_).nome == nome
    assert nivel(id_).expressao == expressao


@pytest.mark.parametrize("id_", ["N3", "N4"])
def test_n3_n4_mudaram_de_regiao_e_de_expressao(id_):
    expressao_anterior, regiao_anterior = ANTERIOR[id_]
    assert nivel(id_).expressao != expressao_anterior
    assert nivel(id_).regiao.V_GS != regiao_anterior


def test_regioes_novas():
    assert nivel("N3").regiao.V_GS == (10.5, 28.0)
    assert nivel("N3").regiao.V_DS == (40.0, 50.0)
    assert nivel("N4").regiao.V_GS == (0.0, 10.0)
    assert nivel("N4").regiao.com_leak
    assert nivel("N4").constantes["I_leak"] == 65.0
    assert nivel("N4").unidades["I_D"] == "pA"


def test_n5_nao_e_executavel():
    with pytest.raises(KeyError):
        nivel("N5")


# ----------------------------------------------------------------------------- N3: discriminabilidade


def _quadratica_em_log_independente(vg, y):
    """Melhor K (V_GS - V_T)^2 em log10 por least_squares (caminho distinto do de niveis.py)."""
    melhor = np.inf
    for vt0 in np.arange(-20.0, vg.min(), 1.0):
        r = least_squares(
            lambda x: x[0] + 2 * np.log10(np.abs(vg - x[1])) - np.log10(y),
            [np.log10(np.mean(y) / np.mean((vg - vt0) ** 2)), vt0],
            bounds=([-np.inf, -50.0], [np.inf, vg.min() - 1e-6]),
        )
        melhor = min(melhor, float(np.sqrt(np.mean(r.fun**2))))
    return melhor


def test_n3_discriminavel_da_lei_quadratica():
    n3 = nivel("N3")
    d = discriminabilidade_quadratica(n3)
    # valores da análise de escolha da região (V_GS em [10,5; 28] V)
    assert d["rmse_log10_quadratica"] == pytest.approx(0.155, abs=0.01)
    assert d["one_minus_r2_quadratica"] == pytest.approx(3.8e-3, rel=0.1)
    vg, vd = _grade_regiao(n3.regiao)
    assert _quadratica_em_log_independente(vg, n3.gerador(vg, vd)) == pytest.approx(0.155, abs=0.01)
    # região anterior (V_GS em [20, 50] V): 0,052 década, ~3x menor
    antiga = dataclasses.replace(n3, regiao=dataclasses.replace(n3.regiao, V_GS=(20.0, 50.0)))
    d_ant = discriminabilidade_quadratica(antiga)
    assert d_ant["rmse_log10_quadratica"] == pytest.approx(0.052, abs=0.005)
    assert d["rmse_log10_quadratica"] > 2.5 * d_ant["rmse_log10_quadratica"]
    # a forma quadrática reproduz o próprio N1 (sanidade do ajuste)
    assert discriminabilidade_quadratica(nivel("N1"))["rmse_log10_quadratica"] < 1e-9


def test_n3_erro_vs_ovsed_aceitavel():
    e = erro_vs_ovsed(nivel("N3"))
    assert e["rmse_log10"] == pytest.approx(0.140, abs=0.005)
    assert e["rmse_log10"] <= 0.15 and e["max_log10"] <= 0.30


# ----------------------------------------------------------------------------- N4: fuga aditiva


def test_n4_concorda_com_org1_medida():
    n4 = nivel("N4")
    m = erro_vs_medida_org1(n4)  # org1_2VDS, V_GS em [4,4; 10] V
    assert m["n_pontos"] == 6
    assert m["rmse_log10"] == pytest.approx(0.033, abs=0.01)
    assert m["max_log10"] < 0.10
    e = erro_vs_ovsed(n4)
    assert e["rmse_log10"] == pytest.approx(0.024, abs=0.005)
    assert e["max_log10"] < 0.2
    assert n4.constantes["S"] == pytest.approx(2.33, abs=0.01)  # V/déc, swing efetivo


def test_n4_sem_fuga_nao_descreve_a_medida():
    """O exponencial puro (sem I_leak) fica décadas abaixo da medida na região do N4."""
    from otft_sr.niveis import pontos_medidos_na_regiao

    n4 = nivel("N4")
    vg, vd, I = pontos_medidos_na_regiao(n4)
    canal = n4.gerador(vg, vd) - n4.constantes["I_leak"]
    assert np.sqrt(np.mean(np.log10(canal / I) ** 2)) > 1.0


def test_n4_variancia_em_pA_acima_do_minimo_normalizado():
    from physics_llm_sr.optimization.objective import MIN_TARGET_VARIANCE

    for d in construir_datasets(nivel("N4"), 0.0, 0).values():
        assert np.var(d.y_true) > 100.0 > MIN_TARGET_VARIANCE  # pA^2 (~1e3)
