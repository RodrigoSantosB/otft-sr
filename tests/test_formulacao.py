"""Formulação da perda do M1 em log10(I_D) (H-OTFT-19) e localização do physics-llm-sr."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from physics_llm_sr.symbolic.expression import evaluate, parse_infix

from otft_sr import experimento
from otft_sr.contexto import contexto_fisico
from otft_sr.dados import construir_datasets
from otft_sr.experimento import controle_positivo, executar_condicao
from otft_sr.formulacao import (
    FORMULACOES,
    datasets_formulados,
    nivel_formulado,
)
from otft_sr.niveis import FEATURES, escada, nivel
from otft_sr.persistencia import chave_ensaio


def test_formulacoes_disponiveis():
    assert FORMULACOES == ("linear", "log10")
    with pytest.raises(ValueError):
        nivel_formulado(nivel("N1"), "ln")


def test_linear_e_identidade():
    n = nivel("N3")
    assert nivel_formulado(n, "linear") is n
    ds = construir_datasets(n, 0.0, 0)
    assert datasets_formulados(ds, "linear") is ds


@pytest.mark.parametrize("n", escada(), ids=lambda n: n.id)
def test_referencia_log10_reproduz_log10_do_gerador(n):
    nl = nivel_formulado(n, "log10")
    assert nl.id == n.id and nl.unidades["I_D"] == f"log10({n.unidades['I_D']})"
    rng = np.random.default_rng(1)
    vg = rng.uniform(*n.regiao.V_GS, 100)
    vd = rng.uniform(*n.regiao.V_DS, 100)
    if n.regiao.V_DS_max_rel_V_GS is not None:
        vd = np.minimum(vd, vg - n.regiao.V_DS_max_rel_V_GS - 0.1)
    X = np.column_stack([vg, vd])
    esperado = np.log10(n.gerador(vg, vd))
    y = evaluate(parse_infix(nl.expressao, FEATURES), X, FEATURES)
    assert np.allclose(y, esperado, rtol=1e-9, atol=1e-9)
    for texto in nl.formas_equivalentes:  # as formas lineares já trazem constantes arredondadas
        y = evaluate(parse_infix(texto, FEATURES), X, FEATURES)
        assert np.allclose(y, esperado, atol=1e-3), texto
    assert np.allclose(nl.gerador(vg, vd), esperado)


def test_datasets_log10_transformam_alvo_e_preservam_particao():
    n = nivel("N4")
    ds = construir_datasets(n, 0.05, 0)
    dl = datasets_formulados(ds, "log10")
    for k in ds:
        assert np.array_equal(dl[k].X, ds[k].X)
        assert np.array_equal(dl[k].simulation_ids, ds[k].simulation_ids)
        assert np.allclose(dl[k].y_true, np.log10(ds[k].y_true))
        assert np.all(np.isfinite(dl[k].y_observed))
        assert dl[k].metadata["formulacao_perda"] == "log10"


def test_contexto_log10_declara_alvo_sem_revelar_equacao():
    n = nivel_formulado(nivel("N1"), "log10")
    ctx = contexto_fisico(n)
    texto = ctx.experiment_description + " ".join(ctx.variable_descriptions.values())
    assert "log10" in texto
    assert "V_Th" not in texto and str(nivel("N1").constantes["K"]) not in texto
    assert "log10" not in contexto_fisico(nivel("N1")).experiment_description


@pytest.mark.parametrize("nid", ["N1", "N2", "N3", "N4"])
def test_controle_positivo_log10(nid):
    cp = controle_positivo(nivel(nid), formulacao_perda="log10")
    assert cp["formulacao_perda"] == "log10"
    assert cp["one_minus_r2"] < 1e-12
    assert cp["rmse_log10"] < 1e-9
    assert cp["one_minus_r2_I_D"] < 1e-12
    assert cp["tree_distance"] == 0


def test_controle_positivo_linear_mantem_colunas():
    cp = controle_positivo(nivel("N1"))
    assert cp["formulacao_perda"] == "linear"
    assert math.isclose(cp["one_minus_r2_I_D"], cp["one_minus_r2"], abs_tol=1e-15)


def test_executar_condicao_log10_deap_minimo():
    linha = executar_condicao(
        nivel("N3"), "deap", "A", 0.0, 0,
        {"sr.population": 8, "sr.generations": 1, "sr.early_stopping.enabled": False},
        formulacao_perda="log10",
    )
    assert linha["status"] == "ok", linha.get("erro")
    assert linha["formulacao_perda"] == "log10"
    assert "log(" in linha["expressao_referencia"]
    assert np.isfinite(linha["rmse_log10"]) and np.isfinite(linha["one_minus_r2_I_D"])


def test_executar_condicao_formulacao_invalida():
    with pytest.raises(ValueError):
        executar_condicao(nivel("N1"), "deap", "A", formulacao_perda="ln")


def test_chave_ensaio_distingue_log10_e_preserva_linear():
    base = {"nivel": "N1", "modo": "M1", "condicao": "A", "algoritmo": "deap",
            "modelo": "nenhum", "ruido": 0.0, "semente": 0}
    assert chave_ensaio(base) == chave_ensaio({**base, "formulacao_perda": "linear"})
    assert chave_ensaio(base) != chave_ensaio({**base, "formulacao_perda": "log10"})


def test_repo_physics_llm_sr_por_variavel_de_ambiente(monkeypatch, tmp_path):
    (tmp_path / "configs").mkdir()
    monkeypatch.setenv("PHYSICS_LLM_SR_REPO_DIR", str(tmp_path))
    assert experimento.repo_physics_llm_sr() == tmp_path
    monkeypatch.delenv("PHYSICS_LLM_SR_REPO_DIR")
    repo = experimento.repo_physics_llm_sr()
    assert (repo / "configs" / "nb01_baseline.yaml").is_file()
    assert isinstance(repo, Path)


# ------------------------------------------- critério de recuperação comum (H-OTFT-19)


def _linha_classificavel(**kw):
    base = {"status": "ok", "tree_distance": 0.0, "tree_distance_equivalent_min": 0.0,
            "tree_distance_algebraic": 0.0, "forma_funcional_sympy": True}
    return {**base, **kw}


def test_criterio_de_ajuste_em_I_D_linear_na_formulacao_log10():
    """Ajuste excelente em log10 mas ruim em I_D não recupera; o critério é sempre I_D."""
    from otft_sr.experimento import classificar_linha

    ruim_em_I_D = _linha_classificavel(formulacao_perda="log10", one_minus_r2=1e-6,
                                       one_minus_r2_I_D=0.05)
    assert not classificar_linha(ruim_em_I_D)["classe"].startswith("recupera estrutura")
    bom_em_I_D = _linha_classificavel(formulacao_perda="log10", one_minus_r2=0.01,
                                      one_minus_r2_I_D=1e-4)
    assert classificar_linha(bom_em_I_D)["classe"] == "recupera estrutura"
    sympy = _linha_classificavel(formulacao_perda="log10", one_minus_r2=0.01,
                                 one_minus_r2_I_D=1e-4, tree_distance=1.0,
                                 tree_distance_equivalent_min=1.0, tree_distance_algebraic=1.0)
    assert classificar_linha(sympy)["classe"].startswith("recupera estrutura (equivalência")


def test_criterio_linha_antiga_sem_I_D_usa_one_minus_r2():
    from otft_sr.experimento import classificar_linha

    antiga = _linha_classificavel(one_minus_r2=1e-5)
    assert classificar_linha(antiga)["classe"] == "recupera estrutura"
