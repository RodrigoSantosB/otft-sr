"""Testes focados do projeto otft-sr (sem busca simbólica: rápidos, sem Julia/LLM)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from physics_llm_sr.llm.interfaces import EquationEvaluation
from physics_llm_sr.llm.prompting import VARIANT_IDS
from physics_llm_sr.llm.scoring import FakeEquationEvaluator
from physics_llm_sr.optimization.objective import build_objective
from physics_llm_sr.symbolic.expression import evaluate, parse_infix

from otft_sr.contexto import DIR_PROMPTS, VARIANTE_PROMPT, carregar_biblioteca, contexto_fisico
from otft_sr.dados import NIVEIS_RUIDO, aplicar_ruido, construir_datasets
from otft_sr.experimento import (
    MOTORES,
    PESOS_LLM,
    construir_config,
    controle_positivo,
    executar_condicao,
)
from otft_sr.niveis import FEATURES, erro_vs_ovsed, escada
from otft_sr.ovsed import (
    carregar_parametros_org1,
    corrente_forma_fechada_uA,
    corrente_ovsed_uA,
    erro_log_medido,
)

NIVEIS = escada()


# ----------------------------------------------------------------------------- OVSED


def test_forma_fechada_vs_ovsed_completo_acima_do_limiar():
    p = carregar_parametros_org1()
    vg, vd = np.meshgrid(np.linspace(25, 50, 26), np.linspace(1, 50, 25))
    full = corrente_ovsed_uA(vg, vd, p, com_R_S=False, com_leak=False)
    ff = corrente_forma_fechada_uA(vg, vd, p)
    rel = np.abs(ff / full - 1)
    assert np.max(rel) < 3.5e-3  # pior caso no canto V_GS = 25 V, V_DS = 1 V
    assert np.max(rel[:, vg[0] >= 27]) < 3e-3


def test_V_th0_adotado_reproduz_medidas_melhor_que_json():
    assert erro_log_medido(carregar_parametros_org1()) < 0.2  # décadas
    assert erro_log_medido(carregar_parametros_org1(V_th0=None)) > 0.5


# ----------------------------------------------------------------------------- níveis


@pytest.mark.parametrize("nivel", NIVEIS, ids=lambda n: n.id)
def test_referencia_parseia_e_reproduz_gerador(nivel):
    rng = np.random.default_rng(0)
    vg = rng.uniform(*nivel.regiao.V_GS, 200)
    vd = rng.uniform(*nivel.regiao.V_DS, 200)
    X = np.column_stack([vg, vd])
    y = nivel.gerador(vg, vd)
    for texto in (nivel.expressao, *nivel.formas_equivalentes):
        yhat = evaluate(parse_infix(texto, FEATURES), X, FEATURES)
        np.testing.assert_allclose(yhat, y, rtol=1e-3)


@pytest.mark.parametrize("nivel", NIVEIS, ids=lambda n: n.id)
def test_nivel_proximo_do_ovsed_na_regiao(nivel):
    assert erro_vs_ovsed(nivel)["rmse_log10"] < 0.15  # < 0,15 década


@pytest.mark.parametrize("nivel", NIVEIS, ids=lambda n: n.id)
def test_controle_positivo_pelo_pipeline(nivel):
    r = controle_positivo(nivel)
    assert r["one_minus_r2"] < 1e-12
    assert r["e_normalizado"] < 1e-12
    assert r["tree_distance"] == 0.0
    assert r["tree_distance_algebraic"] == 0.0
    assert r["forma_funcional_sympy"]


def test_equivalencia_sympy_reconhece_formas_reescritas():
    from otft_sr.equivalencia import mesma_forma_funcional

    P = lambda s: parse_infix(s, FEATURES)  # noqa: E731
    n1, n2, _, n4 = NIVEIS
    assert mesma_forma_funcional(P("log(1.00064 ** (4.15 ** V_GS))"), P(n4.expressao))
    assert mesma_forma_funcional(P("exp(1.42 * V_GS - 7.35)"), P(n4.expressao))
    assert mesma_forma_funcional(P("0.0083 * V_GS ** 2 - 0.33 * V_GS + 3.4"), P(n1.expressao))
    assert not mesma_forma_funcional(P("1.048 ** V_GS - 3.33"), P(n1.expressao))
    assert not mesma_forma_funcional(P("4.1e-05 * (V_GS - 8.6) ** 3.0"), P(n2.expressao))
    assert mesma_forma_funcional(P("(V_GS / 23.52 - 0.3644) ** 3.18"), P(n2.expressao))
    assert not mesma_forma_funcional(P("(V_GS / 23.52 - 0.382) ** 3.137"), P(n2.expressao))


# ----------------------------------------------------------------------------- dados


@pytest.mark.parametrize("nivel", NIVEIS, ids=lambda n: n.id)
def test_particao_por_curva_sem_vazamento(nivel):
    ds = construir_datasets(nivel, 0.05, semente=3)
    ids = {k: set(np.unique(v.simulation_ids)) for k, v in ds.items()}
    assert ids["treino"] and ids["validacao"] and ids["teste"]
    assert not ids["treino"] & ids["validacao"]
    assert not ids["treino"] & ids["teste"]
    assert not ids["validacao"] & ids["teste"]
    # nenhum par (V_GS, V_DS) do teste aparece no treino
    tr = {tuple(r) for r in np.round(ds["treino"].X, 9)}
    assert not any(tuple(r) in tr for r in np.round(ds["teste"].X, 9))
    # pontos dentro da região
    r = nivel.regiao
    for d in ds.values():
        assert np.all((d.X[:, 0] >= r.V_GS[0]) & (d.X[:, 0] <= r.V_GS[1]))
        assert np.all(d.y_true > 0)


def test_ruido_multiplicativo_proporcional():
    rng = np.random.default_rng(1)
    y = np.logspace(-3, 3, 20000)
    for s in NIVEIS_RUIDO:
        r = aplicar_ruido(y, s, rng) / y - 1
        assert abs(np.std(r) - s) < 0.01 * max(s, 1e-3) + 1e-3


# ----------------------------------------------------------------------------- objetivo


def test_objetivo_com_fake_evaluator_chama_c_e_condicao_A_nao():
    nivel = NIVEIS[1]
    ds = construir_datasets(nivel, 0.0, 0)["treino"]
    ctx = contexto_fisico(nivel)
    fake = FakeEquationEvaluator.from_function(
        lambda e, c, v: EquationEvaluation(1.0, 0.5, 0.0, "teste")
    )
    cfg = construir_config("deap")
    exprs = [parse_infix(nivel.expressao, FEATURES), parse_infix("V_GS", FEATURES)]
    obj = build_objective(cfg, fake, ctx, "deap", baseline=False, feature_names=FEATURES)
    out = obj.evaluate_population(exprs, ds.X, ds.y_observed)
    assert obj.weights == PESOS_LLM
    assert fake.calls == 2
    assert out[0].c == pytest.approx(0.5)
    assert out[0].e < 1e-12
    assert out[0].total == pytest.approx(PESOS_LLM[1] * 7 / 30 + PESOS_LLM[2] * 0.5)
    base = build_objective(cfg, fake, ctx, "deap", baseline=True, feature_names=FEATURES)
    base.evaluate_population(exprs, ds.X, ds.y_observed)
    assert fake.calls == 2  # w3 = 0: avaliador não chamado


def test_condicao_com_llm_exige_avaliador():
    with pytest.raises(ValueError):
        executar_condicao(NIVEIS[0], "deap", "C")


@pytest.mark.parametrize("motor", MOTORES)
def test_config_valida_por_motor(motor):
    for orc in ("fumaca", "completo"):
        cfg = construir_config(motor, orc)
        assert cfg.loss.normalization == "normalized"
        assert cfg.loss.max_nodes == 30


# ----------------------------------------------------------------------------- prompts


@pytest.mark.parametrize("condicao", ["B", "C"])
def test_prompts_carregam_8_variantes(condicao):
    lib = carregar_biblioteca(condicao)
    assert set(lib.variants) == set(VARIANT_IDS)


def _numeros(texto: str) -> set[str]:
    return set(re.findall(r"\d+\.\d+(?:e[-+]?\d+)?", texto))


@pytest.mark.parametrize("condicao", ["B", "C"])
@pytest.mark.parametrize("nivel", NIVEIS, ids=lambda n: n.id)
def test_prompt_nao_revela_equacao_alvo(condicao, nivel):
    lib = carregar_biblioteca(condicao)
    arquivos = "\n".join(
        p.read_text(encoding="utf-8") for p in (DIR_PROMPTS).rglob("*") if p.is_file()
    )
    assert nivel.expressao not in arquivos
    sonda = parse_infix("V_GS * V_DS", FEATURES)
    r = lib.render(sonda, contexto_fisico(nivel), VARIANTE_PROMPT)
    assert not r.gt_disclosed
    texto = r.system + r.user
    assert nivel.expressao not in texto
    constantes = _numeros(nivel.expressao)
    assert not constantes & _numeros(texto), constantes & _numeros(texto)


def test_prompt_dispositivo_tem_vocabulario_de_otft():
    s = carregar_biblioteca("C").system_template
    for termo in ("C_i", "V_Th", "W/L", "hopping", "contact resistance", "saturation"):
        assert termo in s
    assert "classical physics" in carregar_biblioteca("B").system_template
