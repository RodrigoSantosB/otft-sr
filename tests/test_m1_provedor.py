"""M1 com provedor arbitrário, auditoria do avaliador, responder simulado e razão de tempo."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from physics_llm_sr.symbolic.expression import parse_infix

from otft_sr.auditoria import (
    CATEGORIAS_AUDITORIA,
    auditar_avaliador,
    expressoes_auditoria,
    resumo_discriminacao,
)
from otft_sr.estatistica import RAZAO_TEMPO_LLM_RS_ARTIGO, tabela_razao_tempo_llm
from otft_sr.experimento import executar_m1
from otft_sr.gerador import executar_geracao, validar_resposta
from otft_sr.niveis import FEATURES, nivel
from otft_sr.provedores_llm import (
    ConfiguracaoLLM,
    RegistroUso,
    criar_avaliador_m1,
    criar_provedor,
    responder_simulado,
)

MINIMA = {"sr.population": 8, "sr.generations": 1, "sr.early_stopping.enabled": False}


def _fake(resposta=responder_simulado):
    return ConfiguracaoLLM("fake", "simulado", resposta_fake=resposta)


# ----------------------------------------------------------------- responder simulado


def test_responder_simulado_avaliador_devolve_escores():
    d = json.loads(responder_simulado("You are a scientific reviewer", "### CANDIDATE EQUATION\nx"))
    assert set(d) >= {"dimensional_consistency", "simplicity", "physical_realism"}


def test_responder_simulado_gerador_devolve_equacao_valida():
    from otft_sr.dados import construir_datasets
    from otft_sr.gerador import montar_prompt

    ds = construir_datasets(nivel("N3"), 0.0, 0)
    sistema, usuario = montar_prompt(nivel("N3"), "G1", ds["treino"], 1, 2)
    r = validar_resposta(responder_simulado(sistema, usuario))
    parse_infix(r.equation, (*FEATURES, *r.parameters))
    assert responder_simulado(sistema, usuario) == responder_simulado(sistema, usuario)


def test_m2_com_responder_simulado_termina():
    linha, cands = executar_geracao(nivel("N3"), "G2", _fake(), 0, 2, 1, n_curvas=6, n_pontos=10)
    assert linha["n_candidatos_avaliados"] == 4
    assert linha["formulacao_perda"] == "ajuste_log10"


# ---------------------------------------------------------------------------- M1


def test_executar_m1_condicao_A_sem_modelo():
    linha = executar_m1(nivel("N1"), "deap", "A", 0.0, 0, MINIMA, n_curvas=6, n_pontos=10)
    assert linha["status"] == "ok", linha.get("erro")
    assert linha["modo"] == "M1" and linha["modelo"] == "nenhum"
    assert linha["chamadas_llm"] == 0


def test_executar_m1_condicao_A_recusa_provedor():
    with pytest.raises(ValueError):
        executar_m1(nivel("N1"), "deap", "A", cfg_llm=_fake())


def test_executar_m1_condicao_B_exige_cfg():
    with pytest.raises(ValueError):
        executar_m1(nivel("N1"), "deap", "B")


@pytest.mark.parametrize("formulacao", ["linear", "log10"])
def test_executar_m1_com_provedor_reusado(tmp_path, formulacao):
    cfg = _fake()
    reg = RegistroUso()
    prov = criar_provedor(cfg, registro=reg)
    linhas = [
        executar_m1(nivel("N3"), "deap", c, 0.0, 0, MINIMA, cfg_llm=cfg, provedor=prov,
                    registro_uso=reg, dir_cache=tmp_path, n_curvas=6, n_pontos=10,
                    formulacao_perda=formulacao)
        for c in ("B", "C")
    ]
    for linha, c in zip(linhas, ("B", "C"), strict=True):
        assert linha["status"] == "ok", linha.get("erro")
        assert linha["condicao"] == c and linha["modelo"] == "fake/simulado"
        assert linha["formulacao_perda"] == formulacao
        assert linha["chamadas_llm"] > 0
        assert math.isfinite(linha["razao_tempo_llm_rs"])
        meta = json.loads(linha["metadados_modelo"])
        assert {"provedor", "modelo", "revisao", "quantizacao", "dtype", "parametros_geracao",
                "determinismo"} <= set(meta)
        assert meta["provedor"] == "fake" and meta["determinismo"] == "deterministico"
    assert reg.obter("fake/simulado").chamadas > 0
    assert any(tmp_path.glob("avaliacoes_B__*.jsonl"))


# ------------------------------------------------------------------------ auditoria


@pytest.mark.parametrize("nid", ["N1", "N2", "N3", "N4"])
@pytest.mark.parametrize("formulacao", ["linear", "log10"])
def test_expressoes_auditoria_parseiam_e_cobrem_categorias(nid, formulacao):
    exprs = expressoes_auditoria(nivel(nid), formulacao)
    assert {e["categoria"] for e in exprs} == set(CATEGORIAS_AUDITORIA)
    assert sum(e["categoria"] == "errada_plausivel" for e in exprs) >= 2
    for e in exprs:
        parse_infix(e["expressao"], FEATURES)
    assert len({e["expressao"] for e in exprs}) == len(exprs)


def _por_rotulo(mapa):
    """Responder que atribui escore conforme a expressão do prompt."""

    def responder(system, user):
        for trecho, v in mapa.items():
            if f"CANDIDATE EQUATION\n{trecho}\n" in user:
                return json.dumps({"dimensional_consistency": v, "simplicity": v,
                                   "physical_realism": v, "feedback": "x"})
        return json.dumps({"dimensional_consistency": 0.5, "simplicity": 0.5,
                           "physical_realism": 0.5, "feedback": "x"})

    return responder


def test_auditoria_detecta_empate_entre_referencia_e_forma_errada(tmp_path):
    n = nivel("N1")
    exprs = expressoes_auditoria(n)
    ref = next(e for e in exprs if e["categoria"] == "referencia")["expressao"]
    errada = next(e for e in exprs if e["categoria"] == "errada_plausivel")["expressao"]
    # o prompt traz a expressão já reformatada pelo parser: usa o texto infixo canônico
    from physics_llm_sr.symbolic.expression import to_infix

    mapa = {to_infix(parse_infix(ref, FEATURES)): 0.9, to_infix(parse_infix(errada, FEATURES)): 0.9}
    av = criar_avaliador_m1(_fake(_por_rotulo(mapa)), "C", dir_cache=None)
    df = auditar_avaliador(av, n)
    assert set(df["categoria"]) == set(CATEGORIAS_AUDITORIA)
    assert df["valida"].all()
    res = resumo_discriminacao(df)
    assert res["c_referencia"] == pytest.approx(0.1)
    assert res["n_erradas_com_c_menor_ou_igual_a_referencia"] >= 1
    assert res["discrimina_erradas"] is False


def test_auditoria_discrimina_quando_referencia_tem_menor_c():
    n = nivel("N3")
    from physics_llm_sr.symbolic.expression import to_infix

    ref = next(e for e in expressoes_auditoria(n) if e["categoria"] == "referencia")["expressao"]
    av = criar_avaliador_m1(
        _fake(_por_rotulo({to_infix(parse_infix(ref, FEATURES)): 1.0})), "B", dir_cache=None)
    res = resumo_discriminacao(auditar_avaliador(av, n))
    assert res["discrimina_erradas"] is True and res["discrimina_absurda"] is True
    assert res["margem_erradas"] == pytest.approx(0.5)


# --------------------------------------------------------------------- razão de tempo


def test_tabela_razao_tempo_llm():
    df = pd.DataFrame({
        "nivel": ["N1", "N1", "N1"], "modo": ["M1"] * 3, "condicao": ["A", "B", "B"],
        "modelo": ["nenhum", "m", "m"], "tempo_s": [10.0, 14.0, 20.0],
        "tempo_llm_s": [0.0, 4.0, 10.0],
    })
    t = tabela_razao_tempo_llm(df)
    b = t[t["condicao"] == "B"].iloc[0]
    assert b["razao_tempo_llm_rs_media"] == pytest.approx((0.4 + 1.0) / 2)
    assert b["razao_artigo"] == RAZAO_TEMPO_LLM_RS_ARTIGO == 0.4
    assert t[t["condicao"] == "A"].empty  # sem LLM: fora da tabela
    assert np.isclose(b["n_ensaios"], 2)


def test_razao_de_tempo_so_M1_e_M2_separado():
    from otft_sr.estatistica import tabela_tempo_inferencia_m2

    df = pd.DataFrame({
        "nivel": ["N1"] * 4, "modo": ["M1", "M2", "M2", "M2"], "condicao": ["B", "G1", "G1", "G1"],
        "modelo": ["m"] * 4, "algoritmo": ["deap"] + ["llm_gerador"] * 3,
        "tempo_s": [14.0, 10.0, 30.0, 99.0], "tempo_llm_s": [4.0, 8.0, 20.0, 1.0],
        "tempo_inferencia_s": [4.0, 8.0, 20.0, 1.0], "status": ["ok", "ok", "ok", "erro_execucao"],
    })
    t = tabela_razao_tempo_llm(df)
    assert list(t["modo"]) == ["M1"]  # comparável ao 0,4 do artigo só no M1
    m2 = tabela_tempo_inferencia_m2(df)
    assert list(m2["modo"]) == ["M2"] and m2.loc[0, "n_ensaios"] == 2
    assert m2.loc[0, "n_excluido_erro"] == 1
    assert m2.loc[0, "razao_inferencia_ajuste_media"] == pytest.approx((8 / 2 + 20 / 10) / 2)
    assert "razao_artigo" not in m2
