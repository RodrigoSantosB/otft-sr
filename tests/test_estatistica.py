"""Estatística comum a M1/M2: agregação, Wilson, comparação pareada e McNemar exato."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from physics_llm_sr.evaluation.statistics import paired_comparison, summarize

from otft_sr.estatistica import (
    GRUPOS_PADRAO,
    agregar,
    comparacao_pareada,
    intervalo_wilson,
    mcnemar_exato,
    recuperou,
    tabela_custo_qualidade,
    tabela_esforco_equivalente,
    taxa_recuperacao,
)
from otft_sr.experimento import CLASSE_SYMPY


def _df():
    linhas = []
    for cond, deslocamento, classes in (
        ("G1", 0.0, ["recupera estrutura", CLASSE_SYMPY, "não converge", "recupera estrutura"]),
        ("G0", 0.1, ["não converge", "não converge", "recupera estrutura", "não converge"]),
    ):
        for s in range(4):
            linhas.append({
                "nivel": "N3", "modo": "M2", "condicao": cond, "modelo": "ollama/m",
                "algoritmo": "llm_gerador", "ruido": 0.0, "semente": s,
                "rmse_log10": 0.01 * (s + 1) + deslocamento, "n_nos": 7 + s,
                "classe": classes[s], "status": "ok", "n_candidatos_avaliados": 6,
                "tempo_s": 10.0 + s, "tempo_inferencia_s": 8.0 + s,
                "tokens_entrada": np.nan, "tokens_saida": np.nan, "custo_api_estimado": np.nan,
            })
    linhas.append({"nivel": "N3", "modo": "M1", "condicao": "A", "modelo": np.nan,
                   "algoritmo": "pysr", "ruido": 0.0, "semente": 0, "rmse_log10": 0.5,
                   "n_nos": 9, "classe": "não converge", "status": "ok", "tempo_s": 60.0})
    return pd.DataFrame(linhas)


def test_wilson_valores_conhecidos():
    lo, hi = intervalo_wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.27753, abs=1e-5)
    lo, hi = intervalo_wilson(5, 10)
    assert lo == pytest.approx(0.23659, abs=1e-5) and hi == pytest.approx(0.76341, abs=1e-5)
    assert all(math.isnan(x) for x in intervalo_wilson(0, 0))


def test_mcnemar_exato():
    a = [True] * 6 + [False] * 4
    b = [False] * 10
    r = mcnemar_exato(a, b)
    assert (r["n10"], r["n01"]) == (6, 0)
    assert r["p_valor"] == pytest.approx(0.03125)
    assert mcnemar_exato([True, False], [True, False])["p_valor"] == 1.0
    with pytest.raises(ValueError):
        mcnemar_exato([True], [True, False])


def test_recuperou_aceita_classe_por_distancia_e_sympy():
    assert list(recuperou(pd.Series(["recupera estrutura", CLASSE_SYMPY, "não converge"]))) == [
        True, True, False]


def test_agregar_igual_ao_summarize_e_ignora_nan():
    df = _df()
    ag = agregar(df, metricas=("rmse_log10", "tokens_entrada"))
    assert tuple(GRUPOS_PADRAO) == ("nivel", "modo", "condicao", "modelo")
    g1 = ag[(ag["condicao"] == "G1") & (ag["metrica"] == "rmse_log10")].iloc[0]
    ref = summarize(df[df["condicao"] == "G1"]["rmse_log10"].to_numpy())
    assert g1["n"] == 4 and g1["media"] == pytest.approx(ref.mean)
    assert g1["desvio"] == pytest.approx(ref.sd)
    assert (g1["ic95_inf"], g1["ic95_sup"]) == pytest.approx((ref.ci_low, ref.ci_high))
    tok = ag[(ag["condicao"] == "G1") & (ag["metrica"] == "tokens_entrada")].iloc[0]
    assert tok["n"] == 0 and math.isnan(tok["media"])
    a = ag[(ag["condicao"] == "A") & (ag["metrica"] == "rmse_log10")].iloc[0]
    assert a["n"] == 1 and math.isnan(a["ic95_inf"])  # n = 1: IC indefinido, nunca 0
    assert pd.isna(a["modelo"])


def test_taxa_recuperacao_com_wilson():
    t = taxa_recuperacao(_df())
    g1 = t[t["condicao"] == "G1"].iloc[0]
    assert (g1["n_recuperados"], g1["n"]) == (3, 4) and g1["taxa"] == 0.75
    assert (g1["ic95_inf"], g1["ic95_sup"]) == pytest.approx(intervalo_wilson(3, 4))


def test_comparacao_pareada_por_semente():
    df = _df()
    df = pd.concat([df, df.iloc[[0]].assign(condicao="G0", semente=99)])  # sem par: ignorada
    r = comparacao_pareada(df, "rmse_log10", {"condicao": "G0"}, {"condicao": "G1"})
    base = df[(df["condicao"] == "G0") & (df["semente"] < 99)].sort_values("semente")
    llm = df[df["condicao"] == "G1"].sort_values("semente")
    ref = paired_comparison(base["rmse_log10"].to_numpy(), llm["rmse_log10"].to_numpy())
    assert r["n_pares"] == 4 and r["p_valor"] == ref.p_value and r["metodo"] == ref.method
    assert r["diferenca_media"] == pytest.approx(0.1)
    mc = r["mcnemar_recuperacao"]
    assert (mc["n10"], mc["n01"]) == (1, 3)  # G0 recupera só onde G1 falha (semente 2)


def test_tabela_esforco_equivalente_so_com_contagem():
    t = tabela_esforco_equivalente(_df())
    assert set(t["condicao"]) == {"G0", "G1"}  # M1 sem contagem de candidatos fica de fora
    assert set(t["n_candidatos_avaliados"]) == {6}
    g1 = t[t["condicao"] == "G1"].iloc[0]
    assert g1["taxa_recuperacao"] == 0.75 and g1["rmse_log10_media"] == pytest.approx(0.025)


def test_tabela_custo_qualidade():
    t = tabela_custo_qualidade(_df())
    g1 = t[t["condicao"] == "G1"].iloc[0]
    assert g1["tempo_s_media"] == pytest.approx(11.5)
    assert g1["tempo_inferencia_s_media"] == pytest.approx(9.5)
    assert math.isnan(g1["tokens_media"]) and math.isnan(g1["custo_api_estimado_total"])
    assert g1["taxa_recuperacao"] == 0.75


# ------------------------------------------------- exclusão de erros de execução


def _com_erros():
    """G0/G1 com 4 sementes; G1 semente 3 e G0 semente 0 com erro de execução; G0 semente 1
    ok, mas sem métrica (não converge sem candidato)."""
    df = _df()
    df = df[df["modo"] == "M2"].copy()
    i_g1 = df.index[(df["condicao"] == "G1") & (df["semente"] == 3)]
    i_g0 = df.index[(df["condicao"] == "G0") & (df["semente"] == 0)]
    df.loc[i_g1, ["status", "classe", "rmse_log10"]] = ["erro_execucao", "erro de execução", np.nan]
    df.loc[i_g0, ["status", "classe", "rmse_log10"]] = ["erro_execucao", "erro de execução", np.nan]
    i_nc = df.index[(df["condicao"] == "G0") & (df["semente"] == 1)]
    df.loc[i_nc, ["classe", "rmse_log10"]] = ["não converge", np.nan]
    return df


def test_taxa_exclui_erro_de_execucao_do_denominador():
    t = taxa_recuperacao(_com_erros())
    g1 = t[t["condicao"] == "G1"].iloc[0]
    assert (g1["n"], g1["n_excluido_erro"], g1["n_recuperados"]) == (3, 1, 2)
    assert g1["taxa"] == pytest.approx(2 / 3)
    assert (g1["ic95_inf"], g1["ic95_sup"]) == pytest.approx(intervalo_wilson(2, 3))


def test_pareamento_unico_para_wilcoxon_e_mcnemar():
    r = comparacao_pareada(_com_erros(), "rmse_log10", {"condicao": "G0"}, {"condicao": "G1"})
    # sementes 0 (erro em G0) e 3 (erro em G1) saem; semente 1 sai por métrica ausente
    assert r["n_excluido_erro"] == 2 and r["n_excluido_metrica"] == 1
    assert r["n_pares"] == 1
    mc = r["mcnemar_recuperacao"]
    assert mc["n11"] + mc["n10"] + mc["n01"] + mc["n00"] == r["n_pares"]


def test_agregacao_e_tabelas_reportam_excluidos():
    df = _com_erros()
    ag = agregar(df, metricas=("rmse_log10", "tempo_s"))
    g1 = ag[(ag["condicao"] == "G1") & (ag["metrica"] == "tempo_s")].iloc[0]
    assert g1["n"] == 3 and g1["n_excluido_erro"] == 1  # tempo do ensaio com erro fica fora
    for t in (tabela_esforco_equivalente(df), tabela_custo_qualidade(df)):
        g1 = t[t["condicao"] == "G1"].iloc[0]
        assert g1["n_ensaios"] == 3 and g1["n_excluido_erro"] == 1
        assert g1["taxa_recuperacao"] == pytest.approx(2 / 3)


def test_comparacao_pareada_sem_pares_validos_nao_falha():
    """M2 sem candidato válido em um lado (rmse_log10 NaN em todos os pares): sem teste
    estatístico, n_pares = 0 e nota explicativa — nunca exceção nem p-valor fictício."""
    df = _df()
    df.loc[df["condicao"] == "G0", "rmse_log10"] = np.nan
    r = comparacao_pareada(df, "rmse_log10", {"condicao": "G0"}, {"condicao": "G1"})
    assert r["n_pares"] == 0 and r["n_excluido_metrica"] == 4
    assert r["p_valor"] is None and math.isnan(r["diferenca_media"])
    assert "sem pares" in r["nota"]
    assert math.isnan(r["mcnemar_recuperacao"]["p_valor"])


def test_comparacao_pareada_sem_ensaios_pareaveis_nao_falha():
    df = _df()
    df.loc[df["condicao"] == "G1", "semente"] += 100  # nenhuma semente em comum
    r = comparacao_pareada(df, "rmse_log10", {"condicao": "G0"}, {"condicao": "G1"})
    assert r["n_pares"] == 0 and r["p_valor"] is None
