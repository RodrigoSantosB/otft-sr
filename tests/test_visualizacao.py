"""Figuras Plotly do notebook 02 (puladas se plotly não estiver instalado)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from otft_sr import visualizacao as vz  # noqa: E402
from otft_sr.estatistica import taxa_recuperacao  # noqa: E402
from otft_sr.niveis import nivel  # noqa: E402


def _linha(nid="N3", formulacao="linear", **kw):
    n = nivel(nid)
    expr = n.expressao
    if formulacao == "log10":
        from otft_sr.formulacao import nivel_formulado

        expr = nivel_formulado(n, "log10").expressao
    return {"nivel": nid, "ruido": 0.0, "semente": 0, "status": "ok", "expressao": expr,
            "formulacao_perda": formulacao, "modo": "M1", "condicao": "A", "algoritmo": "deap",
            "modelo": "nenhum", **kw}


def test_figura_contexto_fisico():
    fig = vz.figura_contexto_fisico()
    assert len(fig.data) == 12


@pytest.mark.parametrize("formulacao", ["linear", "log10"])
def test_previsoes_da_referencia_coincidem_com_o_verdadeiro(formulacao):
    prev = vz.previsoes_teste(_linha(formulacao=formulacao), n_curvas=10, n_pontos=10)
    assert len(prev) > 0
    assert np.allclose(prev["I_D_previsto"], prev["I_D_verdadeiro"], rtol=1e-8)


def test_previsoes_usam_a_protecao_numerica_das_metricas(monkeypatch):
    """Mesma ProtectionParams de metricas_expressao (configuração do M1), não o padrão."""
    import dataclasses

    from otft_sr.gerador import protecao_padrao

    linha = _linha("N1", expressao="log(V_GS - 49.9)")
    base = vz.previsoes_teste(linha, 10, 10)["I_D_previsto"].to_numpy()
    outra = dataclasses.replace(protecao_padrao(), log_epsilon=10.0)
    monkeypatch.setattr(vz, "protecao_padrao", lambda: outra)
    mudou = vz.previsoes_teste(linha, 10, 10)["I_D_previsto"].to_numpy()
    assert not np.allclose(base, mudou)


def test_previsoes_linha_com_erro_vazia():
    assert vz.previsoes_teste({"status": "erro", "expressao": float("nan")}).empty


def test_figuras_de_previsao(tmp_path):
    prev = {"ref": vz.previsoes_teste(_linha(), 10, 10),
            "N2": vz.previsoes_teste(_linha("N2"), 10, 10)}
    for fig in (vz.figura_observado_previsto(prev["ref"]),
                vz.figura_curvas_reconstruidas(prev["N2"]),
                vz.figura_residuos(prev)):
        assert len(fig.data) > 0
    arquivos = vz.salvar_figura(vz.figura_residuos(prev), tmp_path / "f" / "residuos")
    assert all(p.endswith((".html", ".json")) for p in arquivos)
    assert (tmp_path / "f" / "residuos.html").is_file()


def test_figuras_de_resumo():
    df = pd.DataFrame([
        _linha(n_nos=7, rmse_log10=0.01, tempo_s=2.0, classe="recupera estrutura"),
        _linha(condicao="B", modelo="fake/x", n_nos=9, rmse_log10=0.1, tempo_s=5.0, classe="não"),
        _linha(modo="M2", condicao="G1", algoritmo="llm_gerador", modelo="fake/x", n_nos=5,
               rmse_log10=0.2, tempo_s=1.0, classe="não"),
    ])
    for fig in (vz.figura_pareto(df), vz.figura_tempo_qualidade(df), vz.figura_comparacao_llms(df),
                vz.figura_barras_ic(taxa_recuperacao(df))):
        assert len(fig.data) > 0
    cand = pd.DataFrame({"chave_ensaio": ["nivel=N3|condicao=G2"] * 4, "linhagem": [0, 0, 1, 1],
                         "rodada": [0, 1, 0, 1], "rmse_log10_validacao": [0.5, 0.2, 0.4, 0.4]})
    assert len(vz.figura_evolucao_refinamento(cand).data) == 2


@pytest.mark.parametrize("ids", [("N3",), ("N1", "N2", "N3", "N4")])
def test_figura_contexto_fisico_ocupa_a_largura_disponivel(ids):
    """Com um único nível a largura fixa (380 px por painel) encolhia a figura e cortava o
    título; a figura deve se ajustar à largura da área de saída e manter a legenda fora dos painéis."""
    from otft_sr.visualizacao import figura_contexto_fisico

    lay = figura_contexto_fisico(ids).layout
    assert lay.width is None and lay.autosize is not False
    assert lay.legend.orientation == "h"
    assert lay.height >= 420
