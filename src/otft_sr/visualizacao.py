"""Figuras Plotly (``graph_objects``) do notebook 02. ``plotly`` é importado só no uso.

Regra comum: nenhuma curva é interpolada ou extrapolada além dos dados. As previsões são
avaliadas nos pontos de teste (curvas reconstruídas = previsões ligadas ponto a ponto dentro
de cada curva de teste); as curvas do OVSED e dos níveis ficam restritas à região do nível ou
à faixa da curva medida.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physics_llm_sr.symbolic.expression import evaluate, parse_infix

from otft_sr import ovsed
from otft_sr.dados import construir_datasets
from otft_sr.gerador import protecao_padrao
from otft_sr.niveis import FEATURES, nivel

#: Curva medida de referência por nível (transferência a V_DS fixo ou saída a V_GS fixo).
CURVA_MEDIDA_POR_NIVEL = {"N1": "org1_50VDS", "N2": "org1_40VGS", "N3": "org1_50VDS",
                          "N4": "org1_2VDS"}


def _go():
    import plotly.graph_objects as go

    return go


def _subplots(**kw: Any):
    from plotly.subplots import make_subplots

    return make_subplots(**kw)


def _rotulo_linha(linha: Mapping[str, Any]) -> str:
    partes = [linha.get(k) for k in ("modo", "condicao", "algoritmo", "modelo")]
    return " / ".join(
        str(p) for p in partes if p is not None and not (isinstance(p, float) and math.isnan(p)))


# ------------------------------------------------------------------------- contexto físico


def figura_contexto_fisico(ids: Sequence[str] = ("N1", "N2", "N3", "N4")):
    """Níveis (na região), OVSED completo e Org1 medida, um painel por nível (I_D em log)."""
    go = _go()
    p = ovsed.carregar_parametros_org1()
    med = {c.nome: c for c in ovsed.carregar_curvas_org1()}
    fig = _subplots(rows=1, cols=len(ids), subplot_titles=[f"{i} — {nivel(i).nome}" for i in ids])
    for j, nid in enumerate(ids, start=1):
        n = nivel(nid)
        c = med[CURVA_MEDIDA_POR_NIVEL[nid]]
        fator = n.fator_uA
        saida = c.tipo == "saida"
        x_med = c.V_DS if saida else c.V_GS
        r = n.regiao
        if saida:
            vg0 = float(c.V_GS[0])
            hi = min(r.V_DS[1], vg0 - (r.V_DS_max_rel_V_GS or 0.0))
            x = np.linspace(r.V_DS[0], hi, 120)
            vg, vd = np.full_like(x, vg0), x
            fixo = f"V_GS = {vg0:g} V"
        else:
            vd0 = float(c.V_DS[0])
            x = np.linspace(*r.V_GS, 120)
            vg, vd = x, np.full_like(x, vd0)
            fixo = f"V_DS = {vd0:g} V"
        dentro = (x_med >= x.min()) & (x_med <= x.max()) & (c.I_D_uA > 0)
        mostra = j == 1
        fig.add_trace(go.Scatter(x=x_med[dentro], y=c.I_D_uA[dentro] * fator, mode="markers",
                                 name="Org1 medida", legendgroup="med", showlegend=mostra,
                                 marker=dict(color="black", size=6)), row=1, col=j)
        fig.add_trace(go.Scatter(x=x, y=ovsed.corrente_ovsed_uA(vg, vd, p) * fator, mode="lines",
                                 name="OVSED completo", legendgroup="ovsed", showlegend=mostra,
                                 line=dict(color="grey")), row=1, col=j)
        fig.add_trace(go.Scatter(x=x, y=n.gerador(vg, vd), mode="lines", name="nível (referência)",
                                 legendgroup="nivel", showlegend=mostra,
                                 line=dict(color="crimson", width=3)), row=1, col=j)
        fig.update_xaxes(title_text=("V_DS (V)" if saida else "V_GS (V)") + f", {fixo}", row=1, col=j)
        fig.update_yaxes(type="log", title_text=f"I_D ({n.unidades['I_D']})", row=1, col=j)
    # Largura automática (ocupa a área de saída do notebook); legenda horizontal abaixo dos painéis
    # para não roubar largura dos eixos nem cortar o título.
    fig.update_layout(title=dict(text="Níveis × OVSED × Org1 medida (região de cada nível)",
                                 x=0.5, xanchor="center"),
                      autosize=True, height=480, margin=dict(l=70, r=30, t=90, b=110),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.22,
                                  yanchor="top"))
    return fig


# ------------------------------------------------------------------------------ previsões


def previsoes_teste(linha: Mapping[str, Any], n_curvas: int = 20, n_pontos: int = 50) -> pd.DataFrame:
    """Pontos de TESTE do ensaio com I_D verdadeiro, observado e previsto (em unidades de I_D).

    Reconstrói os dados (determinísticos por nível, ruído e semente). Na formulação ``log10``
    a expressão prevê log10(I_D) e a previsão é ``10^f``. Proteção numérica e corte do
    expoente iguais aos de ``experimento.metricas_expressao``. Linha sem expressão ou com
    erro de execução: DataFrame vazio.
    """
    expr = linha.get("expressao")
    if not isinstance(expr, str) or linha.get("status") != "ok":
        return pd.DataFrame()
    n = nivel(str(linha["nivel"]))
    ds = construir_datasets(n, float(linha.get("ruido", 0.0)), int(linha.get("semente", 0)),
                            n_curvas, n_pontos)["teste"]
    yhat = evaluate(parse_infix(expr, FEATURES), ds.X, FEATURES, protecao_padrao())
    if linha.get("formulacao_perda") == "log10":
        yhat = 10.0 ** np.clip(yhat, -300, 300)
    return pd.DataFrame({
        "V_GS": ds.X[:, 0], "V_DS": ds.X[:, 1], "curva": ds.simulation_ids,
        "I_D_verdadeiro": ds.y_true, "I_D_observado": ds.y_observed, "I_D_previsto": yhat,
        "rotulo": _rotulo_linha(linha), "nivel": n.id, "unidade": n.unidades["I_D"],
    })


def _concat(prev: Mapping[str, pd.DataFrame] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(prev, pd.DataFrame):
        return prev
    return pd.concat([d.assign(rotulo=k) for k, d in prev.items() if len(d)], ignore_index=True) \
        if any(len(d) for d in prev.values()) else pd.DataFrame()


def figura_observado_previsto(prev: Mapping[str, pd.DataFrame] | pd.DataFrame):
    """I_D previsto × I_D observado (teste), log-log, com a diagonal y = x."""
    go = _go()
    df = _concat(prev)
    fig = go.Figure()
    if df.empty:
        return fig
    for rot, g in df.groupby("rotulo", sort=True):
        fig.add_trace(go.Scatter(x=g["I_D_observado"], y=np.maximum(g["I_D_previsto"], 1e-300),
                                 mode="markers", name=str(rot), marker=dict(size=5, opacity=0.7)))
    pos = df["I_D_observado"][df["I_D_observado"] > 0]
    lo, hi = float(pos.min()), float(pos.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
                             line=dict(color="black", dash="dash")))
    u = df["unidade"].iloc[0]
    fig.update_layout(title="Observado × previsto (teste)",
                      xaxis=dict(type="log", title=f"I_D observado ({u})"),
                      yaxis=dict(type="log", title=f"I_D previsto ({u})"), height=480)
    return fig


def figura_curvas_reconstruidas(prev: pd.DataFrame, max_curvas: int = 4):
    """Curvas de teste: dados (pontos) e previsão nos mesmos pontos (linhas), por curva."""
    go = _go()
    fig = go.Figure()
    if prev.empty:
        return fig
    transf = prev.groupby("curva")["V_DS"].nunique().max() == 1
    x = "V_GS" if transf else "V_DS"
    fixo = "V_DS" if transf else "V_GS"
    for i, (cid, g) in enumerate(prev.groupby("curva", sort=True)):
        if i >= max_curvas:
            break
        g = g.sort_values(x)
        nome = f"{fixo} = {g[fixo].iloc[0]:.3g} V"
        fig.add_trace(go.Scatter(x=g[x], y=g["I_D_observado"], mode="markers", name=f"dados, {nome}",
                                 legendgroup=str(cid), marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=g[x], y=np.maximum(g["I_D_previsto"], 1e-300), mode="lines",
                                 name=f"previsto, {nome}", legendgroup=str(cid)))
    fig.update_layout(
        title=f"Curvas reconstruídas (teste, sem extrapolação): {prev['rotulo'].iloc[0]}",
        xaxis_title=f"{x} (V)", yaxis=dict(type="log", title=f"I_D ({prev['unidade'].iloc[0]})"),
        height=460)
    return fig


def figura_residuos(prev: Mapping[str, pd.DataFrame] | pd.DataFrame):
    """Resíduo em década, log10(I_D previsto / I_D verdadeiro), contra V_GS e V_DS."""
    go = _go()
    df = _concat(prev)
    fig = _subplots(rows=1, cols=2, subplot_titles=["resíduo × V_GS", "resíduo × V_DS"])
    if df.empty:
        return fig
    res = np.log10(np.maximum(df["I_D_previsto"], 1e-300) / df["I_D_verdadeiro"])
    for rot, g in df.assign(res=res).groupby("rotulo", sort=True):
        for col, v in ((1, "V_GS"), (2, "V_DS")):
            fig.add_trace(go.Scatter(x=g[v], y=g["res"], mode="markers", name=str(rot),
                                     legendgroup=str(rot), showlegend=col == 1,
                                     marker=dict(size=5, opacity=0.7)), row=1, col=col)
            fig.update_xaxes(title_text=f"{v} (V)", row=1, col=col)
    fig.update_yaxes(title_text="log10(previsto / verdadeiro) (década)", row=1, col=1)
    fig.update_layout(title="Resíduos no teste", height=440)
    return fig


# ------------------------------------------------------------------------------ resumos


def _grupo(df: pd.DataFrame) -> pd.Series:
    return df.apply(_rotulo_linha, axis=1)


def figura_pareto(df: pd.DataFrame):
    """Erro (RMSE log10 no teste) × complexidade (nós), por modo/condição/modelo."""
    go = _go()
    ok = df[df["status"] == "ok"].copy()
    fig = go.Figure()
    for rot, g in ok.assign(grupo=_grupo(ok) if len(ok) else []).groupby("grupo", sort=True):
        fig.add_trace(go.Scatter(x=g["n_nos"], y=g["rmse_log10"], mode="markers", name=str(rot),
                                 text=g["nivel"], marker=dict(size=9, opacity=0.8)))
    fig.update_layout(title="Erro × complexidade", xaxis_title="complexidade (nós)",
                      yaxis=dict(type="log", title="RMSE de log10 I_D no teste (década)"), height=460)
    return fig


def figura_tempo_qualidade(df: pd.DataFrame):
    """Tempo do ensaio × RMSE log10 no teste."""
    go = _go()
    ok = df[df["status"] == "ok"].copy()
    fig = go.Figure()
    for rot, g in ok.assign(grupo=_grupo(ok) if len(ok) else []).groupby("grupo", sort=True):
        fig.add_trace(go.Scatter(x=g["tempo_s"], y=g["rmse_log10"], mode="markers", name=str(rot),
                                 text=g["nivel"], marker=dict(size=9, opacity=0.8)))
    fig.update_layout(title="Tempo × qualidade", xaxis=dict(type="log", title="tempo do ensaio (s)"),
                      yaxis=dict(type="log", title="RMSE de log10 I_D no teste (década)"), height=460)
    return fig


def figura_evolucao_refinamento(candidatos: pd.DataFrame):
    """G2: RMSE log10 de VALIDAÇÃO por rodada de refinamento, uma linha por linhagem."""
    go = _go()
    fig = go.Figure()
    if candidatos.empty or "rodada" not in candidatos:
        return fig
    c = candidatos.copy()
    c["rmse_log10_validacao"] = pd.to_numeric(c.get("rmse_log10_validacao"), errors="coerce")
    for (ch, lin), g in c.groupby(["chave_ensaio", "linhagem"], sort=True):
        g = g.sort_values("rodada")
        if g["rodada"].nunique() < 2:
            continue
        nome = f"{_curta(ch)} · linhagem {int(lin)}"
        fig.add_trace(go.Scatter(x=g["rodada"], y=g["rmse_log10_validacao"], mode="lines+markers",
                                 name=nome))
    fig.update_layout(title="G2: evolução do erro de validação no refinamento",
                      xaxis=dict(title="rodada", dtick=1),
                      yaxis=dict(type="log", title="RMSE de log10 I_D na validação (década)"), height=440)
    return fig


def _curta(chave: str) -> str:
    campos = dict(p.split("=", 1) for p in str(chave).split("|") if "=" in p)
    return " ".join(campos.get(k, "") for k in ("nivel", "condicao", "modelo", "semente"))


def figura_comparacao_llms(df: pd.DataFrame, metrica: str = "rmse_log10"):
    """Distribuição de ``metrica`` por modelo, separada por modo/condição (caixas + pontos)."""
    go = _go()
    ok = df[(df["status"] == "ok") & df["modelo"].notna()].copy()
    fig = go.Figure()
    for (modo, cond), g in ok.groupby(["modo", "condicao"], sort=True):
        fig.add_trace(go.Box(x=g["modelo"], y=pd.to_numeric(g[metrica], errors="coerce"),
                             name=f"{modo} {cond}", boxpoints="all", jitter=0.3))
    fig.update_layout(title=f"Comparação entre LLMs: {metrica}", boxmode="group",
                      yaxis=dict(type="log", title=metrica), height=460)
    return fig


def figura_barras_ic(taxa: pd.DataFrame, x: str = "condicao", cor: str = "modelo"):
    """Taxa de recuperação com IC de Wilson (``estatistica.taxa_recuperacao``)."""
    go = _go()
    fig = go.Figure()
    if taxa.empty:
        return fig
    t = taxa.copy()
    t[cor] = t[cor].fillna("nenhum")
    t["_x"] = t["nivel"].astype(str) + " " + t[x].astype(str)
    for rot, g in t.groupby(cor, sort=True):
        fig.add_trace(go.Bar(x=g["_x"], y=g["taxa"], name=str(rot),
                             error_y=dict(type="data", symmetric=False,
                                          array=g["ic95_sup"] - g["taxa"],
                                          arrayminus=g["taxa"] - g["ic95_inf"])))
    fig.update_layout(title="Recuperação simbólica (IC 95 % de Wilson)", barmode="group",
                      yaxis=dict(range=[0, 1.05], title="taxa de recuperação"), height=440)
    return fig


def salvar_figura(fig: Any, caminho_base: Any) -> list[str]:
    """Grava ``<base>.html`` e ``<base>.json`` (figura serializada pelo plotly)."""
    base = Path(caminho_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(base.with_suffix(".html")), include_plotlyjs="cdn")
    base.with_suffix(".json").write_text(fig.to_json(), "utf-8")
    return [str(base.with_suffix(".html")), str(base.with_suffix(".json"))]

