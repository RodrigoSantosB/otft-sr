"""Estatística dos ensaios, comum aos modos M1 e M2.

- Agregação por nível x modo x condição x modelo: média, desvio-padrão amostral e IC 95 %
  t de Student (``physics_llm_sr.evaluation.statistics.summarize``; n = 1 -> IC indefinido,
  NaN, nunca 0). Valores NaN/infinitos de uma métrica são descartados só dessa métrica.
- Taxa de recuperação (classes "recupera estrutura" e "recupera estrutura (equivalência
  sympy)") com IC de Wilson.
- Comparação pareada por semente (Wilcoxon de ``physics_llm_sr``: ``paired_comparison``) e
  McNemar exato (binomial bilateral nos pares discordantes) para a recuperação binária,
  ambos sobre o MESMO conjunto de pares.
- Erros de execução (``status = "erro_execucao"``: falha de infraestrutura, não resultado)
  ficam fora de todos os denominadores (Wilson, McNemar, Wilcoxon, médias); toda tabela
  reporta o n efetivo e ``n_excluido_erro``. Não convergência tem ``status = "ok"`` e conta
  como não recuperação.
- Tabela sob esforço equivalente (mesmo número de candidatos avaliados) e tabela
  custo-qualidade (tempo, tokens, custo estimado).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from physics_llm_sr.evaluation.statistics import paired_comparison, summarize

from otft_sr.experimento import CLASSE_SYMPY, STATUS_OK

GRUPOS_PADRAO: tuple[str, ...] = ("nivel", "modo", "condicao", "modelo")
METRICAS_PADRAO: tuple[str, ...] = (
    "rmse_log10", "one_minus_r2", "n_nos", "tree_distance", "tree_distance_algebraic",
    "rmse_log10_validacao", "taxa_validos", "tempo_s", "tempo_inferencia_s",
)
CLASSES_RECUPERACAO: frozenset[str] = frozenset({"recupera estrutura", CLASSE_SYMPY})
CHAVE_PAREAMENTO: tuple[str, ...] = ("nivel", "ruido", "semente")
CONFIANCA = 0.95


def recuperou(classes: pd.Series) -> pd.Series:
    """``True`` para as classes de recuperação de estrutura (distância ou sympy)."""
    return classes.isin(CLASSES_RECUPERACAO)


def intervalo_wilson(k: int, n: int, confianca: float = CONFIANCA) -> tuple[float, float]:
    """IC de Wilson para a proporção k/n (``(nan, nan)`` se n = 0)."""
    if n <= 0:
        return math.nan, math.nan
    z = float(stats.norm.ppf(0.5 + confianca / 2))
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if k == 0 else max(0.0, centro - meia)
    hi = 1.0 if k == n else min(1.0, centro + meia)
    return lo, hi


def mcnemar_exato(a: Sequence[bool], b: Sequence[bool]) -> dict[str, Any]:
    """McNemar exato: binomial bilateral (p = 0,5) sobre os pares discordantes.

    ``n10`` = a sim e b não; ``n01`` = a não e b sim. Sem discordantes, p = 1.
    """
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    if a.shape != b.shape:
        raise ValueError("a e b devem ter o mesmo tamanho (pareados)")
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    p = 1.0 if n10 + n01 == 0 else float(stats.binomtest(n10, n10 + n01, 0.5).pvalue)
    return {"n11": int(np.sum(a & b)), "n10": n10, "n01": n01, "n00": int(np.sum(~a & ~b)),
            "p_valor": p, "metodo": "mcnemar_exato"}


def erro_execucao(df: pd.DataFrame) -> pd.Series:
    """``True`` nas linhas com erro de execução (``status`` presente e diferente de ``ok``);
    linhas sem ``status`` (resultados antigos) não são tratadas como erro."""
    if "status" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["status"].notna() & (df["status"] != STATUS_OK)


def _separar_erros(sub: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    e = erro_execucao(sub)
    return sub[~e], int(e.sum())


def _com_grupos(df: pd.DataFrame, grupos: Sequence[str]) -> pd.DataFrame:
    faltam = [g for g in grupos if g not in df.columns]
    return df.assign(**{g: np.nan for g in faltam}) if faltam else df


def _valores(sub: pd.DataFrame, coluna: str) -> np.ndarray:
    if coluna not in sub.columns:
        return np.array([])
    v = pd.to_numeric(sub[coluna], errors="coerce").to_numpy(dtype=float)
    return v[np.isfinite(v)]


def agregar(
    df: pd.DataFrame,
    grupos: Sequence[str] = GRUPOS_PADRAO,
    metricas: Sequence[str] = METRICAS_PADRAO,
) -> pd.DataFrame:
    """Formato longo: ``grupos + metrica, n, n_excluido_erro, media, desvio, ic95_inf,
    ic95_sup`` (``n`` = valores finitos entre os ensaios sem erro de execução)."""
    df = _com_grupos(df, grupos)
    linhas = []
    for chave, sub in df.groupby(list(grupos), dropna=False, sort=True):
        chave = chave if isinstance(chave, tuple) else (chave,)
        sub, n_err = _separar_erros(sub)
        for m in metricas:
            v = _valores(sub, m)
            reg: dict[str, Any] = dict(zip(grupos, chave, strict=True))
            reg["metrica"] = m
            reg["n_excluido_erro"] = n_err
            if v.size == 0:
                reg.update(n=0, media=np.nan, desvio=np.nan, ic95_inf=np.nan, ic95_sup=np.nan)
            else:
                s = summarize(v)
                reg.update(
                    n=s.n, media=s.mean,
                    desvio=np.nan if s.sd is None else s.sd,
                    ic95_inf=np.nan if s.ci_low is None else s.ci_low,
                    ic95_sup=np.nan if s.ci_high is None else s.ci_high,
                )
            linhas.append(reg)
    return pd.DataFrame(linhas)


def taxa_recuperacao(
    df: pd.DataFrame, grupos: Sequence[str] = GRUPOS_PADRAO, coluna_classe: str = "classe"
) -> pd.DataFrame:
    """``grupos + n, n_excluido_erro, n_recuperados, taxa, ic95_inf, ic95_sup`` (IC de
    Wilson; ``n`` = ensaios sem erro de execução)."""
    df = _com_grupos(df, grupos)
    linhas = []
    for chave, sub in df.groupby(list(grupos), dropna=False, sort=True):
        chave = chave if isinstance(chave, tuple) else (chave,)
        sub, n_err = _separar_erros(sub)
        n = len(sub)
        k = int(recuperou(sub[coluna_classe]).sum()) if coluna_classe in sub else 0
        lo, hi = intervalo_wilson(k, n)
        linhas.append({**dict(zip(grupos, chave, strict=True)), "n": n,
                       "n_excluido_erro": n_err, "n_recuperados": k,
                       "taxa": k / n if n else np.nan, "ic95_inf": lo, "ic95_sup": hi})
    return pd.DataFrame(linhas)


def _filtrar(df: pd.DataFrame, filtro: Mapping[str, Any]) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    for k, v in filtro.items():
        m &= df[k].isna() if (v is None or (isinstance(v, float) and math.isnan(v))) else df[k] == v
    return df[m]


def comparacao_pareada(
    df: pd.DataFrame,
    metrica: str,
    filtro_a: Mapping[str, Any],
    filtro_b: Mapping[str, Any],
    chave: Sequence[str] = CHAVE_PAREAMENTO,
    alternativa: str = "two-sided",
) -> dict[str, Any]:
    """Compara ``a`` (referência) com ``b`` nos ensaios pareados por ``chave`` (nível, ruído,
    semente). Diferença = a - b (positiva = b melhor em métricas de erro). Ensaios sem par
    são ignorados; chave repetida dentro de um lado levanta ``ValueError``.

    Conjunto pareado único: saem os pares com erro de execução em algum lado
    (``n_excluido_erro``) e os pares sem ``metrica`` finita nos dois lados
    (``n_excluido_metrica``); Wilcoxon e ``mcnemar_recuperacao`` (quando a coluna
    ``classe`` existe) usam exatamente os ``n_pares`` restantes.
    """
    chave = [c for c in chave if c in df.columns]
    a, b = _filtrar(df, filtro_a), _filtrar(df, filtro_b)
    for nome, lado in (("a", a), ("b", b)):
        if lado.duplicated(chave).any():
            raise ValueError(f"pareamento ambíguo: chave {chave} repetida no lado {nome}")
    cols = [*chave, metrica] + (["classe"] if "classe" in df.columns else [])
    a = a[cols].assign(_erro=erro_execucao(a))
    b = b[cols].assign(_erro=erro_execucao(b))
    par = a.merge(b, on=chave, suffixes=("_a", "_b"))
    com_erro = par["_erro_a"] | par["_erro_b"]
    va = pd.to_numeric(par[f"{metrica}_a"], errors="coerce")
    vb = pd.to_numeric(par[f"{metrica}_b"], errors="coerce")
    finito = np.isfinite(va) & np.isfinite(vb)
    ok = ~com_erro & finito
    res = paired_comparison(va[ok].to_numpy(), vb[ok].to_numpy(), alternative=alternativa)
    saida: dict[str, Any] = {
        "metrica": metrica, "filtro_a": dict(filtro_a), "filtro_b": dict(filtro_b),
        "n_pares": res.n, "n_excluido_erro": int(com_erro.sum()),
        "n_excluido_metrica": int((~com_erro & ~finito).sum()),
        "n_nao_nulos": res.n_nonzero, "diferenca_media": res.mean_diff,
        "estatistica": res.statistic, "p_valor": res.p_value, "cohens_dz": res.cohens_dz,
        "rank_biserial": res.rank_biserial, "metodo": res.method, "nota": res.note,
    }
    if "classe" in df.columns:
        saida["mcnemar_recuperacao"] = mcnemar_exato(
            recuperou(par.loc[ok, "classe_a"]).to_numpy(), recuperou(par.loc[ok, "classe_b"]).to_numpy())
    return saida


def tabela_esforco_equivalente(
    df: pd.DataFrame, grupos: Sequence[str] = GRUPOS_PADRAO
) -> pd.DataFrame:
    """Qualidade por ``grupos`` e número de candidatos avaliados (só linhas com a contagem;
    as linhas do M1 não a registram e ficam de fora)."""
    if "n_candidatos_avaliados" not in df.columns:
        return pd.DataFrame()
    df = _com_grupos(df[pd.to_numeric(df["n_candidatos_avaliados"], errors="coerce").notna()],
                     grupos)
    linhas = []
    g = [*grupos, "n_candidatos_avaliados"]
    for chave, sub in df.groupby(g, dropna=False, sort=True):
        sub, n_err = _separar_erros(sub)
        n = len(sub)
        k = int(recuperou(sub["classe"]).sum()) if "classe" in sub else 0
        lo, hi = intervalo_wilson(k, n)
        rm = _valores(sub, "rmse_log10")
        linhas.append({
            **dict(zip(g, chave, strict=True)), "n_ensaios": n, "n_excluido_erro": n_err,
            "rmse_log10_media": float(rm.mean()) if rm.size else np.nan,
            "rmse_log10_validacao_media": (lambda v: float(v.mean()) if v.size else np.nan)(
                _valores(sub, "rmse_log10_validacao")),
            "taxa_validos_media": (lambda v: float(v.mean()) if v.size else np.nan)(
                _valores(sub, "taxa_validos")),
            "taxa_recuperacao": k / n if n else np.nan, "recuperacao_ic95_inf": lo,
            "recuperacao_ic95_sup": hi,
        })
    return pd.DataFrame(linhas)


def tabela_custo_qualidade(
    df: pd.DataFrame, grupos: Sequence[str] = GRUPOS_PADRAO
) -> pd.DataFrame:
    """Tempo, tokens e custo estimado contra a qualidade (RMSE de teste e recuperação), só
    com ensaios sem erro de execução (``n_excluido_erro`` reportado)."""
    df = _com_grupos(df, grupos)
    linhas = []

    def media(sub: pd.DataFrame, col: str) -> float:
        v = _valores(sub, col)
        return float(v.mean()) if v.size else np.nan

    for chave, sub in df.groupby(list(grupos), dropna=False, sort=True):
        chave = chave if isinstance(chave, tuple) else (chave,)
        sub, n_err = _separar_erros(sub)
        tokens = np.nan
        if {"tokens_entrada", "tokens_saida"} <= set(sub.columns):
            tok = (pd.to_numeric(sub["tokens_entrada"], errors="coerce")
                   + pd.to_numeric(sub["tokens_saida"], errors="coerce"))
            tokens = float(tok.mean()) if tok.notna().any() else np.nan
        custo = (pd.to_numeric(sub["custo_api_estimado"], errors="coerce").sum(min_count=1)
                 if "custo_api_estimado" in sub else np.nan)
        n = len(sub)
        k = int(recuperou(sub["classe"]).sum()) if "classe" in sub else 0
        linhas.append({
            **dict(zip(grupos, chave, strict=True)), "n_ensaios": n, "n_excluido_erro": n_err,
            "tempo_s_media": media(sub, "tempo_s"),
            "tempo_inferencia_s_media": media(sub, "tempo_inferencia_s"),
            "tokens_media": tokens, "custo_api_estimado_total": float(custo),
            "rmse_log10_media": media(sub, "rmse_log10"),
            "taxa_recuperacao": k / n if n else np.nan,
        })
    return pd.DataFrame(linhas)


#: Razão tempo de LLM / tempo de RS informada como referência do artigo (Taskin et al. 2026).
RAZAO_TEMPO_LLM_RS_ARTIGO = 0.4


def tabela_razao_tempo_llm(
    df: pd.DataFrame, grupos: Sequence[str] = ("nivel", "modo", "condicao", "modelo", "algoritmo")
) -> pd.DataFrame:
    """Razão tempo de LLM / tempo de RS (``tempo_llm_s / (tempo_s - tempo_llm_s)``) por grupo,
    só para ensaios com LLM (``tempo_llm_s > 0``), ao lado da referência do artigo (0,4)."""
    if not {"tempo_s", "tempo_llm_s"} <= set(df.columns):
        return pd.DataFrame()
    t = pd.to_numeric(df["tempo_s"], errors="coerce")
    tl = pd.to_numeric(df["tempo_llm_s"], errors="coerce")
    sub = _com_grupos(df.assign(_razao=tl / (t - tl))[(tl > 0) & (t - tl > 0)], grupos)
    linhas = []
    for chave, g in sub.groupby(list(grupos), dropna=False, sort=True):
        chave = chave if isinstance(chave, tuple) else (chave,)
        v = g["_razao"].to_numpy(float)
        linhas.append({
            **dict(zip(grupos, chave, strict=True)), "n_ensaios": len(g),
            "tempo_llm_s_media": float(pd.to_numeric(g["tempo_llm_s"]).mean()),
            "tempo_rs_s_media": float((pd.to_numeric(g["tempo_s"])
                                       - pd.to_numeric(g["tempo_llm_s"])).mean()),
            "razao_tempo_llm_rs_media": float(v.mean()),
            "razao_artigo": RAZAO_TEMPO_LLM_RS_ARTIGO,
        })
    return pd.DataFrame(linhas)
