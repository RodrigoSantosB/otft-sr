"""Auditoria do avaliador M1: o LLM discrimina a equação-alvo de formas erradas?

Pré-requisito para interpretar as condições B/C: se o escore ``c`` (penalidade
``1 - (c1 + c2 + c3)/3``) da referência não é menor que o de formas erradas plausíveis, o
termo ``w3 c`` não informa a busca sobre a forma funcional correta. Conjunto por nível
(mesma formulação da perda que a busca usará):

- ``referencia``: a expressão de referência do nível;
- ``equivalente``: a primeira forma equivalente declarada (mesma física, outra escrita);
- ``errada_plausivel``: formas de OTFT fisicamente plausíveis, mas erradas para o nível
  (expoente errado, regime errado, termo faltante), com as constantes do nível;
- ``constante``: I_D constante (mediana da referência na região);
- ``absurda``: sem sentido físico.

As chamadas não usam cache nem retentativas (``llm_inspection.probe_expressions``): mostram
exatamente o que o modelo respondeu.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from physics_llm_sr.analysis.llm_inspection import probe_expressions
from physics_llm_sr.llm.scoring import LLMEquationEvaluator

from otft_sr.contexto import VARIANTE_PROMPT, contexto_fisico
from otft_sr.formulacao import INV_LN10, log10_expandido, nivel_formulado
from otft_sr.niveis import FEATURES, NivelEquacao, _grade_regiao

CATEGORIAS_AUDITORIA: tuple[str, ...] = (
    "referencia", "equivalente", "errada_plausivel", "constante", "absurda")
ABSURDA = "exp(V_GS * V_DS) + sin(V_DS) * log(V_GS)"


def _erradas(nivel: NivelEquacao) -> list[tuple[str, str]]:
    c = nivel.constantes
    if nivel.id == "N1":
        K, V = c["K"], c["V_Th"]
        return [
            ("linear em V_GS", f"{K!r} * (V_GS - {V!r})"),
            ("cúbica em V_GS", f"{K!r} * (V_GS - {V!r}) ** 3"),
            ("forma de triodo", f"{K!r} * (2 * (V_GS - {V!r}) * V_DS - V_DS ** 2)"),
            ("exponencial", f"{K!r} * exp(0.1 * V_GS)"),
        ]
    if nivel.id == "N2":
        K, V = c["K"], c["V_Th"]
        return [
            ("saturação (sem V_DS)", f"{K!r} * (V_GS - {V!r}) ** 2"),
            ("triodo sem o termo V_DS²", f"{2 * K!r} * (V_GS - {V!r}) * V_DS"),
            ("quadrática só em V_DS", f"{K!r} * V_DS ** 2"),
        ]
    if nivel.id == "N3":
        K, V, ell = c["K"], c["V_th0"], c["l"]
        return [
            ("lei quadrática", f"{K!r} * (V_GS - {V!r}) ** 2"),
            ("lei de potência com V_DS", f"{K!r} * (V_GS - {V!r}) ** {ell!r} * V_DS"),
            ("exponencial", f"{K!r} * exp(0.3 * V_GS)"),
        ]
    if nivel.id == "N4":
        I_leak, I_0, b = c["I_leak"], c["I_0"], c["ln10_sobre_S"]
        return [
            ("exponencial sem fuga", f"{I_0!r} * exp({b!r} * V_GS)"),
            ("fuga + quadrática", f"{I_leak!r} + {I_0!r} * V_GS ** 2"),
            ("fuga + exponencial em V_DS", f"{I_leak!r} + {I_0!r} * exp({b!r} * V_DS)"),
        ]
    raise KeyError(f"nível sem formas erradas definidas: {nivel.id}")


def expressoes_auditoria(nivel: NivelEquacao, formulacao: str = "linear") -> list[dict[str, str]]:
    """Expressões da auditoria (``rotulo``, ``categoria``, ``expressao``) na formulação dada."""
    vg, vd = _grade_regiao(nivel.regiao)
    mediana = float(np.median(nivel.gerador(vg, vd)))
    itens = [("referência", "referencia", nivel.expressao)]
    if nivel.formas_equivalentes:
        itens.append(("forma equivalente", "equivalente", nivel.formas_equivalentes[0]))
    itens += [(r, "errada_plausivel", e) for r, e in _erradas(nivel)]
    itens.append(("constante", "constante", f"{float(f'{mediana:.4g}')!r}"))
    if formulacao == "log10":
        nl = nivel_formulado(nivel, "log10")
        conv = {nivel.expressao: nl.expressao}
        if nivel.formas_equivalentes:
            # forma equivalente em log10: a expandida, se distinta; senão log10 sem expandir
            conv[nivel.formas_equivalentes[0]] = (
                nl.formas_equivalentes[0] if nl.formas_equivalentes
                else f"log({nivel.expressao}) * {INV_LN10!r}")
        itens = [(r, k, conv.get(e) or log10_expandido(e)) for r, k, e in itens]
    else:
        nivel_formulado(nivel, formulacao)  # valida o nome da formulação
    itens.append(("absurda", "absurda", ABSURDA))
    vistos: set[str] = set()
    saida = []
    for r, k, e in itens:
        if e not in vistos:
            vistos.add(e)
            saida.append({"rotulo": r, "categoria": k, "expressao": e})
    return saida


def auditar_avaliador(
    avaliador: LLMEquationEvaluator, nivel: NivelEquacao, formulacao: str = "linear"
) -> pd.DataFrame:
    """Uma chamada por expressão; colunas ``rotulo, categoria, expressao, valida, c1, c2, c3,
    c, feedback, erro, resposta_crua``. ``c`` é a penalidade que entraria na perda."""
    itens = expressoes_auditoria(nivel, formulacao)
    ctx = contexto_fisico(nivel_formulado(nivel, formulacao))
    df = probe_expressions(avaliador, [i["expressao"] for i in itens], ctx, VARIANTE_PROMPT,
                           FEATURES, labels=[i["rotulo"] for i in itens])
    df = df.rename(columns={"expression": "expressao", "penalty": "c"})
    df.insert(1, "categoria", [i["categoria"] for i in itens])
    df.insert(0, "nivel", nivel.id)
    df.insert(1, "formulacao_perda", formulacao)
    return df[["nivel", "formulacao_perda", "rotulo", "categoria", "expressao", "valida", "c1",
               "c2", "c3", "c", "feedback", "erro", "resposta_crua"]]


def resumo_discriminacao(df: pd.DataFrame) -> dict[str, Any]:
    """Resumo da auditoria de UM nível/modelo/condição.

    ``discrimina_erradas``: c(referência) < c de TODAS as formas erradas plausíveis válidas
    (empate conta como falha). ``margem_erradas`` = min c(errada) - c(referência).
    """
    v = df[df["valida"]]

    def cs(cat: str) -> np.ndarray:
        return v.loc[v["categoria"] == cat, "c"].to_numpy(float)

    ref, err, absu, const = cs("referencia"), cs("errada_plausivel"), cs("absurda"), cs("constante")
    c_ref = float(ref[0]) if ref.size else np.nan
    ok = ref.size > 0
    return {
        "n_expressoes": int(len(df)),
        "n_invalidas": int((~df["valida"]).sum()),
        "c_referencia": c_ref,
        "c_equivalente": float(cs("equivalente")[0]) if cs("equivalente").size else np.nan,
        "c_errada_min": float(err.min()) if err.size else np.nan,
        "c_errada_media": float(err.mean()) if err.size else np.nan,
        "c_constante": float(const[0]) if const.size else np.nan,
        "c_absurda": float(absu[0]) if absu.size else np.nan,
        "n_erradas_com_c_menor_ou_igual_a_referencia": int((err <= c_ref).sum()) if ok else 0,
        "margem_erradas": float(err.min() - c_ref) if (ok and err.size) else np.nan,
        "discrimina_erradas": bool(ok and err.size > 0 and bool(np.all(err > c_ref))),
        "discrimina_absurda": bool(ok and absu.size > 0 and bool(absu[0] > c_ref)),
    }
