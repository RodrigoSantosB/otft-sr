"""Formulação da perda do modo M1: ``linear`` (artigo) ou ``log10`` (H-OTFT-19).

- ``linear``: alvo y = I_D; e = MSE/Var(I_D) (perda normalizada do artigo, H-OTFT-07).
- ``log10``: alvo y = log10(I_D); e = MSE/Var(log10 I_D). A expressão de referência é
  log10(<referência>) escrita com ``log`` natural e expandida (sympy ``expand_log`` com
  ``force=True``, válido na região do nível, onde as bases são positivas), p.ex. N3:
  ``-4,658 + 1,381 log(V_GS - 8,57)``. A expansão é necessária, e não só conveniente: o
  ``log`` protegido do ``physics_llm_sr`` vale 0 para |x| < 1e-3 (GAP-06), e o I_D do N3
  desce a ~2e-4 µA, de modo que ``log(K (V_GS - V_th0)^l)`` avaliaria errado. Nada muda no
  ``physics_llm_sr``: só o alvo dos ``Dataset`` e o nível (expressão, formas equivalentes,
  gerador e unidade) são transformados aqui.

Ruído: o ruído multiplicativo é aplicado em I_D antes do log (os mesmos dados da formulação
linear); valores observados <= 0 (só possíveis com sigma alto) recebem o piso
``PISO_RELATIVO * min(I_D observado > 0)`` antes do log.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

import numpy as np
import sympy as sp

from physics_llm_sr.core_types import Dataset
from physics_llm_sr.symbolic.expression import from_sympy, parse_infix, to_infix, to_sympy

from otft_sr.niveis import FEATURES, NivelEquacao

FORMULACOES: tuple[str, ...] = ("linear", "log10")
INV_LN10 = 1.0 / math.log(10.0)
PISO_RELATIVO = 1e-6


def validar_formulacao(formulacao: str) -> str:
    """Devolve ``formulacao`` ou levanta ``ValueError``."""
    if formulacao not in FORMULACOES:
        raise ValueError(f"formulação da perda deve ser uma de {FORMULACOES}, recebeu {formulacao!r}")
    return formulacao


def log10_expandido(expr: str) -> str:
    """Texto infixo de log10(``expr``) com ``log`` natural, expandido e com constantes numéricas."""
    e = to_sympy(parse_infix(expr, FEATURES))
    lg = sp.expand(sp.expand_log(sp.log(e), force=True) / sp.log(10), deep=False).evalf()
    return to_infix(from_sympy(lg, FEATURES))


def nivel_formulado(nivel: NivelEquacao, formulacao: str = "linear") -> NivelEquacao:
    """Nível na formulação pedida (``linear`` devolve o próprio objeto)."""
    if validar_formulacao(formulacao) == "linear":
        return nivel
    gerador = nivel.gerador
    principal = log10_expandido(nivel.expressao)
    equivalentes = tuple(dict.fromkeys(
        f for f in (log10_expandido(x) for x in nivel.formas_equivalentes) if f != principal))
    return dataclasses.replace(
        nivel,
        expressao=principal,
        formas_equivalentes=equivalentes,
        gerador=lambda vg, vd: np.log10(gerador(vg, vd)),
        unidades={**nivel.unidades, "I_D": f"log10({nivel.unidades['I_D']})"},
    )


def _log10_seguro(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    pos = y[y > 0]
    piso = PISO_RELATIVO * float(np.min(pos)) if pos.size else 1e-300
    return np.log10(np.maximum(y, piso))


def datasets_formulados(
    datasets: Mapping[str, Dataset], formulacao: str = "linear"
) -> Mapping[str, Dataset]:
    """Partições com o alvo da formulação (``linear`` devolve o próprio mapeamento)."""
    if validar_formulacao(formulacao) == "linear":
        return datasets
    return {
        k: dataclasses.replace(
            d,
            y_true=_log10_seguro(d.y_true),
            y_observed=_log10_seguro(d.y_observed),
            target_name=f"log10({d.target_name})",
            metadata={**d.metadata, "formulacao_perda": formulacao},
        )
        for k, d in datasets.items()
    }


def formulacao_do_nivel(nivel: NivelEquacao) -> str:
    """``log10`` se a unidade de I_D do nível é ``log10(...)``, senão ``linear``."""
    return "log10" if nivel.unidades["I_D"].startswith("log10(") else "linear"
