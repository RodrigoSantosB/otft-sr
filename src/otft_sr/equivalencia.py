"""Recuperação da forma funcional por manipulação simbólica (sympy), proposta §6.

Complementa a distância de árvore do ``physics_llm_sr`` (que não reconhece, por exemplo,
``log(a**(b**V)) = log(a) exp(V log b)``). Procedimento:

1. ``c**f(V)`` (base numérica) -> ``exp(f(V) log c)``; ``powsimp``/``expand_log`` (force);
2. coeficiente líder fora de bases polinomiais com expoente real; Floats inteiros viram inteiros; expoentes reais são arredondados a 2 decimais;
3. expansão em soma de termos; cada termo perde o coeficiente numérico e as demais
   constantes internas viram o símbolo ``c``.

A forma funcional é recuperada quando os conjuntos de termos-base coincidem (iguais a menos
dos valores das constantes). A exatidão das constantes é avaliada à parte por 1 - R².
"""

from __future__ import annotations

import sympy as sp

from physics_llm_sr.symbolic.expression import Expr, to_sympy

_C = sp.Symbol("c")


def _monica(x: sp.Pow) -> sp.Expr:
    """(a V + b)^p -> a^p (V + b/a)^p: coeficiente líder fora da base (polinômio)."""
    base, ex = x.base, x.exp
    simbolos = sorted(base.free_symbols, key=str)
    if not simbolos or not base.is_polynomial(*simbolos):
        return x
    lc = sp.Poly(base, *simbolos).LC()
    if not lc.is_number or lc == 0 or lc == 1:
        return x
    return sp.Abs(lc) ** ex * sp.Pow(sp.expand(base / lc), ex, evaluate=False)


def _normalizar(e: sp.Expr) -> sp.Expr:
    e = e.replace(
        lambda x: x.is_Pow and not x.exp.is_number and x.base.is_number,
        lambda x: sp.exp(x.exp * sp.log(x.base)),
    )
    e = sp.expand_log(sp.powsimp(e, force=True), force=True)
    e = e.replace(lambda x: x.is_Pow and x.exp.is_number and not x.exp.is_Integer, _monica)
    e = e.xreplace(
        {f: sp.Integer(int(round(float(f)))) for f in e.atoms(sp.Float) if abs(f - round(f)) < 1e-9}
    )
    e = e.replace(
        lambda x: x.is_Pow and x.exp.is_Float,
        lambda x: sp.Pow(x.base, sp.Rational(round(float(x.exp) * 100), 100)),
    )
    return sp.expand(e, power_exp=True)


def termos_base(expr: Expr) -> frozenset[str]:
    """Conjunto de termos-base (sem constantes numéricas) da expressão."""
    e = _normalizar(to_sympy(expr))
    base = set()
    for termo in sp.Add.make_args(e):
        _, resto = termo.as_coeff_Mul()
        if resto == 1:
            base.add("1")
            continue
        resto = resto.xreplace({f: _C for f in resto.atoms(sp.Float)})
        base.add(sp.srepr(resto))
    return frozenset(base)


def mesma_forma_funcional(expr: Expr, referencia: Expr) -> bool:
    """``True`` se ``expr`` tem a forma funcional da referência (constantes livres)."""
    try:
        return termos_base(expr) == termos_base(referencia)
    except Exception:  # noqa: BLE001 - expressão que o sympy não normaliza = não recuperada
        return False
