"""Escada de equações-alvo (N1-N4) ancoradas no OVSED com parâmetros da Org1.

Cada nível é um :class:`NivelEquacao` com expressão de referência em infixo (constantes
numéricas, variáveis ``V_GS`` e ``V_DS`` em módulo, saída ``I_D`` em µA), formas
equivalentes, gerador numpy (a própria expressão), região de polarização e descrição.

Regra de ancoragem (H-OTFT-04): as constantes de cada nível são ajustadas ao OVSED completo
(R_S e I_leak incluídos; I_leak removido no N4) na região do nível, minimizando o erro
relativo quadrático, e arredondadas a 4 algarismos significativos. O gerador usa as
constantes arredondadas, logo gerador == expressão de referência.

- N1 saturação quadrática (controle canônico): I_D = K1 (V_GS - V_Th1)^2.
- N2 lei de potência OVSED (acumulação forte, F_sat constante): I_D = K2 (V_GS - V_th0)^l,
  com V_th0 e l da Org1 fixos e K2 = W J_th F_sat,ref / (n V_T)^l.
- N3 regime linear (triodo) quadrático clássico: I_D = K3 (2 (V_GS - V_Th3) V_DS - V_DS^2),
  válido para V_DS <= V_GS - V_Th3 (H-OTFT-05: tanh não é operador permitido).
- N4 sublimiar exponencial: I_D = I_0 exp(ln10 V_GS / S), I_leak removido.
- N5 (R_S implícita) fora do escopo de execução (H-OTFT-08).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

import numpy as np

from otft_sr.ovsed import (
    V_T,
    ParametrosOVSED,
    carregar_parametros_org1,
    corrente_ovsed_uA,
    fator_saturacao,
)

FEATURES: tuple[str, str] = ("V_GS", "V_DS")
ALGARISMOS = 4
LN10 = math.log(10.0)


def arred(x: float, digitos: int = ALGARISMOS) -> float:
    """Arredonda a ``digitos`` algarismos significativos."""
    return float(f"{x:.{digitos - 1}e}")


@dataclass(frozen=True)
class RegiaoPolarizacao:
    """Região de polarização do nível e forma da grade de curvas.

    ``tipo_curva``: ``transferencia`` (V_DS fixo por curva, V_GS varrido) ou ``saida``
    (V_GS fixo por curva, V_DS varrido). ``V_DS_max_rel_V_GS``: se definido, o limite
    superior de V_DS numa curva de saída é ``V_GS - V_DS_max_rel_V_GS`` (triodo).
    """

    V_GS: tuple[float, float]
    V_DS: tuple[float, float]
    tipo_curva: str
    com_leak: bool = True
    V_DS_max_rel_V_GS: float | None = None
    descricao: str = ""


@dataclass(frozen=True)
class NivelEquacao:
    """Nível da escada de equações-alvo."""

    id: str
    nome: str
    expressao: str  # infixo com constantes numéricas (parse_infix)
    formas_equivalentes: tuple[str, ...]
    gerador: Callable[[np.ndarray, np.ndarray], np.ndarray] = field(repr=False)
    regiao: RegiaoPolarizacao
    descricao_fisica: str
    constantes: dict[str, float]
    derivacao: str
    operadores_necessarios: tuple[str, ...]
    unidades: dict[str, str] = field(
        default_factory=lambda: {"V_GS": "V", "V_DS": "V", "I_D": "uA"}
    )
    #: I_D do nível = I_D(µA) x fator_uA (1 em N1-N3; 1e6 em N4, saída em pA: H-OTFT-11).
    fator_uA: float = 1.0


# ----------------------------------------------------------------------------- ajustes


def _grade_regiao(r: RegiaoPolarizacao, n: int = 41) -> tuple[np.ndarray, np.ndarray]:
    """Grade densa (V_GS, V_DS) cobrindo a região (usada nos ajustes e na validação)."""
    if r.tipo_curva == "transferencia":
        vg, vd = np.meshgrid(np.linspace(*r.V_GS, n), np.linspace(*r.V_DS, 6))
        return vg.ravel(), vd.ravel()
    vgs, vds = [], []
    for vg in np.linspace(*r.V_GS, 6):
        hi = r.V_DS[1] if r.V_DS_max_rel_V_GS is None else vg - r.V_DS_max_rel_V_GS
        hi = min(hi, r.V_DS[1])
        vd = np.linspace(r.V_DS[0], hi, n)
        vgs.append(np.full_like(vd, vg))
        vds.append(vd)
    return np.concatenate(vgs), np.concatenate(vds)


def _K_relativo(X: np.ndarray, I: np.ndarray) -> float:
    """K que minimiza sum((K X / I - 1)^2)."""
    u = X / I
    return float(np.sum(u) / np.sum(u * u))


def _ajuste_quadratico(
    p: ParametrosOVSED,
    regiao_de: Callable[[float], RegiaoPolarizacao],
    base: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    V_th_busca: np.ndarray,
) -> tuple[float, float]:
    """Busca em V_Th (grade grossa + refinamento 0,005 V) com K fechado; minimiza o erro
    relativo quadrático médio."""

    def avaliar(grade: np.ndarray) -> tuple[float, float, float]:
        melhor = (math.inf, math.nan, math.nan)
        for vt in grade:
            r = regiao_de(float(vt))
            vg, vd = _grade_regiao(r)
            I = corrente_ovsed_uA(vg, vd, p, com_leak=r.com_leak)
            X = base(vg, vd, float(vt))
            if np.any(X <= 0):
                continue
            K = _K_relativo(X, I)
            err = float(np.mean((K * X / I - 1.0) ** 2))
            if err < melhor[0]:
                melhor = (err, float(vt), K)
        return melhor

    _, vt0, _ = avaliar(V_th_busca)
    _, vt, K = avaliar(np.arange(vt0 - 0.25, vt0 + 0.25, 0.005))
    return vt, K


# ----------------------------------------------------------------------------- níveis


def _regiao_n1() -> RegiaoPolarizacao:
    return RegiaoPolarizacao(
        V_GS=(25.0, 50.0),
        V_DS=(40.0, 50.0),
        tipo_curva="transferencia",
        descricao="saturação: curvas de transferência com V_DS em [40, 50] V, V_GS em [25, 50] V",
    )


def _nivel_n1(p: ParametrosOVSED) -> NivelEquacao:
    r = _regiao_n1()
    V_Th, K = _ajuste_quadratico(
        p, lambda _vt: r, lambda vg, _vd, vt: (vg - vt) ** 2, np.arange(10.0, 24.0, 0.25)
    )
    K, V_Th = arred(K), arred(V_Th)
    return NivelEquacao(
        id="N1",
        nome="Saturação quadrática (lei de Shockley, controle canônico)",
        expressao=f"{K!r} * (V_GS - {V_Th!r}) ** 2",
        formas_equivalentes=(
            f"{K!r} * (V_GS - {V_Th!r}) * (V_GS - {V_Th!r})",
            f"{K!r} * V_GS ** 2 - {2 * K * V_Th!r} * V_GS + {K * V_Th**2!r}",
        ),
        gerador=lambda vg, vd: K * (vg - V_Th) ** 2 + 0.0 * vd,
        regiao=r,
        descricao_fisica=(
            "Lei quadrática de saturação I_D = (W/2L) C_i mu (V_GS - V_Th)^2 com mobilidade "
            "constante; K = (W/2L) C_i mu e V_Th efetivos."
        ),
        constantes={"K": K, "V_Th": V_Th},
        derivacao=(
            "K e V_Th por mínimo erro relativo quadrático contra o OVSED completo (Org1) na "
            "região de saturação; mesma ordem de grandeza de I_D da Org1 em saturação."
        ),
        operadores_necessarios=("sub", "mul", "pow"),
    )


def _nivel_n2(p: ParametrosOVSED) -> NivelEquacao:
    r = RegiaoPolarizacao(
        V_GS=(20.0, 50.0),
        V_DS=(40.0, 50.0),
        tipo_curva="transferencia",
        descricao=(
            "acumulação forte em saturação: transferência com V_DS em [40, 50] V, "
            "V_GS em [20, 50] V (theta >= 3,8: softplus ~ identidade)"
        ),
    )
    vg, vd = _grade_regiao(r)
    I = corrente_ovsed_uA(vg, vd, p)
    X = np.maximum(vg - p.V_th0, 0.0) ** p.l
    K = arred(_K_relativo(X, I))
    theta = (vg - p.V_th0 - p.delta * vd) / (p.n * V_T)
    F = fator_saturacao(vd, np.log1p(np.exp(theta)), p)
    F_ref = K * (p.n * V_T) ** p.l / (p.W * p.J_th)
    V_th0, l = p.V_th0, p.l
    return NivelEquacao(
        id="N2",
        nome="Lei de potência OVSED (mobilidade dependente de V_GS, F_sat constante)",
        expressao=f"{K!r} * (V_GS - {V_th0!r}) ** {l!r}",
        formas_equivalentes=(f"{K!r} * exp({l!r} * log(V_GS - {V_th0!r}))",),
        gerador=lambda vg, vd: K * np.abs(vg - V_th0) ** l + 0.0 * vd,
        regiao=r,
        descricao_fisica=(
            "Acumulação forte do OVSED: J_free ~ q_tot^l com q_tot ~ (V_GS - V_th0)/(n V_T) "
            "(transporte por saltos, l = gamma + 2) e F_sat fixo em F_sat,ref."
        ),
        constantes={"K": K, "V_th0": V_th0, "l": l, "F_sat_ref": float(F_ref)},
        derivacao=(
            "V_th0 e l da Org1 fixos; K = W J_th F_sat,ref/(n V_T)^l com F_sat,ref por mínimo "
            f"erro relativo contra o OVSED completo (F_sat,ref = {F_ref:.4f}; F_sat do OVSED "
            f"na região varia de {F.min():.3f} a {F.max():.3f})."
        ),
        operadores_necessarios=("sub", "mul", "pow"),
    )


def _regiao_n3(V_Th: float) -> RegiaoPolarizacao:
    return RegiaoPolarizacao(
        V_GS=(30.0, 50.0),
        V_DS=(0.5, 50.0),
        tipo_curva="saida",
        V_DS_max_rel_V_GS=V_Th,
        descricao=(
            "regime linear (triodo): curvas de saída com V_GS em [30, 50] V e "
            f"V_DS em [0,5, V_GS - {V_Th:.4g}] V"
        ),
    )


def _nivel_n3(p: ParametrosOVSED) -> NivelEquacao:
    V_Th, K = _ajuste_quadratico(
        p,
        _regiao_n3,
        lambda vg, vd, vt: 2.0 * (vg - vt) * vd - vd**2,
        np.arange(15.0, 28.0, 0.25),
    )
    K, V_Th = arred(K), arred(V_Th)
    return NivelEquacao(
        id="N3",
        nome="Regime linear quadrático clássico (triodo)",
        expressao=f"{K!r} * (2 * (V_GS - {V_Th!r}) * V_DS - V_DS ** 2)",
        formas_equivalentes=(
            f"{K!r} * (2 * (V_GS - {V_Th!r}) * V_DS - V_DS * V_DS)",
            f"{2 * K!r} * (V_GS - {V_Th!r}) * V_DS - {K!r} * V_DS ** 2",
            f"{K!r} * V_DS * (2 * V_GS - {2 * V_Th!r} - V_DS)",
        ),
        gerador=lambda vg, vd: K * (2.0 * (vg - V_Th) * vd - vd**2),
        regiao=_regiao_n3(V_Th),
        descricao_fisica=(
            "Nível 2 da proposta com mobilidade constante: I_D = (W/L) C_i mu "
            "[(V_GS - V_Th) V_DS - V_DS^2/2]; reproduz a expansão de tau(2 - tau) do F_sat "
            "OVSED para V_DS << V_gn."
        ),
        constantes={"K": K, "V_Th": V_Th},
        derivacao=(
            "K e V_Th por mínimo erro relativo quadrático contra o OVSED completo no triodo "
            "(região dependente de V_Th, ajuste autoconsistente)."
        ),
        operadores_necessarios=("sub", "mul", "pow"),
    )


def _nivel_n4(p: ParametrosOVSED) -> NivelEquacao:
    r = RegiaoPolarizacao(
        V_GS=(0.0, 4.0),
        V_DS=(1.0, 5.0),
        tipo_curva="transferencia",
        com_leak=False,
        descricao=(
            "sublimiar profundo (I_leak removido): transferência com V_DS em [1, 5] V, "
            "V_GS em [0, 4] V; I_D em pA"
        ),
    )
    vg, vd = _grade_regiao(r)
    I = corrente_ovsed_uA(vg, vd, p, com_leak=False) * 1e6  # pA
    b, a = np.polyfit(vg, np.log10(I), 1)
    S = arred(1.0 / b)
    I_0 = arred(10.0**a)
    c = arred(LN10 / S)
    S_l = LN10 * p.n * V_T / p.l
    S_l1 = LN10 * p.n * V_T / (p.l + 1)
    return NivelEquacao(
        id="N4",
        nome="Corrente sublimiar exponencial",
        expressao=f"{I_0!r} * exp({c!r} * V_GS)",
        formas_equivalentes=(f"exp({c!r} * V_GS + {arred(math.log(I_0))!r})",),
        gerador=lambda vg, vd: I_0 * np.exp(c * vg) + 0.0 * vd,
        regiao=r,
        descricao_fisica=(
            "I_D = I_0 10^(V_GS/S) = I_0 exp(ln10 V_GS / S); S = subthreshold swing."
        ),
        constantes={"I_0": I_0, "S": S, "ln10_sobre_S": c, "S_teorico_l": S_l, "S_teorico_l1": S_l1},
        derivacao=(
            f"Regressão de log10 I_D (OVSED sem I_leak) em V_GS: S = {S:.4g} V/déc. "
            f"Referências: ln10 n V_T/l = {S_l:.3f} V/déc (só J_free) e ln10 n V_T/(l+1) = "
            f"{S_l1:.3f} V/déc (assíntota, F_sat ~ q_tot no sublimiar)."
        ),
        operadores_necessarios=("mul", "exp"),
        unidades={"V_GS": "V", "V_DS": "V", "I_D": "pA"},
        fator_uA=1e6,
    )


@cache
def escada(V_th0: float | None = None) -> tuple[NivelEquacao, ...]:
    """Níveis N1-N4 (cache por V_th0; ``None`` = valor adotado em ``ovsed``)."""
    p = carregar_parametros_org1() if V_th0 is None else carregar_parametros_org1(V_th0=V_th0)
    return (_nivel_n1(p), _nivel_n2(p), _nivel_n3(p), _nivel_n4(p))


def nivel(nivel_id: str) -> NivelEquacao:
    """Nível pelo identificador (``N1``..``N4``)."""
    for n in escada():
        if n.id == nivel_id:
            return n
    raise KeyError(f"nível desconhecido: {nivel_id!r} (válidos: N1-N4; N5 fora do escopo)")


def erro_vs_ovsed(n: NivelEquacao, p: ParametrosOVSED | None = None) -> dict[str, float]:
    """Erro relativo |I_nivel/I_OVSED - 1| na grade densa da região do nível."""
    p = carregar_parametros_org1() if p is None else p
    vg, vd = _grade_regiao(n.regiao)
    I = corrente_ovsed_uA(vg, vd, p, com_leak=n.regiao.com_leak) * n.fator_uA
    rel = np.abs(n.gerador(vg, vd) / I - 1.0)
    return {
        "erro_rel_medio": float(np.mean(rel)),
        "erro_rel_mediano": float(np.median(rel)),
        "erro_rel_max": float(np.max(rel)),
        "rmse_log10": float(np.sqrt(np.mean((np.log10(n.gerador(vg, vd)) - np.log10(I)) ** 2))),
    }


N5_DOCUMENTACAO = (
    "N5 (saturação com R_S): I_D = K (V_GS - V_Th - I_D R_S)^(gamma+2) é implícita em I_D; "
    "a AST e os algoritmos de RS (DEAP, gplearn, PySR) só representam y = f(X) explícita. Fica como "
    "trabalho futuro (ex.: oferecer I_D como entrada e buscar o resíduo, ou RS sobre a "
    "inversa V_GS(I_D, V_DS)). Para a Org1, I_D R_S <= ~0,1 V (I_D < 10 µA, R_S = 10 kΩ): "
    "efeito de R_S < 1% em I_D na faixa medida."
)
