"""Escada de equações-alvo (níveis N1-N4 da proposta, Seção 5.1) ancoradas no OVSED (Org1).

O identificador ``N<k>`` coincide com o nível da proposta (``nivel_proposta`` = k). Cada
nível é um :class:`NivelEquacao` com nome científico (``nome``), expressão de referência em
infixo (constantes numéricas, variáveis ``V_GS`` e ``V_DS`` em módulo, saída ``I_D`` em µA),
formas equivalentes, gerador numpy (a própria expressão), região de polarização e descrição.

Regra de ancoragem (H-OTFT-04): as constantes de cada nível são ajustadas ao OVSED completo
(R_S e I_leak incluídos) na região do nível, minimizando o erro relativo quadrático (N1-N3)
ou o erro em log10 I_D (N4), e arredondadas a 4 algarismos significativos. O gerador usa as
constantes arredondadas, logo gerador == expressão de referência.

- N1 saturação quadrática ideal (controle canônico): I_D = K (V_GS - V_Th)^2.
- N2 regime linear (triodo) com termo quadrático em V_DS:
  I_D = K (2 (V_GS - V_Th) V_DS - V_DS^2), válido para V_DS <= V_GS - V_Th
  (H-OTFT-05: tanh não é operador permitido).
- N3 saturação com mobilidade em lei de potência (OVSED, F_sat constante), V_GS de 10,5 a
  28 V (perto do limiar, H-OTFT-18): I_D = K (V_GS - V_th0)^l, com V_th0 e l da Org1 fixos e
  K = W J_th F_sat,ref / (n V_T)^l.
- N4 corrente sublimiar exponencial com fuga aditiva (H-OTFT-17):
  I_D = I_leak + I_0 exp(ln10 V_GS / S), em pA, V_GS de 0 a 10 V.
- N5 saturação com resistência de contato (R_S implícita): fora do escopo de execução
  (H-OTFT-08), só documentado.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

import numpy as np

from otft_sr.ovsed import (
    V_T,
    CurvaMedida,
    ParametrosOVSED,
    carregar_curvas_org1,
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
    nivel_proposta: int  # 1..5 = índice do id (Seção 5.1 da proposta)
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


def _regiao_saturacao_quadratica() -> RegiaoPolarizacao:
    return RegiaoPolarizacao(
        V_GS=(25.0, 50.0),
        V_DS=(40.0, 50.0),
        tipo_curva="transferencia",
        descricao="saturação: curvas de transferência com V_DS em [40, 50] V, V_GS em [25, 50] V",
    )


def _nivel_saturacao_quadratica(p: ParametrosOVSED) -> NivelEquacao:
    r = _regiao_saturacao_quadratica()
    V_Th, K = _ajuste_quadratico(
        p, lambda _vt: r, lambda vg, _vd, vt: (vg - vt) ** 2, np.arange(10.0, 24.0, 0.25)
    )
    K, V_Th = arred(K), arred(V_Th)
    return NivelEquacao(
        id="N1",
        nivel_proposta=1,
        nome="Saturação quadrática ideal",
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


def _nivel_saturacao_lei_de_potencia(p: ParametrosOVSED) -> NivelEquacao:
    # Região (H-OTFT-18): desce até perto do limiar, onde lei quadrática e lei de potência
    # mais divergem; limites escolhidos por máxima discriminabilidade contra K (V_GS - V_T)^2
    # com erro do N3 vs OVSED <= 0,15 década (RMSE) e <= 0,30 década (máximo).
    r = RegiaoPolarizacao(
        V_GS=(10.5, 28.0),
        V_DS=(40.0, 50.0),
        tipo_curva="transferencia",
        descricao=(
            "saturação próxima do limiar: transferência com V_DS em [40, 50] V, "
            "V_GS em [10,5, 28] V (theta de 0,5 a 7: transição para a acumulação forte)"
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
        id="N3",
        nivel_proposta=3,
        nome="Saturação com mobilidade em lei de potência",
        expressao=f"{K!r} * (V_GS - {V_th0!r}) ** {l!r}",
        formas_equivalentes=(f"{K!r} * exp({l!r} * log(V_GS - {V_th0!r}))",),
        gerador=lambda vg, vd: K * np.abs(vg - V_th0) ** l + 0.0 * vd,
        regiao=r,
        descricao_fisica=(
            "Acumulação do OVSED: J_free ~ q_tot^l com q_tot ~ (V_GS - V_th0)/(n V_T) "
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


def _regiao_triodo_quadratico(V_Th: float) -> RegiaoPolarizacao:
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


def _nivel_triodo_quadratico(p: ParametrosOVSED) -> NivelEquacao:
    V_Th, K = _ajuste_quadratico(
        p,
        _regiao_triodo_quadratico,
        lambda vg, vd, vt: 2.0 * (vg - vt) * vd - vd**2,
        np.arange(15.0, 28.0, 0.25),
    )
    K, V_Th = arred(K), arred(V_Th)
    return NivelEquacao(
        id="N2",
        nivel_proposta=2,
        nome="Regime linear (triodo) com termo quadrático em V_DS",
        expressao=f"{K!r} * (2 * (V_GS - {V_Th!r}) * V_DS - V_DS ** 2)",
        formas_equivalentes=(
            f"{K!r} * (2 * (V_GS - {V_Th!r}) * V_DS - V_DS * V_DS)",
            f"{2 * K!r} * (V_GS - {V_Th!r}) * V_DS - {K!r} * V_DS ** 2",
            f"{K!r} * V_DS * (2 * V_GS - {2 * V_Th!r} - V_DS)",
        ),
        gerador=lambda vg, vd: K * (2.0 * (vg - V_Th) * vd - vd**2),
        regiao=_regiao_triodo_quadratico(V_Th),
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


#: Limite superior de V_GS do N4 (H-OTFT-17): pouco acima do joelho em que a corrente de canal
#: do OVSED cruza I_leak (~8,9 V a V_DS = 2 V); acima de ~10 V a inclinação local passa de
#: 2,8 V/déc e o canal deixa de ser exponencial.
V_GS_MAX_SUBLIMIAR = 10.0


def _ajuste_exponencial_com_fuga(
    vg: np.ndarray, I: np.ndarray, I_leak: float
) -> tuple[float, float]:
    """(I_0, b) que minimizam sum((log10(I_leak + I_0 10^(b V_GS)) - log10 I)^2), I_leak fixo.

    Gauss-Newton em (log10 I_0, b) a partir da regressão log-linear dos pontos com canal
    acima de I_leak.
    """
    m = I > 2.0 * I_leak
    b, a = np.polyfit(vg[m], np.log10(I[m] - I_leak), 1)
    x = np.array([a, b])
    alvo = np.log10(I)
    for _ in range(100):
        ch = 10.0 ** (x[0] + x[1] * vg)
        mod = I_leak + ch
        r = np.log10(mod) - alvo
        w = ch / mod  # d log10(mod) / d (log10 I_0); vezes V_GS para b
        J = np.column_stack([w, w * vg])
        passo = np.linalg.lstsq(J, -r, rcond=None)[0]
        x = x + passo
        if np.max(np.abs(passo)) < 1e-12:
            break
    return float(10.0 ** x[0]), float(x[1])


def _nivel_sublimiar_exponencial(p: ParametrosOVSED) -> NivelEquacao:
    r = RegiaoPolarizacao(
        V_GS=(0.0, V_GS_MAX_SUBLIMIAR),
        V_DS=(1.0, 5.0),
        tipo_curva="transferencia",
        com_leak=True,
        descricao=(
            "sublimiar observável (com I_leak): transferência com V_DS em [1, 5] V, "
            f"V_GS em [0, {V_GS_MAX_SUBLIMIAR:g}] V; I_D em pA"
        ),
    )
    vg, vd = _grade_regiao(r)
    I = corrente_ovsed_uA(vg, vd, p) * 1e6  # pA, com I_leak
    I_leak = arred(p.I_leak * 1e12)  # pA (parâmetro do OVSED, não ajustado)
    I_0, b = _ajuste_exponencial_com_fuga(vg, I, I_leak)
    S = arred(1.0 / b)
    I_0 = arred(I_0)
    c = arred(LN10 / S)
    S_l = LN10 * p.n * V_T / p.l
    S_l1 = LN10 * p.n * V_T / (p.l + 1)
    return NivelEquacao(
        id="N4",
        nivel_proposta=4,
        nome="Corrente sublimiar exponencial",
        expressao=f"{I_leak!r} + {I_0!r} * exp({c!r} * V_GS)",
        formas_equivalentes=(
            f"exp({c!r} * V_GS + {arred(math.log(I_0))!r}) + {I_leak!r}",
            f"{I_0!r} * (exp({c!r} * V_GS) + {arred(I_leak / I_0)!r})",
        ),
        gerador=lambda vg, vd: I_leak + I_0 * np.exp(c * vg) + 0.0 * vd,
        regiao=r,
        descricao_fisica=(
            "I_D = I_leak + I_0 10^(V_GS/S) = I_leak + I_0 exp(V_GS/V_S), V_S = S/ln10; "
            "S = subthreshold swing efetivo; I_leak = patamar de fuga aditivo do OVSED."
        ),
        constantes={
            "I_leak": I_leak, "I_0": I_0, "S": S, "ln10_sobre_S": c,
            "S_teorico_l": S_l, "S_teorico_l1": S_l1,
        },
        derivacao=(
            f"I_leak = {I_leak:g} pA (parâmetro do OVSED); I_0 e S por mínimo erro em log10 I_D "
            f"contra o OVSED completo (com I_leak): S = {S:.4g} V/déc (swing efetivo da "
            f"transição acima de I_leak). Referências: ln10 n V_T/l = {S_l:.3f} V/déc (só "
            f"J_free) e ln10 n V_T/(l+1) = {S_l1:.3f} V/déc (assíntota sublimiar profunda)."
        ),
        operadores_necessarios=("add", "mul", "exp"),
        unidades={"V_GS": "V", "V_DS": "V", "I_D": "pA"},
        fator_uA=1e6,
    )


@cache
def escada(V_th0: float | None = None) -> tuple[NivelEquacao, ...]:
    """Níveis N1-N4 da proposta, nessa ordem (cache por V_th0; ``None`` = valor em ``ovsed``)."""
    p = carregar_parametros_org1() if V_th0 is None else carregar_parametros_org1(V_th0=V_th0)
    return (
        _nivel_saturacao_quadratica(p),
        _nivel_triodo_quadratico(p),
        _nivel_saturacao_lei_de_potencia(p),
        _nivel_sublimiar_exponencial(p),
    )


def nivel(nivel_id: str) -> NivelEquacao:
    """Nível pelo identificador (``N1``..``N4``)."""
    for n in escada():
        if n.id == nivel_id:
            return n
    raise KeyError(
        f"nível desconhecido: {nivel_id!r} (válidos: {[n.id for n in escada()]}; "
        "N5 fora do escopo)")


def erro_vs_ovsed(n: NivelEquacao, p: ParametrosOVSED | None = None) -> dict[str, float]:
    """Erro relativo |I_nivel/I_OVSED - 1| na grade densa da região do nível."""
    p = carregar_parametros_org1() if p is None else p
    vg, vd = _grade_regiao(n.regiao)
    I = corrente_ovsed_uA(vg, vd, p, com_leak=n.regiao.com_leak) * n.fator_uA
    y = n.gerador(vg, vd)
    rel = np.abs(y / I - 1.0)
    dlog = np.log10(y) - np.log10(I)
    return {
        "erro_rel_medio": float(np.mean(rel)),
        "erro_rel_mediano": float(np.median(rel)),
        "erro_rel_max": float(np.max(rel)),
        "rmse_log10": float(np.sqrt(np.mean(dlog**2))),
        "max_log10": float(np.max(np.abs(dlog))),
    }


def _melhor_quadratica(vg: np.ndarray, y: np.ndarray, em_log: bool) -> tuple[float, float, float]:
    """Melhor K (V_GS - V_T)^2 para ``y``: (erro, K, V_T), com V_T livre (busca em grade de
    0,05 V e refinamento de 0,0005 V; K em forma fechada).

    ``em_log=False``: erro = 1 - R^2 = MSE/Var(y) (K por mínimos quadrados lineares);
    ``em_log=True``: erro = RMSE de log10 (log10 K = média do resíduo em log10).
    """
    ly = np.log10(y)

    def erro(vt: float) -> tuple[float, float]:
        X = (vg - vt) ** 2
        if em_log:
            lx = np.log10(np.maximum(X, 1e-300))
            a = float(np.mean(ly - lx))
            return float(np.sqrt(np.mean((a + lx - ly) ** 2))), 10.0**a
        K = float(np.dot(X, y) / np.dot(X, X))
        return float(np.mean((K * X - y) ** 2) / np.var(y)), K

    def varrer(grade: np.ndarray) -> tuple[float, float, float]:
        return min((*erro(float(vt)), float(vt)) for vt in grade)

    _, _, vt0 = varrer(np.arange(-30.0, float(vg.max()), 0.05))
    e, K, vt = varrer(np.arange(vt0 - 0.05, vt0 + 0.05, 0.0005))
    return e, K, vt


def discriminabilidade_quadratica(n: NivelEquacao) -> dict[str, float]:
    """Quão mal a forma de N1, K (V_GS - V_T)^2 com K e V_T livres, descreve os dados do nível.

    Avaliada no gerador do nível (dados sem ruído) na grade densa da região. Critério de
    escolha da região do N3 (H-OTFT-18): quanto maior, melhor posta a recuperação da lei de
    potência frente à lei quadrática. Para o próprio nível os dois erros são 0.
    """
    vg, vd = _grade_regiao(n.regiao)
    y = n.gerador(vg, vd)
    e_lin, K_lin, vt_lin = _melhor_quadratica(vg, y, em_log=False)
    e_log, K_log, vt_log = _melhor_quadratica(vg, y, em_log=True)
    return {
        "one_minus_r2_quadratica": e_lin,
        "K_quadratica": K_lin,
        "V_T_quadratica": vt_lin,
        "rmse_log10_quadratica": e_log,
        "K_quadratica_log": K_log,
        "V_T_quadratica_log": vt_log,
    }


def pontos_medidos_na_regiao(
    n: NivelEquacao, curvas: list[CurvaMedida] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(V_GS, V_DS, I_D na unidade do nível) das curvas medidas da Org1 do mesmo tipo do nível,
    restritas à região de polarização (inclusive o limite de triodo)."""
    curvas = carregar_curvas_org1() if curvas is None else curvas
    r = n.regiao
    vg_l, vd_l, i_l = [], [], []
    for c in curvas:
        if c.tipo != r.tipo_curva:
            continue
        m = (
            (c.V_GS >= r.V_GS[0]) & (c.V_GS <= r.V_GS[1])
            & (c.V_DS >= r.V_DS[0]) & (c.V_DS <= r.V_DS[1]) & (c.I_D_uA > 0)
        )
        if r.V_DS_max_rel_V_GS is not None:
            m &= c.V_DS <= c.V_GS - r.V_DS_max_rel_V_GS
        vg_l.append(c.V_GS[m])
        vd_l.append(c.V_DS[m])
        i_l.append(c.I_D_uA[m] * n.fator_uA)
    return np.concatenate(vg_l), np.concatenate(vd_l), np.concatenate(i_l)


def erro_vs_medida_org1(n: NivelEquacao, curvas: list[CurvaMedida] | None = None) -> dict[str, float]:
    """RMSE e máximo de |log10(I_nivel / I_medida)| nos pontos medidos da Org1 na região."""
    vg, vd, I = pontos_medidos_na_regiao(n, curvas)
    if I.size == 0:
        return {"n_pontos": 0, "rmse_log10": math.nan, "max_log10": math.nan}
    r = np.log10(n.gerador(vg, vd)) - np.log10(I)
    return {
        "n_pontos": int(I.size),
        "rmse_log10": float(np.sqrt(np.mean(r**2))),
        "max_log10": float(np.max(np.abs(r))),
    }


N5_DOCUMENTACAO = (
    "N5 (saturação com resistência de contato, R_S): I_D = K (V_GS - V_Th - I_D R_S)^(gamma+2) é implícita em I_D; "
    "a AST e os algoritmos de RS (DEAP, gplearn, PySR) só representam y = f(X) explícita. Fica como "
    "trabalho futuro (ex.: oferecer I_D como entrada e buscar o resíduo, ou RS sobre a "
    "inversa V_GS(I_D, V_DS)). Para a Org1, I_D R_S <= ~0,1 V (I_D < 10 µA, R_S = 10 kΩ): "
    "efeito de R_S < 1% em I_D na faixa medida."
)
