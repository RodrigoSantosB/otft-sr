"""Modelo compacto OVSED (canal p, em módulo) e forma fechada acima do limiar.

Reimplementação vetorizada de ``Model_OTFT/modules_otft/_model.py`` (TFTModel.calc_model),
sem importar ``modules_otft``. Convenções:

- Entradas ``V_GS``, ``V_DS`` em módulo (V), canal p tratado como |·| (H-OTFT-01).
- Saída ``I_D`` em ampères (funções ``*_A``) ou µA (``*_uA``).
- ``V_T = k_B T / q`` com k_B = 8.617e-5 eV/K e T = 293 K (mesmo valor do TCC).

Equações (por ponto de polarização):

    theta  = (V_GS - V_th0 - delta V_DS) / (n V_T)
    q_tot  = softplus(theta)
    J_free = J_th 1e-6 q_tot^l
    V_Gt   = n V_T q_tot
    V_gn   = 2 V_Gt / (1 + sqrt(2 V_Gt / V_crit))
    eta    = 1 - tanh(V_DS / V_gn),   y = V_gn / V_T
    ll     = 2 Lambda / (y^2 (1 - eta^2)) (e^{y(eta-1)} (1 - y eta) - (1 - y))
    a      = 1 / (1 + 2 ll)
    F_sat  = a (1 - e^{-V_DS/V_T}) / (1 + a e^{-V_DS/V_T})
    I_D    = I_leak + W J_free F_sat

R_S (escala 1e4 Ω) entra por ponto fixo amortecido (V_GS -> V_GS - I R_S,
V_DS -> V_DS - 2 I R_S), como no código original.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

K_B_EV = 8.617e-5  # eV/K
T_JUNCAO = 293.0  # K
V_T = K_B_EV * T_JUNCAO  # V (tensão térmica, ~25,25 mV)

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
ARQUIVO_PARAMETROS_ORG1 = RAIZ_PROJETO / "data" / "org1" / "inputs_Org1_SB.json"
DIR_DADOS_ORG1 = RAIZ_PROJETO / "data" / "org1"

#: V_th0 adotado (H-OTFT-02): o JSON traz 18.57, fora dos limites de otimização [8, 12] do
#: próprio arquivo; 8.57 (índices sintéticos/RELATORIO_MLP) reproduz melhor os dados medidos.
V_TH0_ADOTADO = 8.57


@dataclass(frozen=True)
class ParametrosOVSED:
    """Parâmetros físicos do OVSED (unidades do TCC)."""

    V_th0: float  # V, tensão de limiar
    delta: float  # adimensional, impacto de V_DS no limiar (DIBL)
    n: float  # adimensional, fator de idealidade sublimiar
    l: float  # adimensional, expoente da lei de potência (l = gamma + 2)  # noqa: E741
    Lambda: float  # adimensional, parâmetro de saturação
    V_crit: float  # V, tensão crítica
    J_th: float  # µA/cm (multiplicado por 1e-6 -> A/cm)
    R_S: float  # x 1e4 Ω
    W: float  # cm
    I_leak: float  # A

    def como_dict(self) -> dict[str, float]:
        """Dicionário serializável."""
        return asdict(self)


def carregar_parametros_org1(
    caminho: Path = ARQUIVO_PARAMETROS_ORG1, V_th0: float | None = V_TH0_ADOTADO
) -> ParametrosOVSED:
    """Lê ``loaded_parameters`` da Org1; ``V_th0=None`` mantém o valor do JSON (18.57)."""
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    bloco = next(d for d in dados if "loaded_parameters" in d)
    p = bloco["loaded_parameters"]
    params = ParametrosOVSED(
        V_th0=float(p["VTHO"]),
        delta=float(p["DELTA"]),
        n=float(p["N"]),
        l=float(p["L"]),
        Lambda=float(p["LAMBDA"]),
        V_crit=float(p["VGCRIT"]),
        J_th=float(p["JTH"]),
        R_S=float(p["RS"]),
        W=float(bloco["with_transistor"]),
        I_leak=float(bloco["loaded_idleak"]),
    )
    return params if V_th0 is None else replace(params, V_th0=float(V_th0))


def _softplus(theta: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return np.where(theta > 0, theta + np.log1p(np.exp(-theta)), np.log1p(np.exp(theta)))


def fator_saturacao(V_DS: np.ndarray, q_tot: np.ndarray, p: ParametrosOVSED) -> np.ndarray:
    """F_sat do OVSED (sem I_leak), vetorizado."""
    V_DS = np.asarray(V_DS, dtype=float)
    V_Gt = p.n * V_T * q_tot
    with np.errstate(all="ignore"):
        V_gn = 2.0 * V_Gt / (1.0 + np.sqrt(2.0 * V_Gt / p.V_crit))
        eta = 1.0 - np.tanh(V_DS / V_gn)
        y = V_gn / V_T
        ll = (
            2.0
            * p.Lambda
            / (y**2 * (1.0 - eta**2))
            * (np.exp(y * (eta - 1.0)) * (1.0 - y * eta) - (1.0 - y))
        )
        ll = np.nan_to_num(ll, nan=p.Lambda, posinf=1e300)
        a = 1.0 / (1.0 + 2.0 * ll)
        e = np.exp(-V_DS / V_T)
        F_sat = a * (1.0 - e) / (1.0 + a * e)
    return np.nan_to_num(F_sat, nan=0.0)


def _corrente_intrinseca_A(
    V_GS: np.ndarray, V_DS: np.ndarray, p: ParametrosOVSED, com_leak: bool
) -> np.ndarray:
    theta = (V_GS - p.V_th0 - p.delta * V_DS) / (p.n * V_T)
    q_tot = _softplus(theta)
    J_free = p.J_th * 1e-6 * q_tot**p.l
    I_leak = p.I_leak if com_leak else 0.0
    return I_leak + p.W * J_free * fator_saturacao(V_DS, q_tot, p)


def corrente_ovsed_A(
    V_GS: np.ndarray,
    V_DS: np.ndarray,
    p: ParametrosOVSED,
    com_R_S: bool = True,
    com_leak: bool = True,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> np.ndarray:
    """I_D (A) do OVSED completo; R_S por ponto fixo amortecido (0,1/0,9, como no TCC)."""
    V_GS, V_DS = np.broadcast_arrays(np.abs(np.asarray(V_GS, float)), np.abs(np.asarray(V_DS, float)))
    I = _corrente_intrinseca_A(V_GS, V_DS, p, com_leak)
    if not com_R_S or p.R_S == 0:
        return I
    R = p.R_S * 1e4
    I_ant = np.full_like(I, p.I_leak if com_leak else 0.0)
    dvg = I_ant * R
    for _ in range(max_iter):
        rel = np.abs(I - I_ant) / np.maximum(np.abs(I), 1e-30)
        if np.all(rel <= tol):
            break
        I_ant = I
        dvg = 0.1 * I * R + 0.9 * dvg
        V_GSi = np.maximum(V_GS - dvg, 0.0)
        V_DSi = np.maximum(V_DS - 2.0 * dvg, 0.0)
        I = _corrente_intrinseca_A(V_GSi, V_DSi, p, com_leak)
    return I


def corrente_ovsed_uA(V_GS, V_DS, p: ParametrosOVSED, **kw) -> np.ndarray:  # noqa: ANN001
    """I_D (µA) do OVSED completo."""
    return corrente_ovsed_A(V_GS, V_DS, p, **kw) * 1e6


def corrente_forma_fechada_uA(V_GS, V_DS, p: ParametrosOVSED) -> np.ndarray:  # noqa: ANN001
    """Forma fechada acima do limiar (R_S = 0, I_leak = 0, softplus -> identidade), em µA.

    V_ov = V_GS - V_th0 - delta V_DS; V_gn = 2 V_ov / (1 + sqrt(2 V_ov / V_crit));
    tau = tanh(V_DS / V_gn); I_D = W J_th (V_ov/(n V_T))^l / (1 + 4 Lambda V_T / (V_gn tau (2 - tau))).
    """
    V_GS = np.abs(np.asarray(V_GS, float))
    V_DS = np.abs(np.asarray(V_DS, float))
    V_ov = np.maximum(V_GS - p.V_th0 - p.delta * V_DS, 1e-12)
    V_gn = 2.0 * V_ov / (1.0 + np.sqrt(2.0 * V_ov / p.V_crit))
    tau = np.tanh(V_DS / V_gn)
    with np.errstate(divide="ignore"):
        F = 1.0 / (1.0 + 4.0 * p.Lambda * V_T / (V_gn * tau * (2.0 - tau)))
    return p.W * p.J_th * (V_ov / (p.n * V_T)) ** p.l * np.nan_to_num(F)


# ----------------------------------------------------------------------------- dados medidos


@dataclass(frozen=True)
class CurvaMedida:
    """Curva medida da Org1 em módulo (V, V, µA)."""

    nome: str
    tipo: str  # "transferencia" (V_DS fixo) ou "saida" (V_GS fixo)
    V_GS: np.ndarray
    V_DS: np.ndarray
    I_D_uA: np.ndarray


def carregar_curvas_org1(diretorio: Path = DIR_DADOS_ORG1) -> list[CurvaMedida]:
    """Lê as 5 curvas da Org1. Transferência (``*VDS.csv``): I_D em A; saída (``*VGS.csv``):
    rotulada A mas em µA (``experimental_data_scale_output = uA`` no JSON)."""
    curvas = []
    for arq in sorted(Path(diretorio).glob("org1_*.csv")):
        bruto = np.genfromtxt(arq, delimiter=",", skip_header=2)
        bruto = bruto[np.all(np.isfinite(bruto), axis=1)]
        if arq.stem.endswith("VDS"):
            V_DS, V_GS, I = bruto[:, 0], bruto[:, 1], bruto[:, 2] * 1e6
            tipo = "transferencia"
        else:
            V_GS, V_DS, I = bruto[:, 0], bruto[:, 1], bruto[:, 2]
            tipo = "saida"
        ordem = np.argsort(np.abs(V_GS) if tipo == "transferencia" else np.abs(V_DS))
        curvas.append(
            CurvaMedida(
                arq.stem, tipo, np.abs(V_GS)[ordem], np.abs(V_DS)[ordem], np.abs(I)[ordem]
            )
        )
    return curvas


def erro_log_medido(p: ParametrosOVSED, curvas: list[CurvaMedida] | None = None) -> float:
    """RMSE em log10|I_D| (décadas) do OVSED completo contra todas as curvas medidas."""
    curvas = carregar_curvas_org1() if curvas is None else curvas
    res = []
    for c in curvas:
        m = c.I_D_uA > 0
        mod = corrente_ovsed_uA(c.V_GS[m], c.V_DS[m], p)
        res.append(np.log10(mod) - np.log10(c.I_D_uA[m]))
    r = np.concatenate(res)
    return float(np.sqrt(np.mean(r**2)))
