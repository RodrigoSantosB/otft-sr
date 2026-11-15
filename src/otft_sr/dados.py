"""Grade de polarização por curvas, ruído multiplicativo e partição por curva.

- Cada curva é uma curva de transferência (V_DS fixo) ou de saída (V_GS fixo), conforme a
  região do nível; os pontos da variável varrida são sorteados uniformemente na região.
- Ruído (proposta §4.5): I_obs = I_true (1 + sigma N(0, 1)), sigma in {0, 0.01, 0.05, 0.10}.
- Partição treino/validação/teste POR CURVA (``simulation_ids`` = id da curva): nenhuma
  curva aparece em mais de uma partição (H-OTFT-06).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics_llm_sr.core_types import Dataset

from otft_sr.niveis import FEATURES, NivelEquacao

NIVEIS_RUIDO: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10)
FRACOES_PARTICAO: tuple[float, float, float] = (0.6, 0.2, 0.2)


@dataclass(frozen=True)
class GradeCurvas:
    """Pontos de polarização agrupados por curva."""

    V_GS: np.ndarray
    V_DS: np.ndarray
    id_curva: np.ndarray
    valor_fixo: np.ndarray  # V_DS (transferência) ou V_GS (saída) de cada curva
    tipo_curva: str


def gerar_grade(
    nivel: NivelEquacao, n_curvas: int, n_pontos: int, rng: np.random.Generator
) -> GradeCurvas:
    """Grade de ``n_curvas`` curvas com ``n_pontos`` pontos sorteados cada."""
    r = nivel.regiao
    vg_l, vd_l, ids = [], [], []
    if r.tipo_curva == "transferencia":
        fixos = np.linspace(*r.V_DS, n_curvas)
        for i, vd in enumerate(fixos):
            vg = np.sort(rng.uniform(*r.V_GS, n_pontos))
            vg_l.append(vg)
            vd_l.append(np.full(n_pontos, vd))
            ids.append(np.full(n_pontos, i))
    elif r.tipo_curva == "saida":
        fixos = np.linspace(*r.V_GS, n_curvas)
        for i, vg in enumerate(fixos):
            hi = r.V_DS[1]
            if r.V_DS_max_rel_V_GS is not None:
                hi = min(hi, vg - r.V_DS_max_rel_V_GS)
            vd = np.sort(rng.uniform(r.V_DS[0], hi, n_pontos))
            vg_l.append(np.full(n_pontos, vg))
            vd_l.append(vd)
            ids.append(np.full(n_pontos, i))
    else:
        raise ValueError(f"tipo de curva desconhecido: {r.tipo_curva!r}")
    return GradeCurvas(
        np.concatenate(vg_l), np.concatenate(vd_l), np.concatenate(ids), fixos, r.tipo_curva
    )


def aplicar_ruido(y: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Ruído gaussiano multiplicativo proporcional a |I_D|."""
    if sigma < 0:
        raise ValueError("sigma deve ser >= 0")
    if sigma == 0:
        return np.array(y, dtype=float, copy=True)
    return y * (1.0 + sigma * rng.standard_normal(y.shape))


def particionar_curvas(
    n_curvas: int, rng: np.random.Generator, fracoes: tuple[float, float, float] = FRACOES_PARTICAO
) -> dict[str, np.ndarray]:
    """Ids de curva por partição (cada partição com >= 1 curva)."""
    if n_curvas < 3:
        raise ValueError("são necessárias >= 3 curvas para treino/validação/teste")
    ordem = rng.permutation(n_curvas)
    n_val = max(1, int(round(fracoes[1] * n_curvas)))
    n_te = max(1, int(round(fracoes[2] * n_curvas)))
    n_tr = n_curvas - n_val - n_te
    if n_tr < 1:
        raise ValueError("frações deixam o treino vazio")
    return {
        "treino": np.sort(ordem[:n_tr]),
        "validacao": np.sort(ordem[n_tr : n_tr + n_val]),
        "teste": np.sort(ordem[n_tr + n_val :]),
    }


def _subconjunto(
    nivel: NivelEquacao,
    grade: GradeCurvas,
    y_true: np.ndarray,
    y_obs: np.ndarray,
    curvas: np.ndarray,
    particao: str,
    sigma: float,
) -> Dataset:
    m = np.isin(grade.id_curva, curvas)
    X = np.column_stack([grade.V_GS[m], grade.V_DS[m]])
    ids = grade.id_curva[m]
    t = np.concatenate([np.arange(np.sum(ids == c)) for c in np.unique(ids)]).astype(float)
    return Dataset(
        X=X,
        y_true=y_true[m],
        y_observed=y_obs[m],
        feature_names=FEATURES,
        target_name="I_D",
        scenario=nivel.id,
        simulation_ids=ids,
        t=t,
        metadata={
            "particao": particao,
            "ruido_sigma": sigma,
            "tipo_curva": grade.tipo_curva,
            "curvas": curvas.tolist(),
            "unidade_I_D": nivel.unidades["I_D"],
        },
    )


def construir_datasets(
    nivel: NivelEquacao,
    sigma: float = 0.0,
    semente: int = 0,
    n_curvas: int = 20,
    n_pontos: int = 50,
    fracoes: tuple[float, float, float] = FRACOES_PARTICAO,
) -> dict[str, Dataset]:
    """Datasets ``treino``/``validacao``/``teste`` do nível (ruído em todas as partições;
    ``y_true`` limpo é mantido para as métricas de teste)."""
    rng = np.random.default_rng(np.random.SeedSequence([semente, int(nivel.id[1:]), 7]))
    grade = gerar_grade(nivel, n_curvas, n_pontos, rng)
    y_true = nivel.gerador(grade.V_GS, grade.V_DS)
    y_obs = aplicar_ruido(y_true, sigma, rng)
    part = particionar_curvas(n_curvas, rng, fracoes)
    return {k: _subconjunto(nivel, grade, y_true, y_obs, v, k, sigma) for k, v in part.items()}
