"""Configuração, execução de uma condição experimental e controle positivo.

Condições experimentais (proposta §5.4): A = RS sem LLM (w3 = 0), B = LLM com prompt
genérico, C = LLM com prompt de física de dispositivo. D (MLP) fica fora do projeto.

Perda (H-OTFT-07): modo ``normalized`` (e = MSE/Var(y), s = nós/max_nodes, trilha
``extended_analysis``), max_nodes = 30. Pesos por razões a w1: w2/w1 = 0,003 (um nó vale
1e-4 de 1 - R^2) e w3/w1 = 0,01 (c inteiro vale 1e-2 de 1 - R^2); A usa w3 = 0 com a mesma
razão w2/w1.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from physics_llm_sr.config.loader import parse_config
from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.evaluation.metrics import regression_metrics, structural_metrics
from physics_llm_sr.experiments.sr_registry import make_regressor
from physics_llm_sr.llm.interfaces import PhysicsEquationEvaluator
from physics_llm_sr.optimization.objective import build_objective
from physics_llm_sr.symbolic.expression import Expr, evaluate, parse_infix, size, to_infix
from physics_llm_sr.symbolic.operators import ProtectionParams
from physics_llm_sr.symbolic.tree_distance import TreeDistanceParams
from physics_llm_sr.workflows.baseline_study import ConsistencyCriteria, classify

from otft_sr.contexto import VARIANTE_PROMPT, contexto_fisico
from otft_sr.dados import construir_datasets
from otft_sr.equivalencia import mesma_forma_funcional
from otft_sr.formulacao import (
    datasets_formulados,
    formulacao_do_nivel,
    nivel_formulado,
    validar_formulacao,
)
from otft_sr.niveis import FEATURES, NivelEquacao

#: Variável de ambiente com a raiz do repositório physics-llm-sr (Colab: clone fixado).
VARIAVEL_REPO_PHYSICS_LLM_SR = "PHYSICS_LLM_SR_REPO_DIR"
REPO_PHYSICS_LLM_SR_LOCAL = Path("/home/rsb6/Desktop/projeto cleber /physics-llm-sr")
ARQUIVO_CONFIG_BASE = Path("configs") / "nb01_baseline.yaml"


def repo_physics_llm_sr() -> Path:
    """Raiz do physics-llm-sr: ``$PHYSICS_LLM_SR_REPO_DIR``, o checkout local validado ou a
    raiz deduzida do pacote importado (instalação editável)."""
    env = os.environ.get(VARIAVEL_REPO_PHYSICS_LLM_SR)
    if env:
        return Path(env)
    if (REPO_PHYSICS_LLM_SR_LOCAL / ARQUIVO_CONFIG_BASE).is_file():
        return REPO_PHYSICS_LLM_SR_LOCAL
    import physics_llm_sr

    return Path(physics_llm_sr.__file__).resolve().parents[2]

ALGORITMOS_RS: tuple[str, ...] = ("pysr", "deap", "gplearn")
CONDICOES: tuple[str, ...] = ("A", "B", "C")
MAX_NOS = 30
RAZAO_W2_W1 = 0.003
RAZAO_W3_W1 = 0.01


def _pesos(r2: float, r3: float) -> tuple[float, float, float]:
    tot = 1.0 + r2 + r3
    return (1.0 / tot, r2 / tot, r3 / tot)


PESOS_LLM: tuple[float, float, float] = _pesos(RAZAO_W2_W1, RAZAO_W3_W1)
PESOS_LINHA_DE_BASE: tuple[float, float, float] = _pesos(RAZAO_W2_W1, 0.0)
CRITERIO = ConsistencyCriteria(max_one_minus_r2=1e-3, max_tree_distance=0.05)
CLASSE_SYMPY = "recupera estrutura (equivalência sympy)"

#: Esquema de colunas de uma linha de ensaio (M1, ``executar_condicao``), na ordem gravada.
#: O modo M2 (``otft_sr.gerador.executar_geracao``) grava as mesmas colunas e acrescenta as
#: próprias; ``erro``/``traceback`` só existem em ensaios com erro.
COLUNAS_M1: tuple[str, ...] = (
    "nivel", "nivel_proposta", "algoritmo", "condicao", "ruido", "semente",
    "configuracao_busca", "formulacao_perda", "expressao_referencia", "n_treino", "n_teste",
    "tempo_s", "stop_reason", "n_geracoes", "expressao", "n_nos", "one_minus_r2",
    "one_minus_r2_I_D", "rmse_log10",
    "frac_pred_nao_positiva", "tree_distance", "tree_distance_equivalent_min",
    "tree_distance_algebraic", "forma_funcional_sympy", "status", "erro", "traceback",
    "chamadas_llm", "expressoes_unicas_llm", "cache_hits_llm", "tempo_llm_s",
    "classe_distancia", "classe",
)

_OPS_BASE = ["add", "sub", "mul", "div", "neg", "exp", "log", "sin", "cos"]
OPERADORES: dict[str, list[str]] = {
    "deap": [*_OPS_BASE, "pow"],
    "gplearn": list(_OPS_BASE),  # gplearn não tem pow: (.)**l via exp(l*log(.))
    "pysr": ["add", "sub", "mul", "div", "pow", "neg", "exp", "log", "sin", "cos"],
}

#: Configurações de busca por algoritmo de RS. "calibrada" = best_config da queda livre
#: (physics-llm-sr/references/convergence/<algoritmo>/free_fall/best_config.yaml);
#: "verificacao" = execução de verificação do pipeline (não constitui resultado).
CONFIGURACOES_BUSCA: dict[str, dict[str, dict[str, Any]]] = {
    "verificacao": {
        "pysr": {
            "sr.early_stopping.enabled": False,
            "sr.pysr.niterations": 2,
            "sr.pysr.populations": 2,
            "sr.pysr.population_size": 20,
            "sr.pysr.ncycles_per_iteration": 20,
        },
        "deap": {"sr.population": 40, "sr.generations": 4, "sr.early_stopping.enabled": False},
        "gplearn": {"sr.population": 40, "sr.generations": 4, "sr.early_stopping.enabled": False},
    },
    "calibrada": {
        "pysr": {
            "sr.early_stopping.enabled": False,
            "sr.pysr.populations": 8,
            "sr.pysr.population_size": 50,
            "sr.pysr.ncycles_per_iteration": 200,
            "sr.pysr.niterations": 20,
        },
        "deap": {
            "sr.population": 500,
            "sr.generations": 200,
            "sr.early_stopping.enabled": False,
            "sr.deap.crossover_prob": 0.7,
            "sr.deap.mutation_prob": 0.2,
            "sr.deap.tournament_size": 7,
        },
        "gplearn": {
            "sr.population": 1000,
            "sr.generations": 100,
            "sr.early_stopping.enabled": False,
            "sr.gplearn.const_range": [0.0, 20.0],
            "sr.gplearn.p_crossover": 0.7,
            "sr.gplearn.p_hoist_mutation": 0.05,
            "sr.gplearn.p_point_mutation": 0.1,
            "sr.gplearn.p_subtree_mutation": 0.1,
            "sr.gplearn.tournament_size": 20,
        },
    },
}


def _set(d: dict[str, Any], caminho: str, valor: Any) -> None:
    *pais, folha = caminho.split(".")
    for k in pais:
        d = d.setdefault(k, {})
    d[folha] = valor


def construir_config(
    algoritmo: str,
    configuracao_busca: str | dict[str, Any] = "verificacao",
    semente: int = 0,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Configuração a partir de ``nb01_baseline.yaml`` com os overrides do domínio OTFT."""
    if algoritmo not in ALGORITMOS_RS:
        raise ValueError(f"algoritmo de RS desconhecido: {algoritmo!r}")
    base = yaml.safe_load((repo_physics_llm_sr() / ARQUIVO_CONFIG_BASE).read_text(encoding="utf-8"))
    d = copy.deepcopy(base)
    comuns: dict[str, Any] = {
        "mode": "pilot",
        "track": "extended_analysis",
        "seeds.master": int(semente),
        "loss.w1": PESOS_LLM[0],
        "loss.w2": PESOS_LLM[1],
        "loss.w3": PESOS_LLM[2],
        "loss.baseline_weights": list(PESOS_LINHA_DE_BASE),
        "loss.normalization": "normalized",
        "loss.max_nodes": MAX_NOS,
        "sr.pysr.loss": "mse",
        "sr.pysr.maxsize": MAX_NOS,
        "sr.operators": copy.deepcopy(OPERADORES),
        "output_dir": "results",
    }
    busca = (
        CONFIGURACOES_BUSCA[configuracao_busca][algoritmo]
        if isinstance(configuracao_busca, str)
        else configuracao_busca
    )
    for k, v in {**comuns, **busca, **(overrides or {})}.items():
        _set(d, k, v)
    return parse_config(d)


# ----------------------------------------------------------------------------- métricas


def rmse_log10(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    """RMSE em log10|I_D| (décadas) e fração de predições <= 0 (piso = 1e-6 min(y))."""
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    piso = 1e-6 * float(np.min(y[y > 0]))
    nao_pos = float(np.mean(yhat <= 0))
    r = np.log10(np.maximum(yhat, piso)) - np.log10(np.maximum(y, piso))
    return float(np.sqrt(np.mean(r**2))), nao_pos


def metricas_expressao(
    expr: Expr, nivel: NivelEquacao, X: np.ndarray, y_true: np.ndarray, cfg: ExperimentConfig
) -> dict[str, Any]:
    """Métricas de ajuste (teste limpo) e estruturais de uma expressão.

    Com ``nivel`` na formulação ``log10`` (:func:`otft_sr.formulacao.nivel_formulado`),
    ``y_true`` e a expressão estão em log10(I_D): ``one_minus_r2`` é medido em log10,
    ``rmse_log10`` é o RMSE direto em década e ``one_minus_r2_I_D`` compara 10^f com I_D.
    Na formulação linear ``one_minus_r2_I_D == one_minus_r2``.
    """
    prot = ProtectionParams.from_config(cfg.symbolic)
    yhat = evaluate(expr, X, FEATURES, prot)
    reg = regression_metrics(y_true, yhat)
    if formulacao_do_nivel(nivel) == "log10":
        y_true = np.asarray(y_true, float)
        rl = float(np.sqrt(np.mean((yhat - y_true) ** 2)))
        nao_pos = math.nan
        with np.errstate(over="ignore", invalid="ignore"):
            r2_I_D = regression_metrics(10.0**y_true, 10.0 ** np.clip(yhat, -300, 300))["one_minus_r2"]
    else:
        rl, nao_pos = rmse_log10(y_true, yhat)
        r2_I_D = reg["one_minus_r2"]
    gt = parse_infix(nivel.expressao, FEATURES)
    eq = [parse_infix(f, FEATURES) for f in nivel.formas_equivalentes]
    est = structural_metrics(expr, gt, eq, TreeDistanceParams.from_config(cfg.symbolic))
    return {
        "expressao": to_infix(expr),
        "n_nos": size(expr),
        "one_minus_r2": reg["one_minus_r2"],
        "one_minus_r2_I_D": float(r2_I_D),
        "rmse_log10": rl,
        "frac_pred_nao_positiva": nao_pos,
        **est,
        "forma_funcional_sympy": mesma_forma_funcional(expr, gt),
    }


# ----------------------------------------------------------------------------- execução


def executar_condicao(
    nivel: NivelEquacao,
    algoritmo: str,
    condicao: str,
    ruido: float = 0.0,
    semente: int = 0,
    configuracao_busca: str | dict[str, Any] = "verificacao",
    avaliador: PhysicsEquationEvaluator | None = None,
    n_curvas: int = 20,
    n_pontos: int = 50,
    formulacao_perda: str = "linear",
) -> dict[str, Any]:
    """Executa um ensaio nível x algoritmo de RS x condição x ruído x semente; devolve uma linha.

    ``condicao='A'`` ignora ``avaliador`` (w3 = 0, LLM nunca chamado). ``B``/``C`` exigem
    um avaliador (ex.: :func:`otft_sr.contexto.criar_avaliador` ou
    :func:`otft_sr.provedores_llm.criar_avaliador_m1`). ``formulacao_perda``: ``linear``
    (artigo) ou ``log10`` (alvo log10 I_D, H-OTFT-19).
    """
    validar_formulacao(formulacao_perda)
    if condicao not in CONDICOES:
        raise ValueError(f"condição desconhecida: {condicao!r}")
    if condicao != "A" and avaliador is None:
        raise ValueError(f"condição {condicao} exige avaliador LLM")
    cfg = construir_config(algoritmo, configuracao_busca, semente)
    ds = datasets_formulados(
        construir_datasets(nivel, ruido, semente, n_curvas, n_pontos), formulacao_perda)
    nivel = nivel_formulado(nivel, formulacao_perda)
    ctx = contexto_fisico(nivel)
    objetivo = build_objective(
        cfg,
        avaliador,
        ctx,
        algoritmo,
        baseline=(condicao == "A"),
        feature_names=FEATURES,
        prompt_variant=VARIANTE_PROMPT,
    )
    linha: dict[str, Any] = {
        "nivel": nivel.id,
        "nivel_proposta": nivel.nivel_proposta,
        "algoritmo": algoritmo,
        "condicao": condicao,
        "ruido": ruido,
        "semente": semente,
        "configuracao_busca": (
            configuracao_busca if isinstance(configuracao_busca, str) else "definida_pelo_usuario"
        ),
        "formulacao_perda": formulacao_perda,
        "expressao_referencia": nivel.expressao,
        "n_treino": int(ds["treino"].y_observed.size),
        "n_teste": int(ds["teste"].y_observed.size),
    }
    t0 = time.perf_counter()
    try:
        reg = make_regressor(algoritmo)
        fit = reg.fit(ds["treino"], objetivo, cfg, validation=ds["validacao"])
        linha["tempo_s"] = time.perf_counter() - t0
        linha["stop_reason"] = fit.stop_reason
        linha["n_geracoes"] = fit.n_generations
        linha.update(
            metricas_expressao(fit.best_expression, nivel, ds["teste"].X, ds["teste"].y_true, cfg)
        )
        linha["status"] = "ok"
    except Exception as exc:  # ensaio com erro é registrado, não interrompe o desenho experimental
        linha["tempo_s"] = time.perf_counter() - t0
        linha["status"] = "erro"
        linha["erro"] = f"{type(exc).__name__}: {exc}"
        linha["traceback"] = traceback.format_exc(limit=5)
    cont = objetivo.counters.as_dict()
    linha["chamadas_llm"] = cont["llm_calls"]
    linha["expressoes_unicas_llm"] = cont["unique_llm"]
    linha["cache_hits_llm"] = cont["cache_hits"]
    linha["tempo_llm_s"] = cont["llm_seconds"]
    linha.update(classificar_linha(linha))
    return linha


MODELO_SEM_LLM = "nenhum"


def executar_m1(
    nivel: NivelEquacao,
    algoritmo: str,
    condicao: str,
    ruido: float = 0.0,
    semente: int = 0,
    configuracao_busca: str | dict[str, Any] = "verificacao",
    *,
    cfg_llm: Any = None,
    provedor: Any = None,
    registro_uso: Any = None,
    dir_cache: Path | None = None,
    n_curvas: int = 20,
    n_pontos: int = 50,
    formulacao_perda: str = "linear",
) -> dict[str, Any]:
    """Ensaio M1 com qualquer provedor de ``otft_sr.provedores_llm`` (linha de ``executar_condicao``
    + ``modo``, ``modelo``, tokens e razão tempo LLM / tempo de RS).

    ``A``: sem LLM (``cfg_llm`` deve ser ``None``; ``modelo = "nenhum"``). ``B``/``C``: exige
    ``cfg_llm`` (:class:`~otft_sr.provedores_llm.ConfiguracaoLLM`); ``provedor`` já criado é
    reusado entre ensaios (carrega os pesos uma vez). Tokens = diferença do ``RegistroUso``
    (``registro_uso``, o do provedor ou o global) antes e depois do ensaio; NaN se o provedor
    não conta tokens. ``dir_cache``: diretório do cache JSONL do avaliador (``None`` = padrão
    ``.cache/llm``).
    """
    from otft_sr import provedores_llm as pl

    if condicao == "A":
        if cfg_llm is not None or provedor is not None:
            raise ValueError("condição A não usa LLM: não passe cfg_llm/provedor")
        linha = executar_condicao(nivel, algoritmo, "A", ruido, semente, configuracao_busca,
                                  n_curvas=n_curvas, n_pontos=n_pontos,
                                  formulacao_perda=formulacao_perda)
        modelo, tok_in, tok_out, meta = MODELO_SEM_LLM, math.nan, math.nan, math.nan
    else:
        if not isinstance(cfg_llm, pl.ConfiguracaoLLM):
            raise ValueError(f"condição {condicao} exige cfg_llm (ConfiguracaoLLM)")
        registro = next(r for r in (registro_uso, getattr(provedor, "registro", None),
                                    pl.REGISTRO_USO) if r is not None)
        antes = registro.obter(cfg_llm.rotulo)
        avaliador = pl.criar_avaliador_m1(
            cfg_llm, condicao, provedor=provedor, registro=registro,
            dir_cache=pl.DIR_CACHE_LLM if dir_cache is None else Path(dir_cache))
        linha = executar_condicao(nivel, algoritmo, condicao, ruido, semente, configuracao_busca,
                                  avaliador=avaliador, n_curvas=n_curvas, n_pontos=n_pontos,
                                  formulacao_perda=formulacao_perda)
        depois = registro.obter(cfg_llm.rotulo)
        d_in = depois.tokens_entrada - antes.tokens_entrada
        d_out = depois.tokens_saida - antes.tokens_saida
        tok_in, tok_out = (float(d_in), float(d_out)) if (d_in or d_out) else (math.nan, math.nan)
        modelo = cfg_llm.rotulo
        backend = getattr(avaliador, "backend", None)
        meta_d = backend.model_metadata() if hasattr(backend, "model_metadata") else {}
        meta = json.dumps({**pl._metadados_padrao(cfg_llm, cfg_llm.quantizacao), **meta_d},
                          default=str)
    t_llm = float(linha.get("tempo_llm_s") or 0.0)
    t_rs = float(linha["tempo_s"]) - t_llm
    linha.update(
        modo="M1", modelo=modelo, metadados_modelo=meta, tokens_entrada=tok_in, tokens_saida=tok_out,
        tempo_inferencia_s=t_llm,
        razao_tempo_llm_rs=(t_llm / t_rs) if (condicao != "A" and t_rs > 0) else math.nan,
    )
    return linha


def classificar_linha(linha: dict[str, Any]) -> dict[str, str]:
    """``classe_distancia`` (critério do ``physics_llm_sr``) e ``classe`` (com a equivalência
    sympy, H-OTFT-07b) de uma linha de ensaio; comum aos modos M1 e M2."""
    classe_distancia = classify(
        linha.get("one_minus_r2", math.nan),
        linha.get("tree_distance", math.nan),
        CRITERIO,
        status="ok" if linha.get("status") == "ok" else "error",
        equivalent_distance=linha.get("tree_distance_equivalent_min", math.nan),
        algebraic_distance=linha.get("tree_distance_algebraic", math.nan),
    )
    classe = classe_distancia
    ajusta = linha.get("one_minus_r2", math.inf) <= CRITERIO.max_one_minus_r2
    if classe != "recupera estrutura" and ajusta and linha.get("forma_funcional_sympy"):
        classe = CLASSE_SYMPY
    return {"classe_distancia": classe_distancia, "classe": classe}


def controle_positivo(
    nivel: NivelEquacao, semente: int = 0, algoritmo: str = "pysr", formulacao_perda: str = "linear"
) -> dict[str, Any]:
    """Avalia a expressão de referência pelo pipeline (objetivo + métricas) sem busca.

    Esperado: 1 - R^2 ~ 0, e normalizado ~ 0 e distâncias = 0 (nas duas formulações).
    """
    cfg = construir_config(algoritmo, "verificacao", semente)
    ds = datasets_formulados(construir_datasets(nivel, 0.0, semente), formulacao_perda)
    nivel = nivel_formulado(nivel, formulacao_perda)
    objetivo = build_objective(
        cfg, None, contexto_fisico(nivel), algoritmo, baseline=True, feature_names=FEATURES
    )
    expr = parse_infix(nivel.expressao, FEATURES)
    e, s = objetivo.components(expr, ds["teste"].X, ds["teste"].y_true)
    m = metricas_expressao(expr, nivel, ds["teste"].X, ds["teste"].y_true, cfg)
    return {
        "nivel": nivel.id,
        "nivel_proposta": nivel.nivel_proposta,
        "formulacao_perda": formulacao_perda,
        "e_normalizado": e,
        "s_normalizado": s,
        **m,
    }
