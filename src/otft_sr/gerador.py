"""Modo M2 (exploratório): LLM como GERADOR de equações para os níveis N1-N4.

Condições de geração:

- **G0**: sem contexto físico; o prompt traz só os nomes das variáveis, as unidades e um
  resumo numérico dos dados de TREINO.
- **G1**: G0 + contexto físico estruturado de OTFT (princípios, regimes, C_i, V_Th, W/L,
  mobilidade, unidades e restrições), sem revelar a equação-alvo nem suas constantes.
- **G2**: G1 + refinamento iterativo: em cada uma das ``k`` rodadas, cada linhagem recebe o
  retorno quantitativo (métricas de treino e validação) e físico (checagens) da sua última
  proposta, calculado por ferramentas, e devolve uma versão refinada.

Fluxo de um candidato: prompt modular (``prompts/gerador/*.txt``, hash registrado) ->
resposta JSON (primeiro objeto, schema pydantic) -> interpretação segura por ``parse_infix``
(AST com whitelist; nenhum ``eval``/``exec``/``sympify`` de texto) com limites de
complexidade -> ajuste dos parâmetros livres por ``scipy.optimize.least_squares`` no
TREINO, resíduo em log10|I_D| (proposta §4.4), multi-start determinístico -> métricas de
treino/validação -> checagens físicas automáticas (só treino e grade da região).

Seleção (critério pré-definido, só VALIDAÇÃO): entre os candidatos interpretados e
ajustados, ordem lexicográfica (fisicamente válido primeiro, menor RMSE de log10 I_D na
validação arredondado a 1e-6 década, menor número de nós, ordem de geração). As métricas de
TESTE são calculadas uma única vez, para o selecionado, com ``experimento.metricas_expressao``
(as mesmas do M1). O teste nunca entra em prompt, ajuste, checagem, seleção ou retorno.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from physics_llm_sr.core_types import Dataset
from physics_llm_sr.evaluation.metrics import regression_metrics
from physics_llm_sr.symbolic.expression import (
    Const,
    Expr,
    ExpressionError,
    Op,
    UnsafeExpressionError,
    Var,
    depth,
    evaluate,
    parse_infix,
    protected_points,
    size,
    to_infix,
)
from physics_llm_sr.symbolic.expression import variables as variaveis_da_expressao
from physics_llm_sr.symbolic.operators import OPERATORS, ProtectionParams

from otft_sr.contexto import contexto_fisico
from otft_sr.dados import construir_datasets, hash_datasets
from otft_sr.experimento import (
    COLUNAS_M1,
    MAX_NOS,
    OPERADORES,
    classificar_linha,
    construir_config,
    metricas_expressao,
    rmse_log10,
)
from otft_sr.niveis import FEATURES, NivelEquacao, _grade_regiao
from otft_sr.ovsed import RAIZ_PROJETO
from otft_sr.provedores_llm import (
    ConfiguracaoLLM,
    RegistroUso,
    criar_provedor,
    fechar_provedor,
)

CONDICOES_GERACAO: tuple[str, ...] = ("G0", "G1", "G2")
ALGORITMO_M2 = "llm_gerador"
MODO_M2 = "M2"
DIR_PROMPTS_GERADOR = RAIZ_PROJETO / "prompts" / "gerador"
SECOES_PROMPT: tuple[str, ...] = (
    "sistema", "definicao_problema", "variaveis", "contexto_fisico", "restricoes",
    "resumo_dados", "instrucoes", "formato_resposta", "feedback", "correcao",
)

#: Operações permitidas: as mesmas do PySR no M1 (``experimento.OPERADORES["pysr"]``).
OPERACOES_PERMITIDAS: frozenset[str] = frozenset(OPERADORES["pysr"])
#: Limites de complexidade: nós = ``loss.max_nodes``/``sr.pysr.maxsize`` do M1 (H-OTFT-07).
MAX_NOS_GERADOR = MAX_NOS
MAX_PROFUNDIDADE = 8
MAX_PARAMETROS = 6
MAX_CARACTERES_EQUACAO = 400
MAX_CARACTERES_RESPOSTA = 20_000
N_INICIOS_PADRAO = 8
TOLERANCIA_EMPATE_RMSE = 6  # casas decimais (década) no critério de seleção
_NOME_PARAMETRO = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")

# Limiares das checagens físicas (fração da corrente máxima na região, salvo indicação).
LIMIAR_ANULACAO = 0.05
LIMIAR_FUGA = 0.25  # |I(V_GS mínimo)/I_leak estimado - 1|
LIMIAR_SATURACAO_V_DS = 0.10  # variação relativa de I_D na faixa de V_DS em saturação
LIMIAR_SATURACAO_CONSTRICAO = 0.20  # (dI/dV_DS na constrição)/(dI/dV_DS perto de V_DS = 0)
TOLERANCIA_MONOTONIA = 1e-9

#: Checagem de limiar por nível: N1/N3 (saturação) I_D -> ~0 em V_GS -> V_Th; N2 (triodo)
#: I_D -> 0 em V_DS -> 0; N4 (sublimiar com fuga) I_D -> I_leak em V_GS baixo.
CHECAGEM_LIMIAR: dict[str, str] = {
    "N1": "anulacao_no_limiar",
    "N2": "anulacao_em_V_DS_nulo",
    "N3": "anulacao_no_limiar",
    "N4": "patamar_de_fuga",
}

_DESCRICAO_CHECAGEM_EN: dict[str, str] = {
    "positiva_finita": "I_D finite and positive over the measurement region",
    "monotonica_V_GS": "I_D non-decreasing in V_GS",
    "anulacao_no_limiar": "I_D close to zero at the threshold voltage extracted from training data",
    "anulacao_em_V_DS_nulo": "I_D vanishes at V_DS = 0",
    "patamar_de_fuga": "I_D tends to the leakage floor at the lowest V_GS",
    "saturacao_V_DS": "I_D nearly independent of V_DS in saturation",
    "saturacao_na_constricao": "dI_D/dV_DS vanishes at pinch-off (V_DS = V_GS - V_Th)",
    "dimensional": "arguments of exp/log/sin/cos and exponents are scaled (qualitative)",
}

_UNIDADES_EN = {"uA": "uA", "pA": "pA"}


# --------------------------------------------------------------------------- templates


def carregar_templates(diretorio: Path | None = None) -> dict[str, str]:
    """Templates de prompt ``<seção>.txt`` de ``prompts/gerador`` (todas as seções)."""
    d = Path(diretorio) if diretorio is not None else DIR_PROMPTS_GERADOR
    t = {s: (d / f"{s}.txt").read_text(encoding="utf-8") for s in SECOES_PROMPT}
    return t


def _sha256(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def hash_templates(templates: Mapping[str, str]) -> dict[str, Any]:
    """SHA-256 de cada template e do conjunto (nome + hash, em ordem de nome)."""
    arquivos = {k: _sha256(v) for k, v in sorted(templates.items())}
    conjunto = _sha256("\n".join(f"{k}:{h}" for k, h in arquivos.items()))
    return {"arquivos": arquivos, "conjunto": conjunto}


# ------------------------------------------------------------------------ resumo e prompt


def _fmt(x: float) -> str:
    return f"{x:.4g}" if np.isfinite(x) else "undefined"


def resumo_dados(treino: Dataset) -> dict[str, str]:
    """Resumo numérico das variáveis do TREINO (faixas, correlações de Spearman, amostra)."""
    X = np.asarray(treino.X, float)
    y = np.asarray(treino.y_observed, float)
    linhas = []
    for i, nome in enumerate(FEATURES):
        col = X[:, i]
        linhas.append(
            f"- {nome} [V]: min {_fmt(col.min())}, max {_fmt(col.max())}, "
            f"mean {_fmt(col.mean())}, distinct values {np.unique(col).size}"
        )
    linhas.append(
        f"- I_D: min {_fmt(y.min())}, median {_fmt(float(np.median(y)))}, max {_fmt(y.max())}"
    )
    corr = []
    for i, nome in enumerate(FEATURES):
        if np.unique(X[:, i]).size < 2:
            corr.append(f"{nome}: undefined (constant)")
            continue
        rho = float(spearmanr(X[:, i], y).statistic)
        corr.append(f"{nome}: {rho:+.3f}")
    pos = y[y > 0]
    decadas = math.log10(pos.max() / pos.min()) if pos.size else math.nan
    ordem = np.lexsort((X[:, 0], X[:, 1]))
    idx = ordem[np.unique(np.linspace(0, len(ordem) - 1, 12).round().astype(int))]
    amostra = "\n".join(f"{_fmt(X[k, 0])}, {_fmt(X[k, 1])}, {_fmt(y[k])}" for k in idx)
    return {
        "n_pontos": str(y.size),
        "n_curvas": str(np.unique(treino.simulation_ids).size),
        "tabela_variaveis": "\n".join(linhas),
        "correlacoes": "; ".join(corr),
        "decadas": f"{decadas:.2f}",
        "amostra": amostra,
    }


def montar_prompt(
    nivel: NivelEquacao,
    condicao: str,
    treino: Dataset,
    indice: int,
    total: int,
    anteriores: Sequence[str] = (),
    feedback: str | None = None,
    templates: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """``(system, user)`` da condição. Usa só o TREINO; ``feedback`` (G2) já vem pronto."""
    if condicao not in CONDICOES_GERACAO:
        raise ValueError(f"condição de geração desconhecida: {condicao!r} ({CONDICOES_GERACAO})")
    t = templates if templates is not None else carregar_templates()
    unidade = _UNIDADES_EN[nivel.unidades["I_D"]]
    fisico = condicao in ("G1", "G2")
    partes = [
        t["definicao_problema"],
        Template(t["variaveis"]).substitute(unidade_I_D=unidade),
    ]
    if fisico:
        partes.append(
            Template(t["contexto_fisico"]).substitute(
                unidade_I_D=unidade,
                descricao_experimento=contexto_fisico(nivel).experiment_description,
            )
        )
    partes.append(
        Template(t["restricoes"]).substitute(
            variaveis=", ".join(FEATURES),
            funcoes=", ".join(sorted(OPERACOES_PERMITIDAS - {"add", "sub", "mul", "div", "pow", "neg"})),
            max_parametros=MAX_PARAMETROS,
            max_nos=MAX_NOS_GERADOR,
            max_profundidade=MAX_PROFUNDIDADE,
        )
    )
    partes.append(Template(t["resumo_dados"]).substitute(resumo_dados(treino)))
    if feedback is not None:
        partes.append(feedback)
    texto_anteriores = ""
    if anteriores:
        lista = "; ".join(f"({i + 1}) {e}" for i, e in enumerate(anteriores))
        texto_anteriores = f"Equations already proposed (propose a different one): {lista}. "
    partes.append(
        Template(t["instrucoes"]).substitute(
            indice=indice,
            total=total,
            anteriores=texto_anteriores,
            complemento_fisico=" and with the physical context" if fisico else "",
        )
    )
    partes.append(t["formato_resposta"])
    return t["sistema"].strip(), "\n\n".join(p.strip() for p in partes)


# ------------------------------------------------------------------- resposta e schema


class RespostaInvalida(ValueError):
    """Resposta do LLM recusada; ``categoria`` classifica o motivo para o registro."""

    def __init__(self, categoria: str, mensagem: str) -> None:
        """``categoria``: json_ausente, schema, inseguro, sintaxe, complexidade,
        operacao_nao_permitida, sem_variavel, ajuste ou erro_provedor."""
        super().__init__(f"{categoria}: {mensagem}")
        self.categoria = categoria
        self.mensagem = mensagem


def _numero_ou_none(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, str):
        try:
            x = float(v)
        except ValueError:
            return None
        return x if math.isfinite(x) else None
    return None


def _validar_nome_parametro(nome: str) -> None:
    if not isinstance(nome, str) or not _NOME_PARAMETRO.match(nome):
        raise ValueError(f"nome de parâmetro inválido: {nome!r}")
    if nome in FEATURES or nome in OPERATORS:
        raise ValueError(f"nome de parâmetro colide com variável/operação: {nome!r}")


class RespostaGerador(BaseModel):
    """Schema da resposta JSON do LLM gerador (chaves extras são ignoradas).

    Obrigatórias: ``equation``, ``variables``, ``parameters``. ``parameters`` aceita dicionário
    nome -> chute inicial (número; texto/nulo viram ``None``), lista de nomes ou lista de
    objetos ``{"name", "initial_value"|"value"|"initial_guess"}``.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    equation: str = Field(min_length=1, max_length=MAX_CARACTERES_EQUACAO)
    variables: list[str]
    parameters: dict[str, float | None]
    assumptions: list[str] = Field(default_factory=list)
    physical_constraints: list[str] = Field(default_factory=list)

    @field_validator("variables")
    @classmethod
    def _variaveis(cls, v: list[str]) -> list[str]:
        fora = [x for x in v if x not in FEATURES]
        if fora:
            raise ValueError(f"variáveis fora de {FEATURES}: {fora}")
        return v

    @field_validator("parameters", mode="before")
    @classmethod
    def _parametros(cls, v: Any) -> dict[str, float | None]:
        if isinstance(v, Mapping):
            itens = [(k, _numero_ou_none(x)) for k, x in v.items()]
        elif isinstance(v, (list, tuple)):
            itens = []
            for x in v:
                if isinstance(x, str):
                    itens.append((x, None))
                elif isinstance(x, Mapping) and "name" in x:
                    chute = next(
                        (x[k] for k in ("initial_value", "value", "initial_guess") if k in x), None
                    )
                    itens.append((x["name"], _numero_ou_none(chute)))
                else:
                    raise ValueError(f"item de parâmetro não reconhecido: {x!r}")
        else:
            raise ValueError("parameters deve ser objeto ou lista")
        if len(itens) > MAX_PARAMETROS:
            raise ValueError(f"mais de {MAX_PARAMETROS} parâmetros")
        for nome, _ in itens:
            _validar_nome_parametro(nome)
        return dict(itens)

    @field_validator("assumptions", "physical_constraints", mode="before")
    @classmethod
    def _textos(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, (list, tuple)):
            return [x if isinstance(x, str) else json.dumps(x, default=str) for x in v]
        return [json.dumps(v, default=str)]


def extrair_primeiro_json(texto: str) -> dict[str, Any]:
    """Primeiro objeto JSON (dicionário) decodificável no texto da resposta."""
    if not isinstance(texto, str):
        raise RespostaInvalida("json_ausente", "resposta não é texto")
    texto = texto[:MAX_CARACTERES_RESPOSTA]
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", texto):
        try:
            obj, _ = dec.raw_decode(texto, m.start())
        except (json.JSONDecodeError, RecursionError):
            continue
        if isinstance(obj, dict):
            return obj
    raise RespostaInvalida("json_ausente", "nenhum objeto JSON na resposta")


def validar_resposta(texto: str) -> RespostaGerador:
    """Extrai o primeiro objeto JSON e valida o schema (``RespostaInvalida`` se falhar)."""
    obj = extrair_primeiro_json(texto)
    try:
        return RespostaGerador.model_validate(obj)
    except ValidationError as exc:
        erros = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]
        )
        raise RespostaInvalida("schema", erros) from None


# ------------------------------------------------------------------------ interpretação


@dataclass(frozen=True)
class EquacaoInterpretada:
    """Equação lida com segurança: parâmetros livres como ``Var`` da AST."""

    texto_original: str
    texto_normalizado: str
    expr: Expr
    parametros: tuple[str, ...]
    chutes: dict[str, float | None]
    parametros_inferidos: tuple[str, ...]
    parametros_nao_usados: tuple[str, ...]


_RE_LADO_ESQUERDO = re.compile(r"^\s*I_?D\s*(?:\([^)]*\))?\s*=(?!=)\s*")


def normalizar_texto(texto: str) -> str:
    """Normalização textual determinística antes do parser: remove ``I_D =``, ``^`` -> ``**``,
    ``ln(`` -> ``log(``, sinais tipográficos (− · ×) -> ASCII."""
    t = _RE_LADO_ESQUERDO.sub("", texto.strip(), count=1)
    t = t.replace("−", "-").replace("·", "*").replace("×", "*").replace("^", "**")
    return re.sub(r"\bln\s*\(", "log(", t)


def _ordem_variaveis(expr: Expr) -> list[str]:
    out: list[str] = []

    def rec(e: Expr) -> None:
        if isinstance(e, Var) and e.name not in out:
            out.append(e.name)
        elif isinstance(e, Op):
            for c in e.children:
                rec(c)

    rec(expr)
    return out


def _operacoes(expr: Expr) -> set[str]:
    if isinstance(expr, Op):
        return {expr.name}.union(*(_operacoes(c) for c in expr.children))
    return set()


def interpretar_equacao(
    texto: str, parametros_declarados: Mapping[str, float | None]
) -> EquacaoInterpretada:
    """Lê a equação com ``parse_infix`` (whitelist de AST, nada é executado) e aplica os
    limites: operações permitidas, nós, profundidade, número e nomes de parâmetros, ao menos
    uma variável de entrada. Nomes livres não declarados viram parâmetros inferidos."""
    if not isinstance(texto, str) or len(texto) > MAX_CARACTERES_EQUACAO:
        raise RespostaInvalida("schema", f"equação ausente ou com mais de {MAX_CARACTERES_EQUACAO} caracteres")
    t = normalizar_texto(texto)
    try:
        expr = parse_infix(t, None)
    except UnsafeExpressionError as exc:
        raise RespostaInvalida("inseguro", str(exc)) from None
    except (ExpressionError, ValueError) as exc:
        raise RespostaInvalida("sintaxe", str(exc)) from None
    proibidas = _operacoes(expr) - OPERACOES_PERMITIDAS
    if proibidas:
        raise RespostaInvalida("operacao_nao_permitida", f"fora da lista permitida: {sorted(proibidas)}")
    usados = _ordem_variaveis(expr)
    if not any(v in FEATURES for v in usados):
        raise RespostaInvalida("sem_variavel", f"a equação não usa {FEATURES}")
    livres = [v for v in usados if v not in FEATURES]
    for nome in livres:
        try:
            _validar_nome_parametro(nome)
        except ValueError as exc:
            raise RespostaInvalida("schema", str(exc)) from None
    declarados = [p for p in parametros_declarados if p in livres]
    inferidos = [p for p in livres if p not in parametros_declarados]
    parametros = tuple(declarados + inferidos)
    if len(parametros) > MAX_PARAMETROS:
        raise RespostaInvalida("complexidade", f"{len(parametros)} parâmetros > {MAX_PARAMETROS}")
    n, h = size(expr), depth(expr)
    if n > MAX_NOS_GERADOR or h > MAX_PROFUNDIDADE:
        raise RespostaInvalida(
            "complexidade",
            f"{n} nós (máx. {MAX_NOS_GERADOR}), profundidade {h} (máx. {MAX_PROFUNDIDADE})",
        )
    return EquacaoInterpretada(
        texto_original=texto,
        texto_normalizado=t,
        expr=expr,
        parametros=parametros,
        chutes={p: parametros_declarados.get(p) for p in parametros},
        parametros_inferidos=tuple(inferidos),
        parametros_nao_usados=tuple(p for p in parametros_declarados if p not in livres),
    )


# ------------------------------------------------------------------------------- ajuste


def substituir_parametros(expr: Expr, valores: Mapping[str, float]) -> Expr:
    """Troca cada ``Var`` de parâmetro pela constante ajustada."""
    if isinstance(expr, Var) and expr.name in valores:
        return Const(valores[expr.name])
    if isinstance(expr, Op):
        return Op(expr.name, tuple(substituir_parametros(c, valores) for c in expr.children))
    return expr


@dataclass(frozen=True)
class ResultadoAjuste:
    """Ajuste dos parâmetros livres (melhor dos inícios) e registro da convergência."""

    valores: dict[str, float]
    expr_final: Expr | None
    convergiu: bool
    custo: float
    n_inicios: int
    n_sucesso: int
    n_no_minimo: int
    nfev: int
    mensagem: str


_PROTECAO_PADRAO: ProtectionParams | None = None


def protecao_padrao() -> ProtectionParams:
    """Semântica protegida da configuração do M1 (a mesma de ``metricas_expressao``)."""
    global _PROTECAO_PADRAO
    if _PROTECAO_PADRAO is None:
        _PROTECAO_PADRAO = ProtectionParams.from_config(
            construir_config("pysr", "verificacao", 0).symbolic
        )
    return _PROTECAO_PADRAO


def ajustar_parametros(
    interp: EquacaoInterpretada,
    X: np.ndarray,
    y: np.ndarray,
    semente: int = 0,
    n_inicios: int = N_INICIOS_PADRAO,
    protecao: ProtectionParams | None = None,
) -> ResultadoAjuste:
    """Mínimos quadrados não lineares em log10|I_D| (``least_squares``, TRF, ``x_scale='jac'``)
    com multi-start determinístico.

    Inícios, na ordem: chutes do LLM (``None`` -> 1), todos 1, e sorteios log-uniformes de
    módulo em [1e-3, 1e2] (sinal negativo com probabilidade 0,2) de um gerador semeado por
    ``(semente, hash do texto normalizado)``. Pontos de treino com I_D <= 0 são descartados.
    Predições abaixo do piso (1e-6 min I_D, o mesmo de ``rmse_log10``) recebem penalidade
    contínua -log10(1 + (piso - ŷ)/piso) somada ao resíduo (já negativo), para manter gradiente
    sem criar mínimo espelhado em ŷ = -y.
    """
    prot = protecao or protecao_padrao()
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    m = y > 0
    X, y = X[m], y[m]
    ly = np.log10(y)
    piso = 1e-6 * float(y.min())
    nomes = (*FEATURES, *interp.parametros)
    p = len(interp.parametros)

    def prever(theta: np.ndarray) -> np.ndarray:
        Xe = np.column_stack([X, np.broadcast_to(theta, (X.shape[0], p))])
        return evaluate(interp.expr, Xe, nomes, prot)

    def residuo(theta: np.ndarray) -> np.ndarray:
        yh = prever(theta)
        r = np.log10(np.maximum(yh, piso)) - ly
        abaixo = yh < piso
        if abaixo.any():  # r < 0 nesses pontos: a penalidade aumenta |r| (sem mínimo espelhado)
            r = r - np.log10(1.0 + np.maximum(piso - yh, 0.0) / piso)
        return r

    if p == 0:
        custo = 0.5 * float(np.sum(residuo(np.zeros(0)) ** 2))
        return ResultadoAjuste({}, interp.expr, True, custo, 0, 0, 0, 0, "sem parâmetros livres")

    rng = np.random.default_rng(
        np.random.SeedSequence([int(semente), int(_sha256(interp.texto_normalizado)[:8], 16)])
    )
    inicios: list[np.ndarray] = []
    if any(v is not None for v in interp.chutes.values()):
        inicios.append(np.array([1.0 if interp.chutes[n] is None else interp.chutes[n]
                                 for n in interp.parametros], float))
    inicios.append(np.ones(p))
    while len(inicios) < n_inicios:
        mod = 10.0 ** rng.uniform(-3.0, 2.0, p)
        inicios.append(np.where(rng.random(p) < 0.2, -mod, mod))
    inicios = inicios[:n_inicios]

    resultados = []
    nfev = 0
    for x0 in inicios:
        try:
            r = least_squares(
                residuo, x0, method="trf", x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12,
                max_nfev=300 * (p + 1),
            )
        except Exception:  # noqa: BLE001 - início sem solução numérica é só descartado
            continue
        nfev += int(r.nfev)
        if np.all(np.isfinite(r.x)) and np.isfinite(r.cost):
            resultados.append(r)
    if not resultados:
        return ResultadoAjuste({}, None, False, math.nan, len(inicios), 0, 0, nfev,
                               "nenhum início produziu solução finita")
    melhor = min(resultados, key=lambda r: r.cost)
    no_minimo = sum(r.cost <= melhor.cost * (1 + 1e-6) + 1e-12 for r in resultados)
    valores = {n: float(v) for n, v in zip(interp.parametros, melhor.x, strict=True)}
    try:
        expr_final = substituir_parametros(interp.expr, valores)
    except ExpressionError as exc:
        return ResultadoAjuste(valores, None, False, float(melhor.cost), len(inicios),
                               sum(r.success for r in resultados), no_minimo, nfev, str(exc))
    return ResultadoAjuste(
        valores=valores,
        expr_final=expr_final,
        convergiu=bool(melhor.success),
        custo=float(melhor.cost),
        n_inicios=len(inicios),
        n_sucesso=int(sum(r.success for r in resultados)),
        n_no_minimo=int(no_minimo),
        nfev=nfev,
        mensagem=str(melhor.message),
    )


# --------------------------------------------------------------------- checagens físicas


@dataclass(frozen=True)
class Checagem:
    """Resultado de uma checagem física: ``aprovado`` = None quando não se aplica."""

    nome: str
    aprovado: bool | None
    valor: float
    limite: float
    obrigatoria: bool
    nota: str = ""


def _prever_vg_vd(expr: Expr, vg: np.ndarray, vd: np.ndarray, prot: ProtectionParams):
    X = np.column_stack([np.asarray(vg, float), np.asarray(vd, float)])
    return evaluate(expr, X, FEATURES, prot), protected_points(expr, X, FEATURES, prot)


def _V_Th_raiz(treino: Dataset) -> float:
    """V_Th pela extrapolação linear de sqrt(I_D) x V_GS (método clássico de saturação)."""
    y = treino.y_observed
    m = y > 0
    a, b = np.polyfit(treino.X[m, 0], np.sqrt(y[m]), 1)
    return float(-b / a) if a > 0 else math.nan


def _V_Th_linear(treino: Dataset) -> float:
    """V_Th no triodo: regressão de I_D/V_DS em (V_GS, V_DS, 1) no quartil inferior de V_DS
    (extrapolação na região linear com correção do termo em V_DS)."""
    vg, vd, y = treino.X[:, 0], treino.X[:, 1], treino.y_observed
    m = (vd <= np.quantile(vd, 0.25)) & (vd > 0)
    A = np.column_stack([vg[m], vd[m], np.ones(m.sum())])
    a, _, c = np.linalg.lstsq(A, y[m] / vd[m], rcond=None)[0]
    return float(-c / a) if a > 0 else math.nan


def _checagem_dimensional(expr: Expr) -> Checagem:
    violacoes = []

    def tem_variavel(e: Expr) -> bool:
        return bool(variaveis_da_expressao(e) & set(FEATURES))

    def tem_escala(e: Expr) -> bool:
        if isinstance(e, Const):
            return True
        if isinstance(e, Var):
            return e.name not in FEATURES
        return any(tem_escala(c) for c in e.children)  # type: ignore[union-attr]

    def rec(e: Expr) -> None:
        if not isinstance(e, Op):
            return
        if e.name in ("exp", "log", "sin", "cos"):
            arg = e.children[0]
            if tem_variavel(arg) and not tem_escala(arg):
                violacoes.append(to_infix(e))
        if e.name == "pow" and tem_variavel(e.children[1]) and not tem_escala(e.children[1]):
            violacoes.append(to_infix(e))
        for c in e.children:
            rec(c)

    rec(expr)
    return Checagem(
        "dimensional",
        not violacoes,
        float(len(violacoes)),
        0.0,
        False,
        "verificação qualitativa (informativa): as unidades dos parâmetros livres não são "
        "declaradas, logo a homogeneidade dimensional não é verificável automaticamente; marca "
        "só argumentos de exp/log/sin/cos e expoentes com tensão sem fator de escala"
        + (f": {violacoes}" if violacoes else ""),
    )


def checagens_fisicas(
    expr: Expr,
    nivel: NivelEquacao,
    treino: Dataset,
    protecao: ProtectionParams | None = None,
) -> list[Checagem]:
    """Checagens automáticas do nível sobre a expressão final (constantes ajustadas).

    Usa a grade densa da região do nível e estimativas feitas só com o TREINO (V_Th por
    extrapolação, I_leak pela mediana no V_GS mais baixo). Obrigatórias: positiva_finita,
    monotonica_V_GS, a de limiar do nível (:data:`CHECAGEM_LIMIAR`) e a de saturação;
    ``dimensional`` é informativa (limitação declarada).
    """
    prot = protecao or protecao_padrao()
    r = nivel.regiao
    out: list[Checagem] = []

    vg, vd = _grade_regiao(r)
    y, protegido = _prever_vg_vd(expr, vg, vd, prot)
    ok = np.isfinite(y) & (y > 0) & ~protegido
    frac = float(np.mean(ok))
    out.append(Checagem("positiva_finita", frac == 1.0, frac, 1.0, True,
                        "fração da grade da região com I_D finito, > 0 e sem proteção numérica"))
    escala = float(np.max(np.abs(y))) or 1.0

    vgs = np.linspace(*r.V_GS, 81)
    hi = r.V_DS[1] if r.V_DS_max_rel_V_GS is None else min(r.V_DS[1], r.V_GS[0] - r.V_DS_max_rel_V_GS)
    pior = math.inf
    for v in np.linspace(r.V_DS[0], hi, 5):
        yy, _ = _prever_vg_vd(expr, vgs, np.full_like(vgs, v), prot)
        pior = min(pior, float(np.min(np.diff(yy))) / (float(np.max(np.abs(yy))) or 1.0))
    out.append(Checagem("monotonica_V_GS", pior >= -TOLERANCIA_MONOTONIA, pior,
                        -TOLERANCIA_MONOTONIA, True,
                        "menor passo relativo de I_D ao longo de V_GS (V_DS fixo)"))

    tipo = CHECAGEM_LIMIAR.get(nivel.id, "anulacao_no_limiar")
    vd_meio = 0.5 * (r.V_DS[0] + hi)
    if tipo == "anulacao_no_limiar":
        vt = _V_Th_raiz(treino)
        if math.isfinite(vt):
            yv, _ = _prever_vg_vd(expr, np.array([vt]), np.array([vd_meio]), prot)
            valor = abs(float(yv[0])) / escala
            out.append(Checagem(tipo, valor <= LIMIAR_ANULACAO, valor, LIMIAR_ANULACAO, True,
                                f"|I_D(V_Th estimado = {vt:.4g} V)| / máx. na região; V_Th por "
                                "extrapolação de sqrt(I_D) no treino"))
        else:
            out.append(Checagem(tipo, None, math.nan, LIMIAR_ANULACAO, True,
                                "V_Th não estimável no treino"))
    elif tipo == "anulacao_em_V_DS_nulo":
        g = np.linspace(*r.V_GS, 6)
        yv, _ = _prever_vg_vd(expr, g, np.zeros_like(g), prot)
        valor = float(np.max(np.abs(yv))) / escala
        out.append(Checagem(tipo, valor <= LIMIAR_ANULACAO, valor, LIMIAR_ANULACAO, True,
                            "máx. |I_D(V_DS = 0)| / máx. na região"))
    elif tipo == "patamar_de_fuga":
        vgt, yt = treino.X[:, 0], treino.y_observed
        lo, hi_t = float(vgt.min()), float(vgt.max())
        I_leak = float(np.median(yt[vgt <= lo + 0.2 * (hi_t - lo)]))
        yv, _ = _prever_vg_vd(expr, np.array([r.V_GS[0]]), np.array([vd_meio]), prot)
        valor = float(yv[0]) / I_leak if I_leak > 0 else math.nan
        aprovado = bool(math.isfinite(valor) and abs(valor - 1.0) <= LIMIAR_FUGA)
        out.append(Checagem(tipo, aprovado, valor, LIMIAR_FUGA, True,
                            f"I_D(V_GS = {r.V_GS[0]:g} V) / I_leak estimado ({I_leak:.4g}, "
                            "mediana do treino nos 20 % mais baixos de V_GS); limite em |razão - 1|"))

    if r.V_DS_max_rel_V_GS is not None:  # triodo: saturação na constrição
        vt = _V_Th_linear(treino)
        razoes = []
        if math.isfinite(vt):
            for g in np.linspace(*r.V_GS, 6):
                vsat = g - vt
                if vsat <= r.V_DS[0] + 1e-3:
                    continue
                h = 1e-3 * max(1.0, vsat)
                pts_vd = np.array([vsat - h, vsat + h, r.V_DS[0], r.V_DS[0] + 2 * h])
                yy, _ = _prever_vg_vd(expr, np.full(4, g), pts_vd, prot)
                d_sat = (yy[1] - yy[0]) / (2 * h)
                d_0 = (yy[3] - yy[2]) / (2 * h)
                razoes.append(abs(d_sat) / abs(d_0) if d_0 != 0 else math.inf)
        if razoes:
            valor = float(max(razoes))
            out.append(Checagem("saturacao_na_constricao", valor <= LIMIAR_SATURACAO_CONSTRICAO,
                                valor, LIMIAR_SATURACAO_CONSTRICAO, True,
                                f"|dI/dV_DS| em V_DS = V_GS - V_Th (V_Th estimado = {vt:.4g} V, "
                                "regressão no triodo do treino) relativo ao valor perto de V_DS = 0"))
        else:
            out.append(Checagem("saturacao_na_constricao", None, math.nan,
                                LIMIAR_SATURACAO_CONSTRICAO, True, "V_Th não estimável no treino"))
    else:  # região inteira em saturação (V_DS > V_GS - V_Th): I_D independente de V_DS
        g = np.linspace(*r.V_GS, 41)
        y_lo, _ = _prever_vg_vd(expr, g, np.full_like(g, r.V_DS[0]), prot)
        y_hi, _ = _prever_vg_vd(expr, g, np.full_like(g, r.V_DS[1]), prot)
        y_md, _ = _prever_vg_vd(expr, g, np.full_like(g, 0.5 * sum(r.V_DS)), prot)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(y_hi - y_lo) / np.abs(y_md)
        valor = float(np.nanmax(np.where(np.isfinite(rel), rel, np.inf)))
        out.append(Checagem("saturacao_V_DS", valor <= LIMIAR_SATURACAO_V_DS, valor,
                            LIMIAR_SATURACAO_V_DS, True,
                            f"máx. |I(V_DS = {r.V_DS[1]:g}) - I(V_DS = {r.V_DS[0]:g})| / I(meio)"))

    out.append(_checagem_dimensional(expr))
    return out


def fisicamente_valido(checagens: Sequence[Checagem]) -> bool:
    """Todas as checagens obrigatórias aprovadas (``None`` = não se aplica, não reprova)."""
    return all(c.aprovado is not False for c in checagens if c.obrigatoria)


# ------------------------------------------------------------------------------- seleção


def selecionar_melhor(candidatos: Sequence[Mapping[str, Any]]) -> int | None:
    """Índice do melhor candidato pelo critério pré-definido (só VALIDAÇÃO):
    (fisicamente válido primeiro, RMSE de log10 I_D na validação arredondado a 1e-6 década,
    número de nós, ordem). ``None`` se nenhum foi interpretado e ajustado."""
    elegiveis = []
    for i, c in enumerate(candidatos):
        rmse = c.get("rmse_log10_validacao", math.nan)
        if c.get("status") != "valido" or rmse is None or not math.isfinite(rmse):
            continue
        chave = (not c.get("fisicamente_valido", False),
                 round(float(rmse), TOLERANCIA_EMPATE_RMSE), c.get("n_nos", math.inf), i)
        elegiveis.append((chave, i))
    return min(elegiveis)[1] if elegiveis else None


# ---------------------------------------------------------------------------- feedback G2


def texto_feedback(candidato: Mapping[str, Any], templates: Mapping[str, str]) -> str:
    """Retorno (G2) de um candidato: só métricas de treino/validação e checagens."""
    if candidato.get("status") != "valido":
        status = f"rejected ({candidato.get('categoria_erro')}: {candidato.get('mensagem_erro')})"
        params, rt, rv, r2v, ch = "none", "n/a", "n/a", "n/a", "not evaluated"
    else:
        status = ("passes all mandatory physical checks" if candidato.get("fisicamente_valido")
                  else "fails mandatory physical checks")
        params = json.dumps({k: float(f"{v:.6g}") for k, v in candidato["parametros_ajustados"].items()})
        rt = _fmt(candidato["rmse_log10_treino"])
        rv = _fmt(candidato["rmse_log10_validacao"])
        r2v = _fmt(candidato["one_minus_r2_validacao"])
        itens = []
        for c in candidato["checagens"]:
            if c["aprovado"] is None:
                res = "not applicable"
            else:
                res = "passed" if c["aprovado"] else "FAILED"
            tipo = "" if c["obrigatoria"] else " [informative]"
            itens.append(f"{_DESCRICAO_CHECAGEM_EN.get(c['nome'], c['nome'])}: {res}{tipo} "
                         f"(value {_fmt(c['valor'])}, limit {_fmt(c['limite'])})")
        ch = "; ".join(itens)
    eq = candidato.get("equacao_normalizada") or candidato.get("equacao_proposta") or "none"
    return Template(templates["feedback"]).substitute(
        equacao=eq, status=status, parametros=params, rmse_treino=rt, rmse_validacao=rv,
        r2_validacao=r2v, checagens=ch,
    )


# ------------------------------------------------------------------------------ execução


def _resolver_provedor(provedor: Any, registro: RegistroUso | None):
    """(backend, cfg | None, rótulo do modelo, registro | None, criado_aqui)."""
    if isinstance(provedor, ConfiguracaoLLM):
        registro = registro if registro is not None else RegistroUso()
        backend = criar_provedor(provedor, registro=registro)
        if provedor.provedor == "ollama":
            backend.resolve_identity()  # type: ignore[attr-defined]
        return backend, provedor, provedor.rotulo, registro, True
    cfg = getattr(provedor, "cfg", None)
    if isinstance(cfg, ConfiguracaoLLM):
        return provedor, cfg, cfg.rotulo, getattr(provedor, "registro", None), False
    meta = provedor.model_metadata() if hasattr(provedor, "model_metadata") else {}
    rotulo = f"{meta.get('provedor') or meta.get('backend', 'desconhecido')}/" \
             f"{meta.get('modelo') or meta.get('checkpoint', 'desconhecido')}"
    return provedor, None, rotulo, None, False


def _metricas_particao(expr: Expr, ds: Dataset, prot: ProtectionParams) -> tuple[float, float]:
    yhat = evaluate(expr, ds.X, FEATURES, prot)
    r, _ = rmse_log10(ds.y_observed, yhat)
    return r, float(regression_metrics(ds.y_observed, yhat)["one_minus_r2"])


def _avaliar_candidato(
    interp: EquacaoInterpretada,
    nivel: NivelEquacao,
    ds: Mapping[str, Dataset],
    semente: int,
    n_inicios: int,
    prot: ProtectionParams,
) -> dict[str, Any]:
    """Ajuste no treino, métricas de treino/validação e checagens (nunca o teste)."""
    aj = ajustar_parametros(interp, ds["treino"].X, ds["treino"].y_observed, semente, n_inicios, prot)
    reg: dict[str, Any] = {
        "parametros_ajustados": aj.valores,
        "ajuste_convergiu": aj.convergiu,
        "ajuste_custo": aj.custo,
        "ajuste_n_inicios": aj.n_inicios,
        "ajuste_n_sucesso": aj.n_sucesso,
        "ajuste_n_no_minimo": aj.n_no_minimo,
        "ajuste_nfev": aj.nfev,
        "ajuste_mensagem": aj.mensagem,
    }
    if aj.expr_final is None:
        reg.update(status="invalido", categoria_erro="ajuste", mensagem_erro=aj.mensagem)
        return reg
    expr = aj.expr_final
    rt, r2t = _metricas_particao(expr, ds["treino"], prot)
    rv, r2v = _metricas_particao(expr, ds["validacao"], prot)
    ch = checagens_fisicas(expr, nivel, ds["treino"], prot)
    reg.update(
        status="valido",
        expressao=to_infix(expr),
        n_nos=size(expr),
        profundidade=depth(expr),
        rmse_log10_treino=rt,
        one_minus_r2_treino=r2t,
        rmse_log10_validacao=rv,
        one_minus_r2_validacao=r2v,
        checagens=[asdict(c) for c in ch],
        fisicamente_valido=fisicamente_valido(ch),
        checagens_reprovadas=[c.nome for c in ch if c.obrigatoria and c.aprovado is False],
    )
    return reg


def executar_geracao(
    nivel: NivelEquacao,
    condicao_geracao: str,
    provedor: ConfiguracaoLLM | Any,
    semente: int = 0,
    n_candidatos: int = 3,
    n_iteracoes_refinamento: int = 0,
    *,
    ruido: float = 0.0,
    n_curvas: int = 20,
    n_pontos: int = 50,
    max_retentativas: int = 2,
    n_inicios: int = N_INICIOS_PADRAO,
    datasets: Mapping[str, Dataset] | None = None,
    registro: RegistroUso | None = None,
    preco_por_milhao_tokens: tuple[float, float] | None = None,
    templates: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Executa um ensaio M2 (nível x condição G0/G1/G2 x modelo x semente).

    ``provedor``: :class:`ConfiguracaoLLM` (o provedor é criado e fechado aqui) ou objeto com
    ``generate``/``model_metadata`` (ex.: ``FakeBackend``, provedor já criado). Geram-se
    ``n_candidatos`` linhagens; em G2 cada linhagem é refinada por ``n_iteracoes_refinamento``
    rodadas (em G0/G1 esse valor é ignorado), logo ``n_candidatos x (1 + k)`` candidatos
    avaliados. Cada candidato admite até ``max_retentativas`` novas tentativas quando a
    resposta é recusada (JSON, schema, segurança, complexidade); todas ficam registradas.

    Devolve ``(linha, candidatos)``: ``linha`` no esquema ``experimento.COLUNAS_M1`` + colunas
    próprias do M2; ``candidatos`` com histórico completo (prompts, respostas, tentativas,
    ajuste, métricas de treino/validação, checagens, seleção).
    """
    if condicao_geracao not in CONDICOES_GERACAO:
        raise ValueError(f"condição de geração desconhecida: {condicao_geracao!r}")
    if n_candidatos < 1 or n_iteracoes_refinamento < 0 or max_retentativas < 0:
        raise ValueError("n_candidatos >= 1, n_iteracoes_refinamento >= 0, max_retentativas >= 0")
    k = n_iteracoes_refinamento if condicao_geracao == "G2" else 0
    t0 = time.perf_counter()
    backend, cfg, rotulo, registro, criado = _resolver_provedor(provedor, registro)
    uso_antes = registro.obter(cfg.rotulo) if (cfg is not None and registro is not None) else None
    ds = dict(datasets) if datasets is not None else construir_datasets(
        nivel, ruido, semente, n_curvas, n_pontos)
    ruido = float(ds["treino"].metadata.get("ruido_sigma", ruido))
    cfg_m = construir_config("pysr", "verificacao", semente)
    prot = ProtectionParams.from_config(cfg_m.symbolic)
    t = dict(templates) if templates is not None else carregar_templates()
    hashes = hash_templates(t)

    candidatos: list[dict[str, Any]] = []
    cache_avaliacao: dict[tuple[str, str], dict[str, Any]] = {}
    chamadas = 0
    tempo_inferencia = 0.0
    caracteres_prompt = caracteres_resposta = 0
    try:
        for rodada in range(k + 1):
            for j in range(n_candidatos):
                pai = None
                if rodada == 0:
                    anteriores = [c.get("equacao_normalizada") or c.get("equacao_proposta")
                                  for c in candidatos if c.get("equacao_proposta")]
                    feedback = None
                else:
                    pai = max(i for i, c in enumerate(candidatos) if c["linhagem"] == j)
                    anteriores = []
                    feedback = texto_feedback(candidatos[pai], t)
                sistema, usuario = montar_prompt(
                    nivel, condicao_geracao, ds["treino"], j + 1, n_candidatos, anteriores,
                    feedback, t)
                cand: dict[str, Any] = {
                    "indice": len(candidatos),
                    "linhagem": j,
                    "rodada": rodada,
                    "prompt_feedback_de": pai,
                    "hash_prompt": _sha256(sistema + "\n" + usuario),
                    "prompt_sistema": sistema,
                    "prompt_usuario": usuario,
                    "tentativas": [],
                    "status": "invalido",
                    "selecionado": False,
                }
                interp = None
                resp = None
                usuario_atual = usuario
                tempo_cand = 0.0
                for _ in range(max_retentativas + 1):
                    ti = time.perf_counter()
                    try:
                        texto = backend.generate([(sistema, usuario_atual)])[0]
                    except Exception as exc:  # noqa: BLE001 - falha do provedor é registrada
                        dt = time.perf_counter() - ti
                        tempo_cand += dt
                        cand["tentativas"].append({"categoria": "erro_provedor",
                                                   "mensagem": f"{type(exc).__name__}"[:200],
                                                   "resposta": None, "tempo_s": dt})
                        continue
                    dt = time.perf_counter() - ti
                    tempo_cand += dt
                    chamadas += 1
                    caracteres_prompt += len(sistema) + len(usuario_atual)
                    caracteres_resposta += len(texto or "")
                    try:
                        resp = validar_resposta(texto)
                        interp = interpretar_equacao(resp.equation, resp.parameters)
                    except RespostaInvalida as exc:
                        cand["tentativas"].append({"categoria": exc.categoria,
                                                   "mensagem": exc.mensagem[:500],
                                                   "resposta": (texto or "")[:4000], "tempo_s": dt})
                        if resp is not None:
                            cand["equacao_proposta"] = resp.equation
                        resp = None
                        usuario_atual = usuario + "\n\n" + Template(t["correcao"]).substitute(
                            categoria=exc.categoria, motivo=exc.mensagem[:300])
                        continue
                    cand["tentativas"].append({"categoria": "aceita", "mensagem": "",
                                               "resposta": texto[:4000], "tempo_s": dt})
                    break
                tempo_inferencia += tempo_cand
                cand["tempo_inferencia_s"] = tempo_cand
                cand["n_tentativas"] = len(cand["tentativas"])
                if interp is None:
                    ultima = cand["tentativas"][-1]
                    cand.update(categoria_erro=ultima["categoria"], mensagem_erro=ultima["mensagem"])
                else:
                    cand.update(
                        equacao_proposta=resp.equation,  # type: ignore[union-attr]
                        equacao_normalizada=interp.texto_normalizado,
                        variaveis_declaradas=resp.variables,  # type: ignore[union-attr]
                        parametros=list(interp.parametros),
                        chutes=interp.chutes,
                        parametros_inferidos=list(interp.parametros_inferidos),
                        parametros_nao_usados=list(interp.parametros_nao_usados),
                        assumptions=resp.assumptions,  # type: ignore[union-attr]
                        physical_constraints=resp.physical_constraints,  # type: ignore[union-attr]
                    )
                    chave = (interp.texto_normalizado, json.dumps(interp.chutes, sort_keys=True))
                    cand["duplicado"] = chave in cache_avaliacao
                    if chave not in cache_avaliacao:
                        cache_avaliacao[chave] = _avaliar_candidato(
                            interp, nivel, ds, semente, n_inicios, prot)
                    cand.update(json.loads(json.dumps(cache_avaliacao[chave])))
                candidatos.append(cand)
    finally:
        if criado:
            fechar_provedor(backend)

    sel = selecionar_melhor(candidatos)
    linha: dict[str, Any] = {
        "nivel": nivel.id,
        "nivel_proposta": nivel.nivel_proposta,
        "algoritmo": ALGORITMO_M2,
        "condicao": condicao_geracao,
        "ruido": ruido,
        "semente": semente,
        "configuracao_busca": f"geracao_n{n_candidatos}_k{k}",
        "expressao_referencia": nivel.expressao,
        "n_treino": int(ds["treino"].y_observed.size),
        "n_teste": int(ds["teste"].y_observed.size),
        "n_geracoes": k + 1,
    }
    metricas_nan = {c: math.nan for c in (
        "expressao", "n_nos", "one_minus_r2", "rmse_log10", "frac_pred_nao_positiva",
        "tree_distance", "tree_distance_equivalent_min", "tree_distance_algebraic",
        "forma_funcional_sympy")}
    if sel is None:
        linha.update(metricas_nan, stop_reason="sem_candidato_valido", status="erro",
                     erro="nenhum candidato interpretado e ajustado")
    else:
        candidatos[sel]["selecionado"] = True
        try:
            expr = parse_infix(candidatos[sel]["expressao"], FEATURES)
            linha.update(metricas_expressao(expr, nivel, ds["teste"].X, ds["teste"].y_true, cfg_m))
            linha.update(stop_reason="selecionado_por_validacao", status="ok")
        except Exception as exc:  # noqa: BLE001 - registrado como no M1
            linha.update(metricas_nan, stop_reason="erro_metricas", status="erro",
                         erro=f"{type(exc).__name__}: {exc}",
                         traceback=traceback.format_exc(limit=5))
    linha["tempo_s"] = time.perf_counter() - t0
    validos = [c for c in candidatos if c["status"] == "valido"]
    fis = [c for c in validos if c.get("fisicamente_valido")]
    linha.update(
        chamadas_llm=chamadas,
        expressoes_unicas_llm=len({c["equacao_normalizada"] for c in validos}),
        cache_hits_llm=0,
        tempo_llm_s=tempo_inferencia,
    )
    linha.update(classificar_linha(linha))

    tok_in = tok_out = math.nan
    if uso_antes is not None:
        depois = registro.obter(cfg.rotulo)  # type: ignore[union-attr]
        d_in = depois.tokens_entrada - uso_antes.tokens_entrada
        d_out = depois.tokens_saida - uso_antes.tokens_saida
        if d_in or d_out:
            tok_in, tok_out = float(d_in), float(d_out)
    custo = math.nan
    if preco_por_milhao_tokens is not None and math.isfinite(tok_in):
        custo = (tok_in * preco_por_milhao_tokens[0] + tok_out * preco_por_milhao_tokens[1]) / 1e6
    n_aval = len(candidatos)
    s = candidatos[sel] if sel is not None else {}
    linha.update(
        modo=MODO_M2,
        modelo=rotulo,
        n_candidatos=n_candidatos,
        n_iteracoes_refinamento=k,
        n_candidatos_avaliados=n_aval,
        n_candidatos_validos=len(validos),
        n_candidatos_fisicamente_validos=len(fis),
        taxa_validos=len(validos) / n_aval,
        taxa_fisicamente_validos=len(fis) / n_aval,
        selecionado_indice=sel if sel is not None else math.nan,
        selecionado_rodada=s.get("rodada", math.nan),
        selecionado_fisicamente_valido=s.get("fisicamente_valido", math.nan),
        checagens_reprovadas=";".join(s.get("checagens_reprovadas", [])) if s else math.nan,
        parametros_ajustados=json.dumps(s.get("parametros_ajustados")) if s else math.nan,
        rmse_log10_treino=s.get("rmse_log10_treino", math.nan),
        rmse_log10_validacao=s.get("rmse_log10_validacao", math.nan),
        one_minus_r2_validacao=s.get("one_minus_r2_validacao", math.nan),
        tokens_entrada=tok_in,
        tokens_saida=tok_out,
        caracteres_prompt=caracteres_prompt,
        caracteres_resposta=caracteres_resposta,
        tempo_inferencia_s=tempo_inferencia,
        custo_api_estimado=custo,
        max_retentativas=max_retentativas,
        n_inicios_ajuste=n_inicios,
        hash_templates=hashes["conjunto"],
        hash_datasets=hash_datasets(ds),
        metadados_modelo=json.dumps(
            backend.model_metadata() if hasattr(backend, "model_metadata") else {}, default=str),
    )
    ordem = [c for c in COLUNAS_M1 if c in linha]
    return {**{c: linha[c] for c in ordem}, **{c: v for c, v in linha.items() if c not in ordem}}, candidatos
