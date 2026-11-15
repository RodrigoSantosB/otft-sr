"""PhysicalContext por nível e avaliadores LLM das condições experimentais B e C.

- Condição B: biblioteca ``prompts/generico`` (prompt do artigo, física clássica).
- Condição C: biblioteca ``prompts/dispositivo`` (física de OTFT e critérios verificáveis
  da proposta §5.2).
Ambas usam o MESMO PhysicalContext e a variante E (variáveis + descrição do experimento,
sem a equação-alvo), de modo que só o prompt de sistema difere entre B e C (H-OTFT-09).
Textos enviados ao LLM ficam em inglês (mesma língua do prompt do artigo; H-OTFT-10).
"""

from __future__ import annotations

from pathlib import Path

import requests

from physics_llm_sr.core_types import PhysicalContext
from physics_llm_sr.llm.cache import EvaluationCache
from physics_llm_sr.llm.prompting import PromptLibrary
from physics_llm_sr.llm.scoring import LLMEquationEvaluator

from otft_sr.niveis import NivelEquacao
from otft_sr.ovsed import RAIZ_PROJETO

DIR_PROMPTS = RAIZ_PROJETO / "prompts"
BIBLIOTECA_POR_CONDICAO: dict[str, str] = {"B": "generico", "C": "dispositivo"}
VARIANTE_PROMPT = "E"
MODELO_OLLAMA = "mistral:7b-instruct"
URL_OLLAMA = "http://localhost:11434"

_DESCRICAO_VARIAVEIS = {
    "V_GS": "gate-source voltage magnitude of a p-type OTFT",
    "V_DS": "drain-source voltage magnitude of a p-type OTFT",
    "I_D": "drain current magnitude (target)",
}

_DESCRICAO_REGIAO = {
    "transferencia": "transfer curves: each curve keeps V_DS fixed and sweeps V_GS",
    "saida": "output curves: each curve keeps V_GS fixed and sweeps V_DS",
}


def _faixa(a: tuple[float, float]) -> str:
    return f"[{a[0]:g}, {a[1]:g}] V"


def contexto_fisico(nivel: NivelEquacao) -> PhysicalContext:
    """Contexto do nível (sem revelar a equação no texto; ``ground_truth`` só entra nas
    variantes D/F/G/H, que não são usadas)."""
    r = nivel.regiao
    limite_vds = (
        f"V_DS from {r.V_DS[0]:g} V up to V_GS minus an effective threshold voltage"
        if r.V_DS_max_rel_V_GS is not None
        else f"V_DS in {_faixa(r.V_DS)}"
    )
    unidade = {"uA": "microamperes", "pA": "picoamperes"}[nivel.unidades["I_D"]]
    leak = "" if r.com_leak else " The constant gate/leakage current floor has been subtracted."
    descricao = (
        "DC electrical characterization of an organic thin-film transistor (p-type, magnitudes "
        f"used). Data are {_DESCRICAO_REGIAO[r.tipo_curva]}, with V_GS in {_faixa(r.V_GS)} and "
        f"{limite_vds}. The drain current I_D is reported in {unidade}.{leak} Both V_GS and "
        "V_DS are offered to the regression even if the current depends on only one of them."
    )
    return PhysicalContext(
        scenario=nivel.id,
        variable_descriptions=dict(_DESCRICAO_VARIAVEIS),
        experiment_description=descricao,
        ground_truth=nivel.expressao,
        units=dict(nivel.unidades),
    )


def carregar_biblioteca(condicao: str) -> PromptLibrary:
    """Biblioteca de prompts da condição B (genérico) ou C (dispositivo)."""
    try:
        nome = BIBLIOTECA_POR_CONDICAO[condicao]
    except KeyError:
        raise ValueError(f"condição sem LLM ou desconhecida: {condicao!r}") from None
    return PromptLibrary.load(DIR_PROMPTS / nome)


def ollama_disponivel(modelo: str = MODELO_OLLAMA, url: str = URL_OLLAMA) -> bool:
    """``True`` se o servidor Ollama responde e lista ``modelo``."""
    try:
        r = requests.get(f"{url}/api/tags", timeout=3)
        r.raise_for_status()
        nomes = {m.get("name") for m in r.json().get("models", [])}
    except Exception:  # noqa: BLE001 - qualquer falha = indisponível
        return False
    return modelo in nomes


def criar_avaliador(
    condicao: str,
    backend: object | None = None,
    dir_cache: Path | None = RAIZ_PROJETO / ".cache" / "llm",
    modelo: str = MODELO_OLLAMA,
) -> LLMEquationEvaluator:
    """Avaliador LLM da condição (``backend=None`` cria ``OllamaBackend(modelo)``).

    O cache é separado por condição (o hash do prompt já entra na chave; o arquivo separado
    só facilita auditoria).
    """
    if backend is None:
        from physics_llm_sr.llm.inference import OllamaBackend

        backend = OllamaBackend(modelo, base_url=URL_OLLAMA, num_predict=256)
        backend.resolve_identity()
    cache = None
    if dir_cache is not None:
        Path(dir_cache).mkdir(parents=True, exist_ok=True)
        cache = EvaluationCache(Path(dir_cache) / f"avaliacoes_{condicao}.jsonl")
    return LLMEquationEvaluator(
        backend, carregar_biblioteca(condicao), cache=cache, max_retries=1
    )
