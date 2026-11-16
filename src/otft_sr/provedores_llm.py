"""Camada de provedores de LLM (Hugging Face, Ollama, OpenAI, Anthropic, Google, simulado).

Todo provedor satisfaz o protocolo ``TextGenerationBackend`` de ``physics_llm_sr``
(``generate(list[(system, user)]) -> list[str]`` e ``model_metadata()``), de modo que o
mesmo objeto serve ao modo M1 (LLM como avaliador de plausibilidade, termo ``c`` da perda,
via ``LLMEquationEvaluator``) e ao modo M2 (LLM gerador de equações, onda seguinte).

Decisões de reprodutibilidade:

- ``model_metadata()`` mantém as chaves mínimas do protocolo (``backend``, ``checkpoint``,
  ``revision``, ``quantization``, ``infra``), que entram na identidade do cache, e acrescenta
  campos padronizados em PT-BR: ``provedor``, ``modelo``, ``revisao``, ``quantizacao``,
  ``parametros_geracao`` e ``determinismo``.
- ``determinismo = "deterministico"`` só quando a decodificação é gulosa e local (Hugging Face
  com temperatura 0, Ollama com temperatura 0 e semente fixa, simulado). As APIs remotas
  (OpenAI, Anthropic, Google) declaram ``"nao_garantido"`` MESMO com temperatura 0: o
  servidor pode trocar o snapshot do modelo, o roteamento de hardware e a ordem de redução
  em ponto flutuante; a semente (OpenAI, Google) é apenas "melhor esforço" e a Anthropic
  não aceita semente. Respostas de API devem ser reaproveitadas pelo cache JSONL, nunca
  presumidas reproduzíveis.
- Credenciais vêm de variável de ambiente ou de ``google.colab.userdata`` (Colab Secrets),
  são lidas só no momento de criar o cliente do SDK e nunca ficam em atributo, ``repr``,
  metadados, registro de uso ou mensagem de erro.
- SDKs e ``torch``/``transformers``/``bitsandbytes`` são importados preguiçosamente: o módulo
  importa sem nenhuma dessas dependências e a ausência vira ``BackendError`` com a instrução
  de instalação no primeiro uso.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from physics_llm_sr.core_types import Track
from physics_llm_sr.llm.cache import EvaluationCache
from physics_llm_sr.llm.hf_snapshot import download_verified_snapshot
from physics_llm_sr.llm.inference import (
    PROMPT_FORMAT_PLAIN,
    FakeBackend,
    OllamaBackend,
    TransformersBackend,
    deterministic_fake_response,
)
from physics_llm_sr.llm.interfaces import (
    BackendConfigurationError,
    BackendError,
    TextGenerationBackend,
)
from physics_llm_sr.llm.scoring import LLMEquationEvaluator

from otft_sr.contexto import carregar_biblioteca
from otft_sr.ovsed import RAIZ_PROJETO

Provedor = Literal["huggingface", "ollama", "openai", "anthropic", "google", "fake"]
Quantizacao = Literal["4bit", "8bit"]

PROVEDORES: tuple[str, ...] = ("huggingface", "ollama", "openai", "anthropic", "google", "fake")
PROVEDORES_API: tuple[str, ...] = ("openai", "anthropic", "google")
QUANTIZACOES: tuple[str, ...] = ("4bit", "8bit")
DTYPES: tuple[str, ...] = ("float16", "bfloat16", "float32", "auto")
DETERMINISTICO = "deterministico"
NAO_GARANTIDO = "nao_garantido"

#: Variável de credencial padrão por provedor (o VALOR nunca é registrado).
CREDENCIAL_PADRAO: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "huggingface": "HF_TOKEN",
}
_NOME_VARIAVEL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

_DICAS_INSTALACAO = {
    "openai": "pip install openai",
    "anthropic": "pip install anthropic",
    "google.genai": "pip install google-genai",
    "bitsandbytes": "pip install bitsandbytes (requer GPU CUDA)",
}

TEMPERATURA_MAX = 2.0
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------- configuração


@dataclass(frozen=True)
class ConfiguracaoLLM:
    """Configuração tipada de um provedor de LLM (validada na construção).

    Campos:
        provedor: ``huggingface``, ``ollama``, ``openai``, ``anthropic``, ``google`` ou ``fake``.
        modelo: identificador do modelo (repositório do Hub, tag do Ollama ou nome na API).
        revisao: SHA do commit no Hub (só Hugging Face; recomendado para reprodutibilidade).
        quantizacao: ``None`` (sem quantização, ``dtype``), ``"4bit"`` (NF4) ou ``"8bit"``
            via bitsandbytes (só Hugging Face).
        dtype: precisão dos pesos/cômputo no Hugging Face.
        device_map: ``device_map`` do ``from_pretrained`` (Hugging Face).
        temperatura: 0 = decodificação gulosa (protocolo do artigo).
        max_novos_tokens: limite de tokens gerados por resposta.
        semente: semente de inferência (ignorada pela Anthropic, que não a aceita).
        timeout_s: tempo máximo por requisição (APIs e Ollama).
        max_tentativas: tentativas por prompt nas APIs (1 = sem retentativa); só erros
            transitórios (timeout, conexão, 429, 5xx) são repetidos.
        variavel_credencial: NOME da variável de ambiente / segredo do Colab com a chave
            (padrão por provedor em :data:`CREDENCIAL_PADRAO`); nunca o valor da chave.
        url_base: endereço do servidor Ollama (``None`` = ``OLLAMA_HOST`` ou localhost).
        tamanho_lote: prompts por lote no Hugging Face.
        espera_inicial_s: espera antes da 1ª retentativa (dobra a cada nova tentativa).
        resposta_fake: resposta fixa, roteiro ou função ``(system, user) -> str`` do provedor
            simulado (``None`` = resposta determinística derivada do hash do prompt).
    """

    provedor: Provedor
    modelo: str
    revisao: str | None = None
    quantizacao: Quantizacao | None = None
    dtype: str = "float16"
    device_map: str | None = "auto"
    temperatura: float = 0.0
    max_novos_tokens: int = 256
    semente: int = 0
    timeout_s: float = 120.0
    max_tentativas: int = 4
    variavel_credencial: str | None = None
    url_base: str | None = None
    tamanho_lote: int = 8
    espera_inicial_s: float = 1.0
    resposta_fake: str | Sequence[str] | Callable[[str, str], str] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Recusa combinações inválidas com mensagens claras (sem ecoar credenciais)."""
        if self.provedor not in PROVEDORES:
            raise ValueError(f"provedor deve ser um de {PROVEDORES}, recebeu {self.provedor!r}")
        if not isinstance(self.modelo, str) or not self.modelo.strip():
            raise ValueError("modelo não pode ser vazio")
        if self.revisao is not None and not str(self.revisao).strip():
            object.__setattr__(self, "revisao", None)
        if self.revisao is not None and self.provedor != "huggingface":
            raise ValueError("revisao só se aplica ao provedor huggingface")
        if self.quantizacao is not None:
            if self.quantizacao not in QUANTIZACOES:
                raise ValueError(f"quantizacao deve ser None ou um de {QUANTIZACOES}")
            if self.provedor != "huggingface":
                raise ValueError("quantizacao (bitsandbytes) só se aplica ao provedor huggingface")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype deve ser um de {DTYPES}")
        if isinstance(self.temperatura, bool) or not isinstance(self.temperatura, (int, float)):
            raise TypeError("temperatura deve ser numérica")
        if not 0.0 <= float(self.temperatura) <= TEMPERATURA_MAX:
            raise ValueError(f"temperatura deve estar em [0, {TEMPERATURA_MAX}]")
        if self.provedor == "ollama" and self.temperatura != 0:
            raise ValueError("Ollama: temperatura deve ser 0 (protocolo do artigo, p.6)")
        for nome in ("max_novos_tokens", "max_tentativas", "tamanho_lote"):
            valor = getattr(self, nome)
            if isinstance(valor, bool) or not isinstance(valor, int) or valor < 1:
                raise ValueError(f"{nome} deve ser inteiro >= 1")
        if isinstance(self.semente, bool) or not isinstance(self.semente, int) or self.semente < 0:
            raise ValueError("semente deve ser inteiro >= 0")
        if not self.timeout_s > 0:
            raise ValueError("timeout_s deve ser > 0")
        if not self.espera_inicial_s >= 0:
            raise ValueError("espera_inicial_s deve ser >= 0")
        if self.variavel_credencial is not None and not _NOME_VARIAVEL.match(
            str(self.variavel_credencial)
        ):
            # Sem ecoar o valor: o erro típico é colar a própria chave neste campo.
            raise ValueError(
                "variavel_credencial deve ser o NOME de uma variável (ex.: OPENAI_API_KEY), "
                "não a chave; o valor recebido foi omitido desta mensagem"
            )
        if self.resposta_fake is not None and self.provedor != "fake":
            raise ValueError("resposta_fake só se aplica ao provedor fake")

    @property
    def credencial(self) -> str | None:
        """Nome efetivo da variável de credencial (``None`` para Ollama/fake)."""
        return self.variavel_credencial or CREDENCIAL_PADRAO.get(self.provedor)

    @property
    def rotulo(self) -> str:
        """Identidade do modelo: ``provedor/modelo[@revisao12][[quantizacao]]``.

        É a coluna ``modelo`` dos ensaios, entra na chave de retomada e no registro de uso:
        a mesma revisão com outra quantização (ou outra revisão) é outro modelo. Sem revisão
        nem quantização (APIs, Ollama, simulado) fica ``provedor/modelo``.
        """
        r = f"{self.provedor}/{self.modelo}"
        if self.revisao:
            r += f"@{self.revisao[:12]}"
        if self.quantizacao:
            r += f"[{self.quantizacao}]"
        return r

    @property
    def rotulo_arquivo(self) -> str:
        """Rótulo seguro para nome de arquivo (cache JSONL por modelo)."""
        partes = [self.provedor, self.modelo]
        if self.revisao:
            partes.append(self.revisao[:12])
        if self.quantizacao:
            partes.append(self.quantizacao)
        return re.sub(r"[^A-Za-z0-9._-]+", "-", "__".join(partes)).strip("-")

    @property
    def determinismo(self) -> str:
        """Determinismo declarado (ver docstring do módulo)."""
        if self.provedor in PROVEDORES_API:
            return NAO_GARANTIDO
        return DETERMINISTICO if self.temperatura == 0 else NAO_GARANTIDO

    def parametros_geracao(self) -> dict[str, Any]:
        """Parâmetros de decodificação efetivamente enviados ao provedor."""
        p: dict[str, Any] = {
            "temperatura": float(self.temperatura),
            "max_novos_tokens": self.max_novos_tokens,
            "semente": None if self.provedor == "anthropic" else self.semente,
        }
        if self.provedor == "huggingface":
            p.update(do_sample=self.temperatura > 0, tamanho_lote=self.tamanho_lote)
        return p

    def como_dict(self) -> dict[str, Any]:
        """Representação JSON-serializável para persistência (sem ``resposta_fake``)."""
        d = asdict(self)
        d.pop("resposta_fake", None)
        d["variavel_credencial"] = self.credencial
        return d


# ------------------------------------------------------------------------ registro de uso


@dataclass
class UsoProvedor:
    """Contadores acumulados de um ``provedor/modelo``."""

    chamadas: int = 0
    falhas: int = 0
    retentativas: int = 0
    tokens_entrada: int = 0
    tokens_saida: int = 0
    segundos: float = 0.0

    def somar(self, outro: UsoProvedor) -> None:
        """Acumula ``outro`` neste contador."""
        for nome in ("chamadas", "falhas", "retentativas", "tokens_entrada", "tokens_saida"):
            setattr(self, nome, getattr(self, nome) + getattr(outro, nome))
        self.segundos += outro.segundos


class RegistroUso:
    """Contador de uso por ``provedor/modelo`` (thread-safe); só números, nunca credenciais."""

    def __init__(self) -> None:
        """Registro vazio."""
        self._uso: dict[str, UsoProvedor] = {}
        self._trava = threading.Lock()

    def registrar(
        self,
        chave: str,
        *,
        chamadas: int = 0,
        falhas: int = 0,
        retentativas: int = 0,
        tokens_entrada: int | None = None,
        tokens_saida: int | None = None,
        segundos: float = 0.0,
    ) -> None:
        """Acumula um evento; tokens ``None`` (API não informou) não somam."""
        with self._trava:
            u = self._uso.setdefault(chave, UsoProvedor())
            u.chamadas += chamadas
            u.falhas += falhas
            u.retentativas += retentativas
            u.tokens_entrada += int(tokens_entrada or 0)
            u.tokens_saida += int(tokens_saida or 0)
            u.segundos += float(segundos)

    def obter(self, chave: str) -> UsoProvedor:
        """Cópia do contador de ``chave`` (zerado se nunca usado)."""
        with self._trava:
            return UsoProvedor(**asdict(self._uso.get(chave, UsoProvedor())))

    def como_dict(self) -> dict[str, dict[str, Any]]:
        """``{provedor/modelo: contadores}`` JSON-serializável."""
        with self._trava:
            return {k: asdict(v) for k, v in sorted(self._uso.items())}

    def por_provedor(self) -> dict[str, dict[str, Any]]:
        """Contadores agregados por provedor (soma dos modelos)."""
        agregado: dict[str, UsoProvedor] = {}
        with self._trava:
            for chave, uso in self._uso.items():
                agregado.setdefault(chave.split("/", 1)[0], UsoProvedor()).somar(uso)
        return {k: asdict(v) for k, v in sorted(agregado.items())}

    def zerar(self) -> None:
        """Remove todos os contadores."""
        with self._trava:
            self._uso.clear()


#: Registro global usado quando o provedor é criado sem ``registro`` explícito.
REGISTRO_USO = RegistroUso()


# ------------------------------------------------------------------- utilidades internas


def _importar(modulo: str) -> Any:
    """Importa ``modulo`` sob demanda; ausência vira :class:`BackendError` com a instrução."""
    try:
        return importlib.import_module(modulo)
    except ImportError as exc:
        dica = _DICAS_INSTALACAO.get(modulo, f"pip install {modulo}")
        raise BackendError(f"dependência {modulo!r} ausente: {dica}") from exc


def _ler_segredo_colab(nome: str) -> str | None:
    """Valor de ``google.colab.userdata.get(nome)`` ou ``None`` fora do Colab/sem acesso."""
    try:
        userdata = importlib.import_module("google.colab.userdata")
    except Exception:  # noqa: BLE001 - fora do Colab
        try:
            userdata = importlib.import_module("google.colab").userdata
        except Exception:  # noqa: BLE001
            return None
    try:
        valor = userdata.get(nome)
    except Exception:  # noqa: BLE001 - SecretNotFoundError, NotebookAccessError etc.
        return None
    return str(valor) if valor else None


def ler_credencial(nome: str) -> str | None:
    """Chave da variável de ambiente ``nome`` ou do segredo homônimo do Colab.

    O valor é devolvido ao chamador para uso imediato e nunca é registrado por este módulo.
    """
    valor = os.environ.get(nome)
    if valor:
        return valor
    return _ler_segredo_colab(nome)


_PADROES_SEGREDO = re.compile(
    r"(sk-[A-Za-z0-9_\-*]{6,}|AIza[0-9A-Za-z_\-]{10,}|hf_[A-Za-z0-9]{10,}|[A-Za-z0-9_\-]{32,})"
)


def _mascarar(texto: str) -> str:
    """Remove trechos com formato de chave de API de mensagens de erro do SDK."""
    return _PADROES_SEGREDO.sub("***", texto)


_NOMES_TRANSITORIOS = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "OverloadedError",
        "ServiceUnavailableError",
        "ServerError",
        "TimeoutError",
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        "TimeoutException",
        "DeadlineExceeded",
        "ServiceUnavailable",
        "ResourceExhausted",
    }
)
_STATUS_TRANSITORIOS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


def _status(exc: BaseException) -> int | None:
    for atributo in ("status_code", "code", "status"):
        valor = getattr(exc, atributo, None)
        if isinstance(valor, int) and not isinstance(valor, bool):
            return valor
    return None


def erro_transitorio(exc: BaseException) -> bool:
    """``True`` para timeout, falha de conexão, limite de taxa (429) e erro 5xx do servidor."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if any(c.__name__ in _NOMES_TRANSITORIOS for c in type(exc).__mro__):
        return True
    return _status(exc) in _STATUS_TRANSITORIOS


def em_colab() -> bool:
    """``True`` quando executado no Google Colab."""
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return True
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        return False


def _infra_local() -> str:
    return "colab" if em_colab() else "local"


def _metadados_padrao(cfg: ConfiguracaoLLM, quantizacao: str | None) -> dict[str, Any]:
    return {
        "provedor": cfg.provedor,
        "modelo": cfg.modelo,
        "revisao": cfg.revisao,
        "quantizacao": quantizacao,
        "dtype": cfg.dtype if cfg.provedor == "huggingface" else None,
        "parametros_geracao": cfg.parametros_geracao(),
        "determinismo": cfg.determinismo,
    }


# ---------------------------------------------------------------------- provedor simulado


class ProvedorFake:
    """Provedor simulado: composição com ``physics_llm_sr.llm.inference.FakeBackend``."""

    def __init__(self, cfg: ConfiguracaoLLM, *, registro: RegistroUso | None = None) -> None:
        """``cfg.resposta_fake`` é repassada ao ``FakeBackend``."""
        self.cfg = cfg
        self.registro = registro if registro is not None else REGISTRO_USO
        self.interno = FakeBackend(cfg.resposta_fake, checkpoint=cfg.modelo)
        self.fechado = False

    def generate(self, prompts: list[tuple[str, str]]) -> list[str]:
        """Delegação ao ``FakeBackend`` com contagem de uso."""
        inicio = time.perf_counter()
        try:
            saida = self.interno.generate(prompts)
        except Exception:
            self.registro.registrar(self.cfg.rotulo, falhas=1)
            raise
        self.registro.registrar(
            self.cfg.rotulo, chamadas=len(prompts), segundos=time.perf_counter() - inicio
        )
        return saida

    def model_metadata(self) -> dict[str, Any]:
        """Metadados do ``FakeBackend`` + campos padronizados."""
        return {**self.interno.model_metadata(), **_metadados_padrao(self.cfg, None)}

    def close(self) -> None:
        """Nada a liberar (marca ``fechado``)."""
        self.fechado = True

    def __repr__(self) -> str:
        return f"ProvedorFake(modelo={self.cfg.modelo!r})"


#: Formas (com chutes) devolvidas pelo responder simulado do modo gerador; genéricas, não
#: são as equações-alvo dos níveis (só exercitam interpretação, ajuste e seleção).
_FORMAS_SIMULADAS: tuple[tuple[str, dict[str, float]], ...] = (
    ("K * (V_GS - V0) ** l", {"K": 1e-3, "V0": 5.0, "l": 2.0}),
    ("A * exp(b * V_GS) + C", {"A": 1.0, "b": 0.5, "C": 1.0}),
    ("K * (V_GS - V0) ** 2", {"K": 1e-3, "V0": 5.0}),
    ("K * V_DS * (V_GS - V0)", {"K": 1e-3, "V0": 5.0}),
)


def responder_simulado(system: str, user: str) -> str:
    """Resposta SIMULADA e determinística para execuções de verificação (não é resultado).

    Prompt do modo gerador (contém ``"equation"``): JSON com uma forma genérica escolhida
    pelo hash do prompt. Demais prompts (avaliador M1): escores derivados do hash
    (``deterministic_fake_response`` do ``physics_llm_sr``).
    """
    if '"equation"' not in system + user:
        return deterministic_fake_response(system, user)
    i = int(hashlib.sha256((system + "\x00" + user).encode("utf-8")).hexdigest()[:8], 16)
    equacao, chutes = _FORMAS_SIMULADAS[i % len(_FORMAS_SIMULADAS)]
    variaveis = [v for v in ("V_GS", "V_DS") if v in equacao]
    return json.dumps({
        "equation": equacao, "variables": variaveis, "parameters": chutes,
        "assumptions": ["simulated response"], "physical_constraints": ["I_D > 0"],
    })


# ------------------------------------------------------------------------------ Ollama


class ProvedorOllama:
    """Ollama local: composição com ``physics_llm_sr.llm.inference.OllamaBackend``.

    Os metadados do backend interno (``backend="ollama"``, ``checkpoint``, ``digest``) são
    preservados, de modo que a identidade do cache é a mesma de ``OllamaBackend`` puro.
    Tokens não são contados: o ``OllamaBackend`` não expõe ``prompt_eval_count``/``eval_count``.
    """

    def __init__(
        self,
        cfg: ConfiguracaoLLM,
        *,
        registro: RegistroUso | None = None,
        sessao: Any = None,
    ) -> None:
        """``sessao`` (``requests.Session``-like) permite simular HTTP nos testes."""
        self.cfg = cfg
        self.registro = registro if registro is not None else REGISTRO_USO
        self.interno = OllamaBackend(
            cfg.modelo,
            base_url=cfg.url_base,
            temperature=float(cfg.temperatura),
            seed=cfg.semente,
            num_predict=cfg.max_novos_tokens,
            timeout=cfg.timeout_s,
            session=sessao,
        )

    def resolve_identity(self) -> None:
        """Digest efetivo do modelo (entra na chave do cache do avaliador)."""
        self.interno.resolve_identity()

    def generate(self, prompts: list[tuple[str, str]]) -> list[str]:
        """Delegação ao ``OllamaBackend`` com contagem de uso."""
        inicio = time.perf_counter()
        try:
            saida = self.interno.generate(prompts)
        except Exception:
            self.registro.registrar(
                self.cfg.rotulo, falhas=1, segundos=time.perf_counter() - inicio
            )
            raise
        self.registro.registrar(
            self.cfg.rotulo, chamadas=len(prompts), segundos=time.perf_counter() - inicio
        )
        return saida

    def model_metadata(self) -> dict[str, Any]:
        """Metadados do ``OllamaBackend`` + campos padronizados (inclui ``digest``)."""
        interno = self.interno.model_metadata()
        meta = {**interno, **_metadados_padrao(self.cfg, interno.get("quantization"))}
        meta["revisao"] = interno.get("digest")
        return meta

    def close(self) -> None:
        """Fecha a sessão HTTP (o modelo continua residente no servidor Ollama)."""
        fechar = getattr(self.interno._session, "close", None)
        if callable(fechar):
            fechar()

    def __repr__(self) -> str:
        return f"ProvedorOllama(modelo={self.cfg.modelo!r}, url={self.interno.base_url!r})"


# ------------------------------------------------------------------------ Hugging Face


def _rotulo_quantizacao_hf(cfg: ConfiguracaoLLM) -> str:
    if cfg.quantizacao == "4bit":
        return f"bnb-4bit-nf4/{cfg.dtype}"
    if cfg.quantizacao == "8bit":
        return "bnb-8bit"
    return cfg.dtype


class ProvedorHuggingFace(TransformersBackend):
    """Hugging Face ``transformers`` com quantização opcional (bitsandbytes 4/8 bits).

    Reaproveita de ``TransformersBackend`` o carregamento do tokenizer, o formato do prompt
    (chat template do tokenizer, com fusão do ``system`` quando o template o recusa, ou texto
    simples), o download verificado do snapshot e o ``close()`` (``del`` dos pesos,
    ``gc.collect``, ``torch.cuda.empty_cache``). Acrescenta ``dtype``, ``BitsAndBytesConfig``
    e amostragem quando ``temperatura > 0`` (gulosa quando 0), sempre com
    ``torch.manual_seed(semente)`` antes de cada lote. ``trust_remote_code`` é sempre falso.
    Modelos restritos usam ``HF_TOKEN`` (ambiente ou segredo do Colab, lido pelo
    ``huggingface_hub``).
    """

    def __init__(self, cfg: ConfiguracaoLLM, *, registro: RegistroUso | None = None) -> None:
        """Valida a configuração sem importar ``torch``/``transformers`` nem carregar pesos."""
        super().__init__(
            cfg.modelo,
            cfg.revisao,
            track=Track.EXTENDED_ANALYSIS,
            max_new_tokens=cfg.max_novos_tokens,
            batch_size=cfg.tamanho_lote,
            seed=cfg.semente,
            infra=_infra_local(),
            device_map=cfg.device_map,
        )
        self.cfg = cfg
        self.registro = registro if registro is not None else REGISTRO_USO

    def _dtype(self, torch: Any) -> Any:
        return "auto" if self.cfg.dtype == "auto" else getattr(torch, self.cfg.dtype)

    def _config_quantizacao(self, torch: Any, transformers: Any) -> Any:
        if self.cfg.quantizacao is None:
            return None
        _importar("bitsandbytes")
        if self.cfg.quantizacao == "8bit":
            return transformers.BitsAndBytesConfig(load_in_8bit=True)
        computo = torch.float16 if self.cfg.dtype == "auto" else self._dtype(torch)
        return transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=computo,
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        torch, transformers = self._modules()
        self._load_tokenizer()
        quant = self._config_quantizacao(torch, transformers)
        snapshot = download_verified_snapshot(self.checkpoint, self.revision)
        kwargs: dict[str, Any] = {
            "torch_dtype": self._dtype(torch),
            "trust_remote_code": False,
            "device_map": self.device_map,
        }
        if quant is not None:
            kwargs["quantization_config"] = quant
        model = transformers.AutoModelForCausalLM.from_pretrained(str(snapshot), **kwargs)
        model.eval()
        self._model = model

    def _kwargs_geracao(self) -> dict[str, Any]:
        if self.cfg.temperatura == 0:
            return {"do_sample": False}
        return {"do_sample": True, "temperature": float(self.cfg.temperatura)}

    def generate(self, prompts: list[tuple[str, str]]) -> list[str]:
        """Geração em lotes (padding à esquerda), na ordem de entrada, com contagem de tokens."""
        inicio = time.perf_counter()
        try:
            saida, tok_in, tok_out = self._gerar(prompts)
        except Exception:
            self.registro.registrar(
                self.cfg.rotulo, falhas=1, segundos=time.perf_counter() - inicio
            )
            raise
        self.registro.registrar(
            self.cfg.rotulo,
            chamadas=len(prompts),
            tokens_entrada=tok_in,
            tokens_saida=tok_out,
            segundos=time.perf_counter() - inicio,
        )
        return saida

    def _gerar(self, prompts: list[tuple[str, str]]) -> tuple[list[str], int, int]:
        self._load()
        torch, tokenizer, model = self._torch, self._tokenizer, self._model
        out: list[str] = []
        tok_in = tok_out = 0
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start : start + self.batch_size]
            formatted = [self._format(s, u) for s, u in chunk]
            self._prompt_format = formatted[-1][1]
            encoding = tokenizer(
                [text for text, _ in formatted],
                return_tensors="pt",
                padding=True,
                add_special_tokens=self._prompt_format == PROMPT_FORMAT_PLAIN,
            ).to(model.device)
            input_len = encoding["input_ids"].shape[1]
            torch.manual_seed(self.seed)
            with torch.no_grad():
                generated = model.generate(
                    **encoding,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    **self._kwargs_geracao(),
                )
            novos = generated[:, input_len:]
            tok_in += int(encoding["attention_mask"].sum())
            tok_out += int((novos != tokenizer.pad_token_id).sum())
            out.extend(str(t) for t in tokenizer.batch_decode(novos, skip_special_tokens=True))
        return out, tok_in, tok_out

    def model_metadata(self) -> dict[str, Any]:
        """Metadados do ``TransformersBackend`` com quantização/dtype reais e campos padronizados."""
        quant = _rotulo_quantizacao_hf(self.cfg)
        meta = super().model_metadata()
        meta.update(
            quantization=quant,
            do_sample=self.cfg.temperatura > 0,
            device_map=self.device_map,
            bitsandbytes_version=_versao("bitsandbytes") if self.cfg.quantizacao else None,
        )
        meta.update(_metadados_padrao(self.cfg, quant))
        return meta

    def __repr__(self) -> str:
        return (
            f"ProvedorHuggingFace(modelo={self.checkpoint!r}, revisao={self.revision!r}, "
            f"quantizacao={self.cfg.quantizacao!r}, carregado={self.loaded})"
        )


# ------------------------------------------------------------------------- APIs remotas


@dataclass(frozen=True)
class _Resposta:
    texto: str
    tokens_entrada: int | None
    tokens_saida: int | None
    modelo_servido: str | None


class _ProvedorAPI:
    """Base das APIs remotas: cliente preguiçoso, retentativa com backoff e registro de uso.

    A chave é lida (ambiente ou Colab Secrets) só ao criar o cliente do SDK e passada
    diretamente ao construtor; este objeto não a guarda. A retentativa interna dos SDKs é
    desligada (``max_retries=0``) para que a política seja a de :class:`ConfiguracaoLLM`.
    """

    modulo_sdk: str = ""

    def __init__(
        self,
        cfg: ConfiguracaoLLM,
        *,
        registro: RegistroUso | None = None,
        dormir: Callable[[float], None] = time.sleep,
    ) -> None:
        """Só valida; o SDK é importado no primeiro :meth:`generate`."""
        self.cfg = cfg
        self.registro = registro if registro is not None else REGISTRO_USO
        self._dormir = dormir
        self._cliente_sdk: Any = None
        self._modelo_servido: str | None = None

    # -- a especializar
    def _criar_cliente(self, sdk: Any, chave: str) -> Any:
        raise NotImplementedError

    def _chamar(self, cliente: Any, system: str, user: str) -> _Resposta:
        raise NotImplementedError

    # -- comum
    def _cliente(self) -> Any:
        if self._cliente_sdk is None:
            sdk = _importar(self.modulo_sdk)
            nome = self.cfg.credencial
            assert nome is not None
            chave = ler_credencial(nome)
            if not chave:
                raise BackendConfigurationError(
                    f"credencial ausente para {self.cfg.provedor}: defina a variável de ambiente "
                    f"{nome} ou o segredo homônimo no Colab (Secrets)"
                )
            self._cliente_sdk = self._criar_cliente(sdk, chave)
            del chave
        return self._cliente_sdk

    def _gerar_um(self, system: str, user: str) -> str:
        cliente = self._cliente()
        rotulo = self.cfg.rotulo
        for tentativa in range(1, self.cfg.max_tentativas + 1):
            inicio = time.perf_counter()
            try:
                r = self._chamar(cliente, system, user)
            except Exception as exc:  # noqa: BLE001 - classificado abaixo
                decorrido = time.perf_counter() - inicio
                if erro_transitorio(exc) and tentativa < self.cfg.max_tentativas:
                    self.registro.registrar(rotulo, retentativas=1, segundos=decorrido)
                    self._dormir(self.cfg.espera_inicial_s * 2 ** (tentativa - 1))
                    continue
                self.registro.registrar(rotulo, falhas=1, segundos=decorrido)
                status = _status(exc)
                detalhe = _mascarar(str(exc))[:300]
                raise BackendError(
                    f"{rotulo}: {type(exc).__name__}"
                    f"{'' if status is None else f' (HTTP {status})'} após {tentativa} "
                    f"tentativa(s): {detalhe}"
                ) from None  # sem encadear: a exceção do SDK pode conter dados da requisição
            self._modelo_servido = r.modelo_servido or self._modelo_servido
            self.registro.registrar(
                rotulo,
                chamadas=1,
                tokens_entrada=r.tokens_entrada,
                tokens_saida=r.tokens_saida,
                segundos=time.perf_counter() - inicio,
            )
            return r.texto
        raise AssertionError("inalcançável")  # pragma: no cover

    def generate(self, prompts: list[tuple[str, str]]) -> list[str]:
        """Uma requisição por prompt, na ordem; falha persistente vira :class:`BackendError`."""
        return [self._gerar_um(system, user) for system, user in prompts]

    def model_metadata(self) -> dict[str, Any]:
        """Chaves mínimas do protocolo + campos padronizados; nunca contém a chave."""
        return {
            "backend": self.cfg.provedor,
            "checkpoint": self.cfg.modelo,
            "revision": None,
            "quantization": None,
            "infra": "api-remota",
            **_metadados_padrao(self.cfg, None),
            "modelo_servido": self._modelo_servido,
            "variavel_credencial": self.cfg.credencial,
            "max_tentativas": self.cfg.max_tentativas,
            "timeout_s": self.cfg.timeout_s,
        }

    def close(self) -> None:
        """Fecha o cliente HTTP do SDK (se houver) e o descarta."""
        cliente, self._cliente_sdk = self._cliente_sdk, None
        fechar = getattr(cliente, "close", None)
        if callable(fechar):
            try:
                fechar()
            except Exception as exc:  # noqa: BLE001 - fechamento é melhor esforço
                logger.debug("falha ao fechar cliente %s: %s", self.cfg.rotulo, type(exc).__name__)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(modelo={self.cfg.modelo!r})"


def _inteiro(obj: Any, *nomes: str) -> int | None:
    for nome in nomes:
        valor = getattr(obj, nome, None) if obj is not None else None
        if isinstance(valor, int) and not isinstance(valor, bool):
            return valor
    return None


class ProvedorOpenAI(_ProvedorAPI):
    """OpenAI Chat Completions (``openai`` oficial).

    Envia ``temperature``, ``seed`` (melhor esforço) e ``max_completion_tokens``; registra
    ``usage.prompt_tokens``/``completion_tokens`` e o ``model`` servido.
    """

    modulo_sdk = "openai"

    def _criar_cliente(self, sdk: Any, chave: str) -> Any:
        return sdk.OpenAI(api_key=chave, timeout=self.cfg.timeout_s, max_retries=0)

    def _chamar(self, cliente: Any, system: str, user: str) -> _Resposta:
        r = cliente.chat.completions.create(
            model=self.cfg.modelo,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=float(self.cfg.temperatura),
            seed=self.cfg.semente,
            max_completion_tokens=self.cfg.max_novos_tokens,
        )
        texto = r.choices[0].message.content
        if not isinstance(texto, str):
            raise BackendError(f"{self.cfg.rotulo}: resposta sem texto")
        uso = getattr(r, "usage", None)
        return _Resposta(
            texto,
            _inteiro(uso, "prompt_tokens"),
            _inteiro(uso, "completion_tokens"),
            getattr(r, "model", None),
        )


class ProvedorAnthropic(_ProvedorAPI):
    """Anthropic Messages (``anthropic`` oficial); a API não aceita semente.

    Registra ``usage.input_tokens``/``output_tokens`` e o ``model`` servido.
    """

    modulo_sdk = "anthropic"

    def _criar_cliente(self, sdk: Any, chave: str) -> Any:
        return sdk.Anthropic(api_key=chave, timeout=self.cfg.timeout_s, max_retries=0)

    def _chamar(self, cliente: Any, system: str, user: str) -> _Resposta:
        r = cliente.messages.create(
            model=self.cfg.modelo,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=float(self.cfg.temperatura),
            max_tokens=self.cfg.max_novos_tokens,
        )
        texto = "".join(
            b.text for b in r.content if getattr(b, "type", "text") == "text" and hasattr(b, "text")
        )
        uso = getattr(r, "usage", None)
        return _Resposta(
            texto,
            _inteiro(uso, "input_tokens"),
            _inteiro(uso, "output_tokens"),
            getattr(r, "model", None),
        )


class ProvedorGoogle(_ProvedorAPI):
    """Google Gemini via ``google-genai`` (``from google import genai``).

    Envia ``system_instruction``, ``temperature``, ``seed`` (melhor esforço) e
    ``max_output_tokens``; registra ``usage_metadata.prompt_token_count``/
    ``candidates_token_count`` e ``model_version``.
    """

    modulo_sdk = "google.genai"

    def _criar_cliente(self, sdk: Any, chave: str) -> Any:
        return sdk.Client(api_key=chave, http_options={"timeout": int(self.cfg.timeout_s * 1000)})

    def _chamar(self, cliente: Any, system: str, user: str) -> _Resposta:
        r = cliente.models.generate_content(
            model=self.cfg.modelo,
            contents=user,
            config={
                "system_instruction": system,
                "temperature": float(self.cfg.temperatura),
                "seed": self.cfg.semente,
                "max_output_tokens": self.cfg.max_novos_tokens,
            },
        )
        texto = getattr(r, "text", None)
        if not isinstance(texto, str):
            raise BackendError(f"{self.cfg.rotulo}: resposta sem texto (bloqueada ou vazia)")
        uso = getattr(r, "usage_metadata", None)
        return _Resposta(
            texto,
            _inteiro(uso, "prompt_token_count"),
            _inteiro(uso, "candidates_token_count"),
            getattr(r, "model_version", None),
        )


# ---------------------------------------------------------------- fábrica e ciclo de vida

_CLASSES: dict[str, type] = {
    "fake": ProvedorFake,
    "ollama": ProvedorOllama,
    "huggingface": ProvedorHuggingFace,
    "openai": ProvedorOpenAI,
    "anthropic": ProvedorAnthropic,
    "google": ProvedorGoogle,
}


def criar_provedor(
    cfg: ConfiguracaoLLM, *, registro: RegistroUso | None = None, **kwargs: Any
) -> TextGenerationBackend:
    """Instancia o provedor de ``cfg`` (sem rede, sem carregar pesos, sem importar SDKs).

    ``kwargs`` vão ao construtor específico (ex.: ``sessao`` do Ollama, ``dormir`` das APIs).
    """
    if not isinstance(cfg, ConfiguracaoLLM):
        raise TypeError("cfg deve ser ConfiguracaoLLM")
    return _CLASSES[cfg.provedor](cfg, registro=registro, **kwargs)


def fechar_provedor(provedor: object) -> None:
    """Chama ``close()`` (se existir) e coleta lixo; idempotente."""
    fechar = getattr(provedor, "close", None)
    if callable(fechar):
        fechar()
    gc.collect()


@contextmanager
def usar_provedor(
    cfg: ConfiguracaoLLM, *, registro: RegistroUso | None = None, **kwargs: Any
) -> Iterator[TextGenerationBackend]:
    """Provedor com liberação garantida de memória ao sair (execução sequencial de modelos)::

        for cfg in configuracoes:
            with usar_provedor(cfg) as provedor:
                ...  # avaliações
    """
    provedor = criar_provedor(cfg, registro=registro, **kwargs)
    try:
        yield provedor
    finally:
        fechar_provedor(provedor)


# ------------------------------------------------------------------------------ ambiente


def _versao(pacote: str) -> str | None:
    try:
        return importlib.metadata.version(pacote)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_nvidia_smi() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        saida = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for linha in saida.strip().splitlines():
        nome, _, mem = linha.rpartition(",")
        try:
            gpus.append({"nome": nome.strip(), "memoria_gb": round(float(mem) / 1024, 2)})
        except ValueError:
            continue
    return gpus


def relatorio_ambiente() -> dict[str, Any]:
    """CPU/GPU, memória de GPU, versões de torch/transformers/bitsandbytes e Colab ou não.

    Não importa ``torch`` se ele não estiver instalado; sem ``torch``, a GPU é consultada
    por ``nvidia-smi`` quando disponível.
    """
    rel: dict[str, Any] = {
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "n_cpus": os.cpu_count(),
        "colab": em_colab(),
        "versoes": {
            p: _versao(p)
            for p in (
                "torch",
                "transformers",
                "accelerate",
                "bitsandbytes",
                "openai",
                "anthropic",
                "google-genai",
            )
        },
        "cuda_disponivel": False,
        "gpus": [],
    }
    try:
        rel["memoria_ram_gb"] = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 2
        )
    except (ValueError, OSError, AttributeError):
        rel["memoria_ram_gb"] = None
    if rel["versoes"]["torch"] is not None:
        try:
            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                rel["cuda_disponivel"] = True
                rel["versao_cuda"] = torch.version.cuda
                for i in range(torch.cuda.device_count()):
                    prop = torch.cuda.get_device_properties(i)
                    rel["gpus"].append(
                        {"nome": prop.name, "memoria_gb": round(prop.total_memory / 1024**3, 2)}
                    )
        except Exception as exc:  # noqa: BLE001 - relatório não deve falhar
            rel["erro_torch"] = f"{type(exc).__name__}: {exc}"
    if not rel["gpus"]:
        rel["gpus"] = _gpu_nvidia_smi()
    rel["memoria_gpu_gb"] = max((g["memoria_gb"] for g in rel["gpus"]), default=0.0)
    return rel


#: GB por bilhão de parâmetros (pesos) e margem fixa (ativações, cache KV, contexto CUDA).
GB_POR_BILHAO = {None: 2.0, "8bit": 1.05, "4bit": 0.6}
MARGEM_GB = 1.5
FATOR_FOLGA = 1.2


def memoria_estimada_gb(n_parametros_b: float, quantizacao: str | None) -> float:
    """Memória de GPU estimada para inferência (pesos × folga + margem)."""
    return n_parametros_b * GB_POR_BILHAO[quantizacao] * FATOR_FOLGA + MARGEM_GB


def recomendar_quantizacao(memoria_gpu_gb: float, n_parametros_b: float) -> str | None:
    """Menor compressão que cabe na GPU: ``None`` (fp16/bf16), ``"8bit"`` ou ``"4bit"``.

    Estimativa grosseira (``memoria_estimada_gb``) para prompts curtos como os do M1.
    Levanta ``ValueError`` sem GPU ou quando nem 4 bits cabe (use Ollama ou API).
    """
    if not n_parametros_b > 0:
        raise ValueError("n_parametros_b deve ser > 0 (bilhões de parâmetros)")
    if not memoria_gpu_gb or memoria_gpu_gb <= 0:
        raise ValueError("sem GPU: bitsandbytes exige CUDA; use Ollama ou um provedor de API")
    for quant in (None, "8bit", "4bit"):
        if memoria_estimada_gb(n_parametros_b, quant) <= memoria_gpu_gb:
            return quant
    raise ValueError(
        f"modelo de {n_parametros_b:g}B não cabe em {memoria_gpu_gb:g} GB nem em 4 bits "
        f"(~{memoria_estimada_gb(n_parametros_b, '4bit'):.1f} GB); use Ollama ou API"
    )


# ------------------------------------------------------------------------ integração M1

DIR_CACHE_LLM = RAIZ_PROJETO / ".cache" / "llm"


def caminho_cache(cfg: ConfiguracaoLLM, condicao: str, dir_cache: Path) -> Path:
    """JSONL do cache do avaliador, separado por condição e por modelo."""
    return Path(dir_cache) / f"avaliacoes_{condicao}__{cfg.rotulo_arquivo}.jsonl"


def criar_avaliador_m1(
    cfg: ConfiguracaoLLM,
    condicao: str,
    *,
    provedor: TextGenerationBackend | None = None,
    dir_cache: Path | None = DIR_CACHE_LLM,
    arquivo_cache: Path | None = None,
    max_retries: int = 1,
    registro: RegistroUso | None = None,
) -> LLMEquationEvaluator:
    """``LLMEquationEvaluator`` (M1: LLM avaliador de plausibilidade, termo ``c``).

    Biblioteca de prompts: ``carregar_biblioteca(condicao)`` (B = genérico, C = dispositivo).
    ``provedor=None`` cria pela fábrica a partir de ``cfg``. Cache JSONL em
    :func:`caminho_cache` (``arquivo_cache`` sobrescreve; ``dir_cache=None`` desliga).
    Para Ollama criado aqui, o digest efetivo do modelo é resolvido já na construção (falha
    cedo se o servidor não responde); um ``provedor`` explícito é usado como está.
    """
    biblioteca = carregar_biblioteca(condicao)
    if provedor is None:
        provedor = criar_provedor(cfg, registro=registro)
        resolver = getattr(provedor, "resolve_identity", None)
        if callable(resolver) and cfg.provedor == "ollama":
            resolver()
    cache = None
    if arquivo_cache is not None or dir_cache is not None:
        caminho = (
            Path(arquivo_cache)
            if arquivo_cache is not None
            else caminho_cache(cfg, condicao, Path(dir_cache))  # type: ignore[arg-type]
        )
        caminho.parent.mkdir(parents=True, exist_ok=True)
        cache = EvaluationCache(caminho)
    return LLMEquationEvaluator(provedor, biblioteca, cache=cache, max_retries=max_retries)
