"""Camada de provedores de LLM: configuração, fábrica, APIs com SDK falso, M1 simulado.

Nenhum teste usa rede, GPU ou credencial real: SDKs, ``torch`` e ``transformers`` são
substituídos por módulos falsos via ``monkeypatch``.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from contextlib import contextmanager

import numpy as np
import pytest

from physics_llm_sr.llm.interfaces import (
    BackendConfigurationError,
    BackendError,
    EquationEvaluation,
    TextGenerationBackend,
)
from physics_llm_sr.llm.scoring import LLMEquationEvaluator
from physics_llm_sr.symbolic.expression import parse_infix

from otft_sr import provedores_llm as pl
from otft_sr.contexto import VARIANTE_PROMPT, contexto_fisico, criar_avaliador
from otft_sr.niveis import FEATURES, nivel
from otft_sr.provedores_llm import (
    NAO_GARANTIDO,
    ConfiguracaoLLM,
    RegistroUso,
    criar_avaliador_m1,
    criar_provedor,
    recomendar_quantizacao,
    relatorio_ambiente,
    usar_provedor,
)

CHAVE = "sk-SEGREDO-de-teste-1234567890abcdef"
RESPOSTA_JSON = json.dumps(
    {
        "dimensional_consistency": 0.9,
        "simplicity": 0.8,
        "physical_realism": 0.7,
        "feedback": "ok",
    }
)


# ------------------------------------------------------------------------ configuração


def test_config_padrao_valida_e_serializavel():
    cfg = ConfiguracaoLLM("openai", "gpt-4o-mini")
    assert cfg.temperatura == 0.0
    assert cfg.credencial == "OPENAI_API_KEY"
    assert cfg.determinismo == NAO_GARANTIDO  # API: mesmo com temperatura 0
    d = cfg.como_dict()
    json.dumps(d)
    assert "resposta_fake" not in d


@pytest.mark.parametrize(
    "kwargs, trecho",
    [
        ({"provedor": "azure", "modelo": "x"}, "provedor"),
        ({"provedor": "openai", "modelo": "  "}, "modelo"),
        ({"provedor": "openai", "modelo": "m", "quantizacao": "4bit"}, "huggingface"),
        ({"provedor": "huggingface", "modelo": "m", "quantizacao": "3bit"}, "quantizacao"),
        ({"provedor": "openai", "modelo": "m", "revisao": "abc"}, "revisao"),
        ({"provedor": "huggingface", "modelo": "m", "dtype": "int4"}, "dtype"),
        ({"provedor": "openai", "modelo": "m", "temperatura": -0.1}, "temperatura"),
        ({"provedor": "ollama", "modelo": "m", "temperatura": 0.5}, "temperatura"),
        ({"provedor": "openai", "modelo": "m", "max_novos_tokens": 0}, "max_novos_tokens"),
        ({"provedor": "openai", "modelo": "m", "max_tentativas": 0}, "max_tentativas"),
        ({"provedor": "openai", "modelo": "m", "timeout_s": 0}, "timeout_s"),
        ({"provedor": "openai", "modelo": "m", "semente": -1}, "semente"),
        ({"provedor": "openai", "modelo": "m", "resposta_fake": "x"}, "fake"),
    ],
)
def test_config_recusa_entradas_invalidas(kwargs, trecho):
    with pytest.raises(ValueError, match=trecho):
        ConfiguracaoLLM(**kwargs)


def test_config_temperatura_nao_numerica():
    with pytest.raises(TypeError, match="temperatura"):
        ConfiguracaoLLM("openai", "m", temperatura=True)


def test_config_recusa_chave_no_lugar_do_nome_sem_ecoar():
    with pytest.raises(ValueError) as info:
        ConfiguracaoLLM("openai", "m", variavel_credencial=CHAVE)
    assert CHAVE not in str(info.value)


def test_rotulo_arquivo_seguro():
    cfg = ConfiguracaoLLM("huggingface", "org/Modelo:7b", revisao="a" * 40, quantizacao="4bit")
    assert "/" not in cfg.rotulo_arquivo and ":" not in cfg.rotulo_arquivo
    assert cfg.rotulo_arquivo.endswith("4bit")


def test_rotulo_inclui_revisao_e_quantizacao():
    """Identidade do modelo (chave de retomada e coluna ``modelo``): revisão e quantização."""
    assert ConfiguracaoLLM("openai", "m").rotulo == "openai/m"
    assert ConfiguracaoLLM("huggingface", "org/m").rotulo == "huggingface/org/m"
    q4 = ConfiguracaoLLM("huggingface", "org/m", revisao="a" * 40, quantizacao="4bit")
    q8 = ConfiguracaoLLM("huggingface", "org/m", revisao="a" * 40, quantizacao="8bit")
    sem = ConfiguracaoLLM("huggingface", "org/m", revisao="a" * 40)
    assert q4.rotulo == "huggingface/org/m@" + "a" * 12 + "[4bit]"
    assert sem.rotulo == "huggingface/org/m@" + "a" * 12
    assert len({q4.rotulo, q8.rotulo, sem.rotulo}) == 3


def test_metadados_padrao_completos():
    from otft_sr.provedores_llm import _metadados_padrao

    cfg = ConfiguracaoLLM("huggingface", "org/m", revisao="a" * 40, quantizacao="4bit")
    meta = _metadados_padrao(cfg, "bnb-4bit-nf4/float16")
    assert {"provedor", "modelo", "revisao", "quantizacao", "dtype", "parametros_geracao",
            "determinismo"} <= set(meta)
    assert meta["dtype"] == "float16"


# ----------------------------------------------------------------------------- fábrica


@pytest.mark.parametrize(
    "cfg",
    [
        ConfiguracaoLLM("fake", "simulado"),
        ConfiguracaoLLM("ollama", "mistral:7b-instruct"),
        ConfiguracaoLLM("huggingface", "org/modelo", quantizacao="8bit"),
        ConfiguracaoLLM("openai", "gpt-4o-mini"),
        ConfiguracaoLLM("anthropic", "claude-haiku"),
        ConfiguracaoLLM("google", "gemini-flash"),
    ],
    ids=lambda c: c.provedor,
)
def test_fabrica_sem_dependencias_satisfaz_protocolo(cfg):
    provedor = criar_provedor(cfg, registro=RegistroUso())
    assert isinstance(provedor, TextGenerationBackend)
    meta = provedor.model_metadata()
    for chave in ("backend", "checkpoint", "revision", "quantization", "infra"):
        assert chave in meta
    for chave in ("provedor", "modelo", "revisao", "quantizacao", "parametros_geracao"):
        assert chave in meta
    assert meta["determinismo"] in ("deterministico", "nao_garantido")
    json.dumps(meta)


def _ausente(monkeypatch, *modulos):
    for m in modulos:
        monkeypatch.setitem(sys.modules, m, None)  # import levanta ImportError


@pytest.mark.parametrize(
    "provedor, modelo, modulo",
    [("openai", "m", "openai"), ("anthropic", "m", "anthropic"), ("google", "m", "google.genai")],
)
def test_sdk_ausente_erro_claro_no_uso(monkeypatch, provedor, modelo, modulo):
    _ausente(monkeypatch, modulo)
    p = criar_provedor(ConfiguracaoLLM(provedor, modelo), registro=RegistroUso())
    with pytest.raises(BackendError, match="pip install"):
        p.generate([("s", "u")])


def test_hf_sem_transformers_erro_claro_no_uso(monkeypatch):
    _ausente(monkeypatch, "torch", "transformers")
    p = criar_provedor(ConfiguracaoLLM("huggingface", "org/m"), registro=RegistroUso())
    assert not p.loaded
    with pytest.raises(BackendError, match="transformers"):
        p.generate([("s", "u")])


# ------------------------------------------------------------------- SDKs falsos (APIs)


class _Erro(Exception):
    pass


class APITimeoutError(_Erro):
    """Mesmo nome da classe transitória dos SDKs OpenAI/Anthropic."""


class AuthenticationError(_Erro):
    status_code = 401


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def _sdk_openai(falhas: list[Exception], chamadas: list[dict]):
    class Completions:
        def create(self, **kw):
            chamadas.append(kw)
            if falhas:
                raise falhas.pop(0)
            return _ns(
                choices=[_ns(message=_ns(content=RESPOSTA_JSON))],
                usage=_ns(prompt_tokens=11, completion_tokens=7),
                model="gpt-servido-2026",
            )

    class OpenAI:
        def __init__(self, api_key, timeout, max_retries):
            assert api_key == CHAVE and max_retries == 0
            self.chat = _ns(completions=Completions())
            self.fechado = False

        def close(self):
            self.fechado = True

    return types.SimpleNamespace(OpenAI=OpenAI)


def _sdk_anthropic(falhas, chamadas):
    class Messages:
        def create(self, **kw):
            chamadas.append(kw)
            if falhas:
                raise falhas.pop(0)
            return _ns(
                content=[_ns(type="text", text=RESPOSTA_JSON)],
                usage=_ns(input_tokens=13, output_tokens=5),
                model="claude-servido",
            )

    class Anthropic:
        def __init__(self, api_key, timeout, max_retries):
            assert api_key == CHAVE and max_retries == 0
            self.messages = Messages()

    return types.SimpleNamespace(Anthropic=Anthropic)


def _sdk_google(falhas, chamadas):
    class Models:
        def generate_content(self, **kw):
            chamadas.append(kw)
            if falhas:
                raise falhas.pop(0)
            return _ns(
                text=RESPOSTA_JSON,
                usage_metadata=_ns(prompt_token_count=17, candidates_token_count=3),
                model_version="gemini-servido",
            )

    class Client:
        def __init__(self, api_key, http_options):
            assert api_key == CHAVE and http_options["timeout"] > 0
            self.models = Models()

    return types.SimpleNamespace(Client=Client)


SDKS = {
    "openai": ("openai", _sdk_openai, "OPENAI_API_KEY", (11, 7)),
    "anthropic": ("anthropic", _sdk_anthropic, "ANTHROPIC_API_KEY", (13, 5)),
    "google": ("google.genai", _sdk_google, "GOOGLE_API_KEY", (17, 3)),
}


@pytest.fixture
def api(monkeypatch):
    """Instala o SDK falso do provedor e a chave no ambiente; devolve (provedor, chamadas)."""

    def montar(provedor, falhas=(), **cfg_kw):
        modulo, fabrica, var, _ = SDKS[provedor]
        chamadas: list[dict] = []
        monkeypatch.setitem(sys.modules, modulo, fabrica(list(falhas), chamadas))
        monkeypatch.setenv(var, CHAVE)
        esperas: list[float] = []
        registro = RegistroUso()
        cfg = ConfiguracaoLLM(provedor, f"modelo-{provedor}", espera_inicial_s=0.5, **cfg_kw)
        p = criar_provedor(cfg, registro=registro, dormir=esperas.append)
        return p, chamadas, esperas, registro

    return montar


@pytest.mark.parametrize("provedor", sorted(SDKS))
def test_api_gera_registra_tokens_e_metadados(api, provedor, caplog):
    caplog.set_level(logging.DEBUG)
    p, chamadas, esperas, registro = api(provedor)
    assert p.generate([("sys", "u1"), ("sys", "u2")]) == [RESPOSTA_JSON] * 2
    assert len(chamadas) == 2 and esperas == []
    assert chamadas[0]["model"] == f"modelo-{provedor}"
    uso = registro.obter(f"{provedor}/modelo-{provedor}")
    tin, tout = SDKS[provedor][3]
    assert (uso.chamadas, uso.falhas, uso.tokens_entrada, uso.tokens_saida) == (
        2,
        0,
        2 * tin,
        2 * tout,
    )
    meta = p.model_metadata()
    assert meta["determinismo"] == NAO_GARANTIDO
    assert meta["modelo_servido"]
    assert meta["variavel_credencial"] == SDKS[provedor][2]
    # A chave nunca aparece em metadados, repr, registro, atributos ou logs.
    for texto in (
        json.dumps(meta),
        repr(p),
        json.dumps(registro.como_dict()),
        repr(vars(p)),
        caplog.text,
    ):
        assert CHAVE not in texto


def test_parametros_enviados_por_provedor(api):
    p, ch_o, *_ = api("openai")
    p.generate([("S", "U")])
    assert ch_o[0]["temperature"] == 0.0 and ch_o[0]["seed"] == 0
    assert ch_o[0]["messages"][0] == {"role": "system", "content": "S"}
    p, ch_a, *_ = api("anthropic")
    p.generate([("S", "U")])
    assert ch_a[0]["system"] == "S" and "seed" not in ch_a[0]
    p, ch_g, *_ = api("google")
    p.generate([("S", "U")])
    assert ch_g[0]["config"]["system_instruction"] == "S" and ch_g[0]["contents"] == "U"


@pytest.mark.parametrize("provedor", sorted(SDKS))
def test_api_retentativa_com_backoff_em_erro_transitorio(api, provedor):
    p, chamadas, esperas, registro = api(
        provedor, falhas=[APITimeoutError("t"), TimeoutError("t")]
    )
    assert p.generate([("s", "u")]) == [RESPOSTA_JSON]
    assert len(chamadas) == 3
    assert esperas == [0.5, 1.0]  # exponencial
    uso = registro.obter(p.cfg.rotulo)
    assert (uso.chamadas, uso.retentativas, uso.falhas) == (1, 2, 0)


def test_api_esgota_tentativas(api):
    p, chamadas, esperas, registro = api(
        "openai", falhas=[APITimeoutError("x")] * 5, max_tentativas=3
    )
    with pytest.raises(BackendError, match="3 tentativa"):
        p.generate([("s", "u")])
    assert len(chamadas) == 3 and len(esperas) == 2
    assert registro.obter(p.cfg.rotulo).falhas == 1


def test_api_erro_permanente_nao_repete_e_mascara_chave(api):
    p, chamadas, esperas, _ = api(
        "anthropic", falhas=[AuthenticationError(f"chave inválida {CHAVE}")]
    )
    with pytest.raises(BackendError) as info:
        p.generate([("s", "u")])
    assert len(chamadas) == 1 and esperas == []
    assert "HTTP 401" in str(info.value)
    assert CHAVE not in str(info.value)
    assert info.value.__cause__ is None


@pytest.mark.parametrize("status, esperado", [(429, True), (503, True), (400, False)])
def test_classificacao_de_erro_transitorio(status, esperado):
    erro = _Erro("x")
    erro.status_code = status
    assert pl.erro_transitorio(erro) is esperado


def test_credencial_ausente_erro_claro(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", _sdk_openai([], []))
    monkeypatch.delenv("MINHA_CHAVE", raising=False)
    monkeypatch.setattr(pl, "_ler_segredo_colab", lambda nome: None)
    p = criar_provedor(
        ConfiguracaoLLM("openai", "m", variavel_credencial="MINHA_CHAVE"), registro=RegistroUso()
    )
    with pytest.raises(BackendConfigurationError, match="MINHA_CHAVE"):
        p.generate([("s", "u")])


def test_credencial_lida_dos_segredos_do_colab(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    userdata = types.SimpleNamespace(get=lambda nome: CHAVE if nome == "OPENAI_API_KEY" else None)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", userdata)
    assert pl.ler_credencial("OPENAI_API_KEY") == CHAVE


def test_close_api_fecha_cliente(api):
    p, *_ = api("openai")
    p.generate([("s", "u")])
    cliente = p._cliente_sdk
    p.close()
    assert cliente.fechado and p._cliente_sdk is None


# --------------------------------------------------------- Hugging Face com módulos falsos


class _Enc(dict):
    def to(self, _device):
        return self


def _hf_falso(registro_chamadas: dict):
    torch = types.ModuleType("torch")
    torch.float16, torch.bfloat16, torch.float32 = "f16", "bf16", "f32"
    torch.manual_seed = lambda s: registro_chamadas.setdefault("sementes", []).append(s)

    @contextmanager
    def no_grad():
        yield

    torch.no_grad = no_grad
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: registro_chamadas.__setitem__("empty_cache", True),
    )

    class Tok:
        chat_template = None
        pad_token = None
        eos_token = "<eos>"
        pad_token_id = 0

        @classmethod
        def from_pretrained(cls, nome, revision, trust_remote_code):
            assert trust_remote_code is False
            return cls()

        def __call__(self, textos, **kw):
            ids = np.array([[0, 5, 6]] * len(textos))
            return _Enc(input_ids=ids, attention_mask=(ids != 0).astype(int))

        def batch_decode(self, novos, skip_special_tokens):
            return [RESPOSTA_JSON] * len(novos)

    class Modelo:
        device = "cuda:0"

        def eval(self):
            return self

        def generate(self, **kw):
            registro_chamadas["gerar"] = kw
            n = kw["input_ids"].shape[0]
            return np.concatenate([kw["input_ids"], np.full((n, 4), 9)], axis=1)

    class Auto:
        @staticmethod
        def from_pretrained(caminho, **kw):
            registro_chamadas["carregar"] = kw
            return Modelo()

    class BitsAndBytesConfig:
        def __init__(self, **kw):
            self.kw = kw

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = Tok
    transformers.AutoModelForCausalLM = Auto
    transformers.BitsAndBytesConfig = BitsAndBytesConfig
    transformers.__version__ = "falso"
    return torch, transformers


@pytest.mark.parametrize("quant", [None, "4bit", "8bit"])
def test_hf_quantizacao_gulosa_semente_e_close(monkeypatch, tmp_path, quant):
    reg: dict = {}
    torch, transformers = _hf_falso(reg)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))
    monkeypatch.setattr(pl, "download_verified_snapshot", lambda repo, rev: tmp_path)
    registro = RegistroUso()
    cfg = ConfiguracaoLLM("huggingface", "org/m", quantizacao=quant, semente=7, tamanho_lote=2)
    with usar_provedor(cfg, registro=registro) as p:
        assert p.generate([("s", "u")] * 3) == [RESPOSTA_JSON] * 3
        assert p.loaded
        meta = p.model_metadata()
    assert not p.loaded and reg["empty_cache"]  # close liberou a GPU
    assert reg["sementes"] == [7, 7]  # um manual_seed por lote
    assert reg["gerar"]["do_sample"] is False and "temperature" not in reg["gerar"]
    carregar = reg["carregar"]
    assert carregar["trust_remote_code"] is False
    if quant is None:
        assert "quantization_config" not in carregar and meta["quantization"] == "float16"
    elif quant == "4bit":
        kw = carregar["quantization_config"].kw
        assert kw["load_in_4bit"] and kw["bnb_4bit_quant_type"] == "nf4"
        assert meta["quantization"].startswith("bnb-4bit")
    else:
        assert carregar["quantization_config"].kw == {"load_in_8bit": True}
    assert meta["determinismo"] == "deterministico"
    uso = registro.obter(p.cfg.rotulo)  # inclui [quantizacao]
    assert uso.chamadas == 3 and uso.tokens_entrada == 6 and uso.tokens_saida == 12


def test_hf_quantizacao_sem_bitsandbytes_erro_claro(monkeypatch, tmp_path):
    torch, transformers = _hf_falso({})
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    _ausente(monkeypatch, "bitsandbytes")
    monkeypatch.setattr(pl, "download_verified_snapshot", lambda repo, rev: tmp_path)
    p = criar_provedor(ConfiguracaoLLM("huggingface", "org/m", quantizacao="4bit"))
    with pytest.raises(BackendError, match="bitsandbytes"):
        p.generate([("s", "u")])


def test_hf_temperatura_positiva_amostra(monkeypatch, tmp_path):
    reg: dict = {}
    torch, transformers = _hf_falso(reg)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(pl, "download_verified_snapshot", lambda repo, rev: tmp_path)
    p = criar_provedor(ConfiguracaoLLM("huggingface", "org/m", temperatura=0.7))
    p.generate([("s", "u")])
    assert reg["gerar"]["do_sample"] is True and reg["gerar"]["temperature"] == 0.7
    assert p.model_metadata()["determinismo"] == NAO_GARANTIDO


# ------------------------------------------------------------------ fake, Ollama, ciclo


def test_fake_aceita_funcao_e_registra_uso():
    registro = RegistroUso()
    cfg = ConfiguracaoLLM("fake", "sim", resposta_fake=lambda s, u: f"{s}|{u}")
    p = criar_provedor(cfg, registro=registro)
    assert p.generate([("a", "b")]) == ["a|b"]
    assert registro.obter("fake/sim").chamadas == 1
    assert registro.por_provedor()["fake"]["chamadas"] == 1


def test_ollama_reusa_backend_do_pacote_com_sessao_simulada():
    class Resp:
        def __init__(self, dados):
            self.dados = dados

        def raise_for_status(self):
            pass

        def json(self):
            return self.dados

    class Sessao:
        fechada = False

        def post(self, url, json, timeout):
            assert json["options"]["temperature"] == 0.0
            return Resp({"message": {"content": RESPOSTA_JSON}})

        def get(self, url, timeout):
            return Resp({"models": [{"name": "mistral:7b-instruct", "digest": "abc123"}]})

        def close(self):
            Sessao.fechada = True

    cfg = ConfiguracaoLLM("ollama", "mistral:7b-instruct")
    with usar_provedor(cfg, registro=RegistroUso(), sessao=Sessao()) as p:
        p.resolve_identity()
        assert p.generate([("s", "u")]) == [RESPOSTA_JSON]
        meta = p.model_metadata()
    assert meta["backend"] == "ollama" and meta["digest"] == "abc123"
    assert meta["revisao"] == "abc123" and meta["determinismo"] == "deterministico"
    assert Sessao.fechada


def test_usar_provedor_fecha_mesmo_com_excecao():
    cfg = ConfiguracaoLLM("fake", "sim")
    with pytest.raises(RuntimeError), usar_provedor(cfg, registro=RegistroUso()) as p:
        raise RuntimeError("falha no meio")
    assert p.fechado


# ----------------------------------------------------------------------------- ambiente


def test_relatorio_ambiente_sem_torch():
    rel = relatorio_ambiente()
    json.dumps(rel)
    for chave in ("colab", "cuda_disponivel", "gpus", "memoria_gpu_gb", "versoes"):
        assert chave in rel
    assert set(rel["versoes"]) >= {"torch", "transformers", "bitsandbytes"}


@pytest.mark.parametrize(
    "mem, n_b, esperado",
    [(40.0, 7, None), (15.0, 7, "8bit"), (15.0, 13, "4bit"), (80.0, 32, None), (16.0, 3, None)],
)
def test_recomendar_quantizacao(mem, n_b, esperado):
    assert recomendar_quantizacao(mem, n_b) == esperado


@pytest.mark.parametrize("mem, n_b", [(0.0, 7), (8.0, 70), (16.0, 0)])
def test_recomendar_quantizacao_recusa(mem, n_b):
    with pytest.raises(ValueError):
        recomendar_quantizacao(mem, n_b)


# --------------------------------------------------------------------------- M1 (fake)


def test_m1_fake_ponta_a_ponta_com_cache_por_modelo(tmp_path):
    cfg = ConfiguracaoLLM("fake", "sim-m1", resposta_fake=RESPOSTA_JSON)
    av = criar_avaliador_m1(cfg, "C", dir_cache=tmp_path, registro=RegistroUso())
    assert isinstance(av, LLMEquationEvaluator)
    n3 = nivel("N3")
    resultado = av.evaluate(
        parse_infix(n3.expressao, FEATURES), contexto_fisico(n3), VARIANTE_PROMPT
    )
    assert isinstance(resultado, EquationEvaluation)
    assert resultado.scores == (0.9, 0.8, 0.7)
    arquivo = tmp_path / "avaliacoes_C__fake__sim-m1.jsonl"
    assert arquivo.exists() and arquivo.read_text().strip()
    outro = ConfiguracaoLLM("fake", "outro")
    assert pl.caminho_cache(outro, "C", tmp_path) != arquivo


def test_criar_avaliador_legado_mantem_arquivo_de_cache(tmp_path):
    from physics_llm_sr.llm.inference import FakeBackend

    av = criar_avaliador("B", backend=FakeBackend(RESPOSTA_JSON), dir_cache=tmp_path)
    n1 = nivel("N1")
    av.evaluate(parse_infix(n1.expressao, FEATURES), contexto_fisico(n1), VARIANTE_PROMPT)
    assert (tmp_path / "avaliacoes_B.jsonl").exists()
    assert av.max_attempts == 2  # max_retries=1 preservado
