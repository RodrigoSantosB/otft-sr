"""Configuração única e plano de ensaios do notebook 02 (multi-LLM, M1 e M2).

:class:`ConfiguracaoExperimento` reúne, validados na construção, todos os fatores da
campanha experimental: modelos, modos, condições, níveis, algoritmos de RS, sementes,
ruídos, tamanho dos dados, parâmetros do M2 e a formulação da perda do M1. As funções
daqui não executam ensaios: montam o plano, as chaves de retomada, a configuração mínima
de verificação, a quantização por GPU, a importação da linha de base e o texto das
conclusões.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import math
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from otft_sr.contexto import MODELO_OLLAMA, URL_OLLAMA
from otft_sr.experimento import ALGORITMOS_RS, CONDICOES, CONFIGURACOES_BUSCA, MODELO_SEM_LLM
from otft_sr.formulacao import FORMULACOES, nivel_formulado
from otft_sr.gerador import ALGORITMO_M2, CONDICOES_GERACAO
from otft_sr.niveis import escada, nivel
from otft_sr.persistencia import RegistroExperimento, carregar_resultados
from otft_sr.provedores_llm import ConfiguracaoLLM, recomendar_quantizacao, responder_simulado

MODOS: tuple[str, ...] = ("M1", "M2")
NIVEIS_VALIDOS: tuple[str, ...] = tuple(n.id for n in escada())

#: Configuração de busca mínima das execuções de verificação (não constitui resultado).
BUSCA_VERIFICACAO_MINIMA: dict[str, Any] = {
    "sr.population": 8, "sr.generations": 1, "sr.early_stopping.enabled": False}

#: Modelos padrão (Hugging Face, revisão fixada por SHA do commit no Hub) e tamanho em
#: bilhões de parâmetros (safetensors do Hub), usado na escolha da quantização.
MISTRAL_7B = ConfiguracaoLLM(
    "huggingface", "mistralai/Mistral-7B-Instruct-v0.1",
    revisao="ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca")  # mesma revisão do physics-llm-sr
QWEN25_3B = ConfiguracaoLLM(
    "huggingface", "Qwen/Qwen2.5-3B-Instruct", revisao="aa8e72537993ba99e69dfaafa59ed015b17504d1")
PHI3_MINI = ConfiguracaoLLM(
    "huggingface", "microsoft/Phi-3-mini-4k-instruct",
    revisao="f39ac1d28e925b323eae81227eaba4464caced4e")
MODELOS_PADRAO: tuple[ConfiguracaoLLM, ...] = (MISTRAL_7B, QWEN25_3B, PHI3_MINI)
N_PARAMETROS_B: dict[str, float] = {
    MISTRAL_7B.modelo: 7.24, QWEN25_3B.modelo: 3.09, PHI3_MINI.modelo: 3.82}

SIMULADO = ConfiguracaoLLM("fake", "simulado", resposta_fake=responder_simulado)
OLLAMA_MISTRAL = ConfiguracaoLLM(
    "ollama", MODELO_OLLAMA, url_base=URL_OLLAMA, max_novos_tokens=256, timeout_s=300.0)


def _tupla(x: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(x)


def _subconjunto(nome: str, valores: Sequence[Any], validos: Sequence[Any]) -> None:
    if not valores:
        raise ValueError(f"{nome} não pode ser vazio")
    fora = [v for v in valores if v not in validos]
    if fora:
        raise ValueError(f"{nome}: valores inválidos {fora} (válidos: {list(validos)})")
    if len(set(valores)) != len(valores):
        raise ValueError(f"{nome}: valores repetidos")


def _busca_valida(nome: str, busca: Any) -> None:
    if isinstance(busca, str):
        if busca not in CONFIGURACOES_BUSCA:
            raise ValueError(f"{nome} deve ser um de {list(CONFIGURACOES_BUSCA)} ou um dict")
    elif not isinstance(busca, Mapping):
        raise TypeError(f"{nome} deve ser str ou dict")


@dataclass(frozen=True)
class ConfiguracaoExperimento:
    """Fatores da campanha (validados). ``niveis_llm``/``algoritmos_rs_llm`` restringem os
    ensaios com LLM (M1 B/C e M2); ``None`` = os mesmos de ``niveis``/``algoritmos_rs``."""

    nome_experimento: str
    modelos: tuple[ConfiguracaoLLM, ...] = MODELOS_PADRAO
    n_parametros_b: Mapping[str, float] = field(default_factory=lambda: dict(N_PARAMETROS_B))
    modos: tuple[str, ...] = MODOS
    condicoes_m1: tuple[str, ...] = CONDICOES
    condicoes_m2: tuple[str, ...] = CONDICOES_GERACAO
    niveis: tuple[str, ...] = NIVEIS_VALIDOS
    niveis_llm: tuple[str, ...] | None = None
    algoritmos_rs: tuple[str, ...] = ("deap", "gplearn")
    algoritmos_rs_llm: tuple[str, ...] | None = ("deap",)
    configuracao_busca: str | Mapping[str, Any] = "calibrada"
    configuracao_busca_llm: str | Mapping[str, Any] | None = None
    sementes: tuple[int, ...] = (0, 1, 2)
    ruidos: tuple[float, ...] = (0.0,)
    n_curvas: int = 20
    n_pontos: int = 50
    n_candidatos: int = 5
    n_iteracoes_refinamento: int = 3
    max_novos_tokens_m2: int = 512
    formulacao_perda: str = "linear"
    execucao_de_verificacao: bool = False

    def __post_init__(self) -> None:
        """Normaliza listas em tuplas e recusa combinações inválidas com mensagem clara."""
        for nome in ("modelos", "modos", "condicoes_m1", "condicoes_m2", "niveis", "algoritmos_rs",
                     "sementes", "ruidos"):
            object.__setattr__(self, nome, _tupla(getattr(self, nome)))
        for nome in ("niveis_llm", "algoritmos_rs_llm"):
            if getattr(self, nome) is not None:
                object.__setattr__(self, nome, _tupla(getattr(self, nome)))
        if not self.nome_experimento or not all(c.isalnum() or c in "_-" for c in self.nome_experimento):
            raise ValueError("nome_experimento: só letras, dígitos, _ ou -")
        _subconjunto("modos", self.modos, MODOS)
        _subconjunto("condicoes_m1", self.condicoes_m1, CONDICOES)
        _subconjunto("condicoes_m2", self.condicoes_m2, CONDICOES_GERACAO)
        _subconjunto("niveis", self.niveis, NIVEIS_VALIDOS)
        _subconjunto("niveis_llm", self.niveis_llm_efetivos, self.niveis)
        _subconjunto("algoritmos_rs", self.algoritmos_rs, ALGORITMOS_RS)
        _subconjunto("algoritmos_rs_llm", self.algoritmos_rs_llm_efetivos, self.algoritmos_rs)
        _busca_valida("configuracao_busca", self.configuracao_busca)
        if self.configuracao_busca_llm is not None:
            _busca_valida("configuracao_busca_llm", self.configuracao_busca_llm)
        if self.formulacao_perda not in FORMULACOES:
            raise ValueError(f"formulacao_perda deve ser uma de {FORMULACOES}")
        if not all(isinstance(m, ConfiguracaoLLM) for m in self.modelos):
            raise TypeError("modelos: cada item deve ser ConfiguracaoLLM")
        if len({m.rotulo for m in self.modelos}) != len(self.modelos):
            raise ValueError("modelos: rótulos provedor/modelo repetidos")
        if self.usa_llm and not self.modelos:
            raise ValueError("há condições com LLM (B/C ou M2) mas nenhum modelo")
        if any(not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in self.sementes):
            raise ValueError("sementes: inteiros >= 0")
        if len(set(self.sementes)) != len(self.sementes):
            raise ValueError("sementes: valores repetidos")
        if any(not 0.0 <= float(r) < 1.0 for r in self.ruidos):
            raise ValueError("ruidos: frações em [0, 1)")
        for nome in ("n_curvas", "n_pontos", "n_candidatos", "max_novos_tokens_m2"):
            v = getattr(self, nome)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"{nome} deve ser inteiro >= 1")
        if self.n_curvas < 5:
            raise ValueError("n_curvas >= 5 (partição 60/20/20 por curva)")
        if not isinstance(self.n_iteracoes_refinamento, int) or self.n_iteracoes_refinamento < 0:
            raise ValueError("n_iteracoes_refinamento deve ser inteiro >= 0")
        for m in self.modelos:
            if m.provedor == "huggingface" and m.modelo not in self.n_parametros_b:
                raise ValueError(f"n_parametros_b sem o tamanho de {m.modelo} (bilhões)")

    # ------------------------------------------------------------------ derivados

    @property
    def niveis_llm_efetivos(self) -> tuple[str, ...]:
        """Níveis dos ensaios com LLM."""
        return self.niveis if self.niveis_llm is None else self.niveis_llm

    @property
    def algoritmos_rs_llm_efetivos(self) -> tuple[str, ...]:
        """Algoritmos de RS das condições B/C."""
        return self.algoritmos_rs if self.algoritmos_rs_llm is None else self.algoritmos_rs_llm

    @property
    def busca_llm_efetiva(self) -> str | Mapping[str, Any]:
        """Configuração de busca das condições B/C."""
        return self.configuracao_busca if self.configuracao_busca_llm is None else self.configuracao_busca_llm

    @property
    def condicoes_m1_llm(self) -> tuple[str, ...]:
        """Condições do M1 com LLM (B/C) efetivamente pedidas."""
        return tuple(c for c in self.condicoes_m1 if c != "A") if "M1" in self.modos else ()

    @property
    def usa_llm(self) -> bool:
        """``True`` se algum ensaio chama LLM."""
        return bool(self.condicoes_m1_llm) or "M2" in self.modos

    @property
    def id_experimento(self) -> str:
        """Id fixo (retomada): ``<nome>`` ou ``<nome>_verificacao``."""
        return self.nome_experimento + ("_verificacao" if self.execucao_de_verificacao else "")

    def como_dict(self) -> dict[str, Any]:
        """Representação JSON (modelos via ``ConfiguracaoLLM.como_dict``)."""
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        d["modelos"] = [m.como_dict() for m in self.modelos]
        d["n_parametros_b"] = dict(self.n_parametros_b)
        for k in ("configuracao_busca", "configuracao_busca_llm"):
            if isinstance(d[k], Mapping):
                d[k] = dict(d[k])
        return json.loads(json.dumps(d, default=list))


def configuracao_de_verificacao(
    cfg: ConfiguracaoExperimento, backend: str = "simulado"
) -> ConfiguracaoExperimento:
    """Configuração mínima de verificação (N3, 1 semente, sem ruído, DEAP mínimo, M2 com 2
    candidatos e 1 rodada de refinamento) com backend ``simulado`` ou ``ollama``
    (``mistral:7b-instruct`` local). Não constitui resultado."""
    modelos = {"simulado": (SIMULADO,), "ollama": (OLLAMA_MISTRAL,)}
    if backend not in modelos:
        raise ValueError(f"backend de verificação deve ser um de {list(modelos)}")
    return dataclasses.replace(
        cfg, modelos=modelos[backend], niveis=("N3",), niveis_llm=None, algoritmos_rs=("deap",),
        algoritmos_rs_llm=None, configuracao_busca=dict(BUSCA_VERIFICACAO_MINIMA),
        configuracao_busca_llm=None, sementes=(0,), ruidos=(0.0,), n_curvas=10, n_pontos=20,
        n_candidatos=2, n_iteracoes_refinamento=1, execucao_de_verificacao=True)


def aplicar_quantizacao(
    cfg: ConfiguracaoExperimento, memoria_gpu_gb: float
) -> tuple[ConfiguracaoExperimento, pd.DataFrame]:
    """Quantização de cada modelo Hugging Face por ``recomendar_quantizacao`` (menor
    compressão que cabe na GPU). Sem GPU ou sem caber: ``ValueError`` com a instrução."""
    novos, linhas = [], []
    for m in cfg.modelos:
        if m.provedor != "huggingface":
            novos.append(m)
            linhas.append({"modelo": m.rotulo, "quantizacao": None, "motivo": "não se aplica"})
            continue
        n_b = float(cfg.n_parametros_b[m.modelo])
        q = recomendar_quantizacao(memoria_gpu_gb, n_b)
        novos.append(dataclasses.replace(m, quantizacao=q))
        linhas.append({"modelo": m.rotulo, "n_parametros_b": n_b, "memoria_gpu_gb": memoria_gpu_gb,
                       "quantizacao": q or "fp16 (sem quantização)", "motivo": "recomendar_quantizacao"})
    return dataclasses.replace(cfg, modelos=tuple(novos)), pd.DataFrame(linhas)


# ------------------------------------------------------------------------ plano e chaves


def campos_m1(nivel_id: str, condicao: str, algoritmo: str, modelo: str, ruido: float,
              semente: int, formulacao_perda: str) -> dict[str, Any]:
    """Campos (e chave de retomada) de um ensaio M1."""
    return {"nivel": nivel_id, "modo": "M1", "condicao": condicao, "algoritmo": algoritmo,
            "modelo": modelo, "ruido": float(ruido), "semente": int(semente),
            "formulacao_perda": formulacao_perda}


def campos_m2(nivel_id: str, condicao: str, modelo: str, ruido: float, semente: int) -> dict[str, Any]:
    """Campos (e chave de retomada) de um ensaio M2."""
    return {"nivel": nivel_id, "modo": "M2", "condicao": condicao, "algoritmo": ALGORITMO_M2,
            "modelo": modelo, "ruido": float(ruido), "semente": int(semente)}


def plano_de_ensaios(cfg: ConfiguracaoExperimento) -> pd.DataFrame:
    """Todos os ensaios da campanha (uma linha por ensaio, com a etapa do notebook)."""
    linhas: list[dict[str, Any]] = []
    f = cfg.formulacao_perda
    if "M1" in cfg.modos and "A" in cfg.condicoes_m1:
        for n in cfg.niveis:
            for a in cfg.algoritmos_rs:
                for r in cfg.ruidos:
                    for s in cfg.sementes:
                        linhas.append({"etapa": "M1 linha de base",
                                       **campos_m1(n, "A", a, MODELO_SEM_LLM, r, s, f)})
    for m in cfg.modelos:
        if "M2" in cfg.modos:
            for n in cfg.niveis_llm_efetivos:
                for c in cfg.condicoes_m2:
                    for r in cfg.ruidos:
                        for s in cfg.sementes:
                            linhas.append({"etapa": "M2", **campos_m2(n, c, m.rotulo, r, s)})
        for c in cfg.condicoes_m1_llm:
            for n in cfg.niveis_llm_efetivos:
                for a in cfg.algoritmos_rs_llm_efetivos:
                    for r in cfg.ruidos:
                        for s in cfg.sementes:
                            linhas.append({"etapa": "M1 com LLM",
                                           **campos_m1(n, c, a, m.rotulo, r, s, f)})
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------- registro e Drive


CAMPOS_COMPATIBILIDADE: tuple[str, ...] = (
    "condicoes_m1", "condicoes_m2", "configuracao_busca", "configuracao_busca_llm",
    "n_curvas", "n_pontos", "n_candidatos", "n_iteracoes_refinamento", "formulacao_perda",
    "execucao_de_verificacao", "modelos")
#: Campos de cada modelo que definem a sua identidade experimental (pesos e decodificação).
CAMPOS_IDENTIDADE_MODELO: tuple[str, ...] = (
    "provedor", "modelo", "revisao", "quantizacao", "dtype", "temperatura", "max_novos_tokens",
    "semente")


def _modelos_incompativeis(anteriores: Sequence[Mapping[str, Any]],
                           atuais: Sequence[Mapping[str, Any]]) -> list[str]:
    """``provedor/modelo`` presentes nas duas configurações com identidade diferente
    (revisão, quantização, dtype ou decodificação). Acrescentar ou retirar modelos é
    compatível: as chaves dos outros modelos não colidem."""
    def por_nome(ms: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        return {f"{m.get('provedor')}/{m.get('modelo')}":
                {k: m.get(k) for k in CAMPOS_IDENTIDADE_MODELO} for m in ms}
    a, b = por_nome(anteriores), por_nome(atuais)
    return sorted(n for n in a.keys() & b.keys() if a[n] != b[n])


def abrir_registro(raiz: Path | str, cfg: ConfiguracaoExperimento,
                   extras: Mapping[str, Any] | None = None) -> RegistroExperimento:
    """Cria ou retoma ``<raiz>/<id_experimento>``. Na retomada, recusa configuração
    incompatível (campos que mudam o significado dos ensaios já gravados): troque o
    ``nome_experimento`` para uma nova campanha."""
    destino = Path(raiz) / cfg.id_experimento
    atual = cfg.como_dict()
    arq = destino / "config.json"
    if arq.is_file():
        anterior = json.loads(arq.read_text("utf-8")).get("config", {}).get("experimento", {})
        dif = [k for k in CAMPOS_COMPATIBILIDADE if k != "modelos" and k in anterior
               and anterior[k] != atual[k]]
        trocados = _modelos_incompativeis(anterior.get("modelos", []), atual["modelos"])
        if trocados:
            dif.append(f"modelos ({', '.join(trocados)}: revisão, quantização, dtype ou "
                       "decodificação)")
        if dif:
            raise ValueError(
                f"{destino} já existe com configuração diferente em {dif}; use outro "
                "nome_experimento (os ensaios gravados não seriam comparáveis)")
    return RegistroExperimento(raiz, {"experimento": atual, **dict(extras or {})},
                               id_experimento=cfg.id_experimento)


def sincronizar(origem: Path | str, destino: Path | str | None) -> Path | None:
    """Copia ``origem`` (diretório) para ``destino`` (sobrescreve arquivos; nunca apaga).
    ``destino=None`` não faz nada (execução local)."""
    if destino is None or not Path(origem).exists():
        return None
    destino = Path(destino)
    shutil.copytree(origem, destino, dirs_exist_ok=True)
    return destino


# --------------------------------------------------------------------- linha de base


def importar_linha_de_base(padroes: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê resultados existentes (``carregar_resultados``: formato atual ou CSV/JSON do
    notebook 01) e mantém só ensaios M1/A cuja ``expressao_referencia`` é a referência ATUAL
    do nível (resultados anteriores à revisão de regiões ou à renumeração são descartados).

    Devolve ``(aceitos, relatorio)``; ``relatorio`` tem uma linha por arquivo.
    """
    atuais = {n.id: n.expressao for n in escada()}
    aceitos, rel = [], []
    arquivos = sorted({a for p in padroes for a in glob.glob(str(p))})
    for arq in arquivos:
        try:
            df = carregar_resultados(arq)
        except Exception as exc:  # noqa: BLE001 - arquivo ilegível é relatado, não interrompe
            rel.append({"arquivo": arq, "linhas": 0, "aceitas": 0, "erro": f"{type(exc).__name__}: {exc}"})
            continue
        ok = df[(df["condicao"] == "A") & (df["expressao_referencia"] == df["nivel"].map(atuais))]
        if "formulacao_perda" in ok:
            ok = ok[ok["formulacao_perda"].isna() | (ok["formulacao_perda"] == "linear")]
        aceitos.append(ok.assign(origem=Path(arq).name))
        rel.append({"arquivo": arq, "linhas": len(df), "aceitas": len(ok), "erro": None})
    df = pd.concat(aceitos, ignore_index=True) if aceitos else pd.DataFrame()
    return df, pd.DataFrame(rel, columns=["arquivo", "linhas", "aceitas", "erro"])


# --------------------------------------------------------------------------- conclusões


def _pct(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f} %"


def texto_conclusoes(
    taxa: pd.DataFrame,
    auditoria: pd.DataFrame,
    comparacoes: pd.DataFrame,
    cfg: ConfiguracaoExperimento,
) -> str:
    """Markdown das conclusões a partir das tabelas (taxa de recuperação com IC de Wilson,
    resumo da auditoria e comparações pareadas). Em execução de verificação, declara que não
    há resultado."""
    if cfg.execucao_de_verificacao:
        return ("**Execução de verificação (não constitui resultado).** A configuração mínima e o "
                "backend de verificação só demonstram que o pipeline executa de ponta a ponta; "
                "nenhuma taxa, diferença ou p-valor desta execução deve ser interpretado.")
    linhas = ["### Recuperação simbólica (IC 95 % de Wilson)"]
    for _, r in taxa.sort_values(["nivel", "modo", "condicao"]).iterrows():
        linhas.append(f"- {r['nivel']} {r['modo']} {r['condicao']} ({r.get('modelo') or 'sem LLM'}): "
                      f"{int(r['n_recuperados'])}/{int(r['n'])} = {_pct(r['taxa'])} "
                      f"[{_pct(r['ic95_inf'])}; {_pct(r['ic95_sup'])}]")
    if len(auditoria):
        linhas.append("\n### Auditoria do avaliador (pré-requisito de B/C)")
        for _, r in auditoria.iterrows():
            veredito = ("discrimina a referência das formas erradas" if r["discrimina_erradas"]
                        else "NÃO discrimina a referência de formas erradas: B/C deste modelo não "
                        "informam a busca sobre a forma funcional")
            linhas.append(f"- {r['modelo']} {r['condicao']} {r['nivel']}: c(ref) = "
                          f"{r['c_referencia']:.3f}, min c(errada) = {r['c_errada_min']:.3f}: {veredito}.")
    if len(comparacoes):
        linhas.append("\n### Comparações pareadas por semente")
        for _, r in comparacoes.iterrows():
            linhas.append(f"- {r['comparacao']} ({r['nivel']}): n = {r['n_pares']}, Δ RMSE log10 "
                          f"médio = {r['diferenca_media']:.3g} déc, p (Wilcoxon) = {r['p_valor']:.3g}, "
                          f"p (McNemar) = {r['p_mcnemar']:.3g} [{r['tipo']}].")
    n_s = len(cfg.sementes)
    linhas.append(f"\nCom {n_s} semente(s) por condição experimental os IC são largos; ausência de significância "
                  "não é evidência de ausência de efeito.")
    return "\n".join(linhas)


def nivel_da_formulacao(nivel_id: str, formulacao: str):  # noqa: ANN201 - atalho do notebook
    """Nível ``nivel_id`` na formulação da perda (atalho do notebook)."""
    return nivel_formulado(nivel(nivel_id), formulacao)
