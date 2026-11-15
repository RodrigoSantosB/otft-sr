"""Persistência dos ensaios M1/M2 em ``results/<id_experimento>/``.

Estrutura::

    results/<id>/config.json               configuração do experimento (saneada)
    results/<id>/environment.json          relatorio_ambiente() + commit git + versões
    results/<id>/metrics.csv               uma linha por ensaio (COLUNAS_PADRAO primeiro)
    results/<id>/metrics_summary.csv       estatistica.agregar (média, desvio, IC 95 % t)
    results/<id>/candidates.csv            candidatos do M2 (histórico completo, achatado)
    results/<id>/equations.json            expressões por ensaio e candidatos resumidos
    results/<id>/generation_metadata.json  hash dos templates, hash dos dados por ensaio
    results/<id>/execution_log.jsonl       eventos (criado, retomado, registrado, pulado)
    results/<id>/figures/

- Id único: ``<prefixo>_<AAAAMMDD-HHMMSS>_<6 hex aleatórios>``.
- Retomada: a chave do ensaio é nível x modo x condição x algoritmo x modelo x ruído x
  semente (algoritmo e ruído entram para que ensaios M1 de algoritmos de RS ou ruídos
  diferentes não colidam); ensaios já registrados são pulados por :meth:`executar`.
- Segredos: tudo o que é gravado passa por :func:`sanear` (chaves sensíveis pelo nome,
  padrões de chave de API e valores das variáveis de credencial do ambiente).
- :func:`carregar_resultados` lê o próprio formato e os CSV/JSON antigos do notebook 01,
  renomeando colunas antigas pelo mapeamento ``data/legado/colunas_notebook01.json``;
  colunas ausentes ficam NaN (nunca inventadas). Única inferência: ``modo = "M1"`` para
  exportações do notebook 01, que só executa ``executar_condicao`` (modo M1).
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata
import json
import math
import os
import re
import secrets
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from otft_sr.estatistica import agregar
from otft_sr.experimento import COLUNAS_M1
from otft_sr.gerador import COLUNAS_M2, carregar_templates
from otft_sr.gerador import hash_templates as _hash_templates
from otft_sr.ovsed import RAIZ_PROJETO
from otft_sr.provedores_llm import CREDENCIAL_PADRAO, relatorio_ambiente

ARQUIVOS: tuple[str, ...] = (
    "config.json", "environment.json", "metrics.csv", "metrics_summary.csv",
    "candidates.csv", "equations.json", "generation_metadata.json", "execution_log.jsonl",
)
DIR_FIGURAS = "figures"
COLUNAS_PADRAO: tuple[str, ...] = (*COLUNAS_M1, *COLUNAS_M2)
CAMPOS_CHAVE: tuple[str, ...] = (
    "nivel", "modo", "condicao", "algoritmo", "modelo", "ruido", "semente")
MAPEAMENTO_LEGADO = RAIZ_PROJETO / "data" / "legado" / "colunas_notebook01.json"
DIR_RESULTADOS = RAIZ_PROJETO / "results"

# ------------------------------------------------------------------------------ segredos

_CHAVE_SENSIVEL = re.compile(
    r"(api_?key|apikey|secret|senha|password|passwd|authorization|bearer|access_token"
    r"|^token$|_token$|^chave$|chave_api|credencial_valor)",
    re.IGNORECASE,
)
_PADROES_SEGREDO = re.compile(
    r"(?<![A-Za-z0-9])(sk-ant-[A-Za-z0-9_\-]{8,}|sk-[A-Za-z0-9_\-]{8,}"
    r"|AIza[0-9A-Za-z_\-]{10,}|hf_[A-Za-z0-9]{10,})"
    r"|Bearer\s+\S+"
)
MASCARA = "***"


def _valores_credenciais() -> list[str]:
    nomes = set(CREDENCIAL_PADRAO.values())
    return [v for n in nomes if (v := os.environ.get(n)) and len(v) >= 8]


def sanear(obj: Any, _creds: list[str] | None = None) -> Any:
    """Cópia de ``obj`` sem segredos: valores de chaves sensíveis viram ``***``; trechos com
    formato de chave de API e valores das variáveis de credencial do ambiente são mascarados.
    Hashes, contadores e o NOME da variável de credencial são preservados."""
    creds = _valores_credenciais() if _creds is None else _creds
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and _CHAVE_SENSIVEL.search(k) and v not in (None, ""):
                out[k] = MASCARA
            else:
                out[k] = sanear(v, creds)
        return out
    if isinstance(obj, (list, tuple)):
        return [sanear(v, creds) for v in obj]
    if isinstance(obj, str):
        s = _PADROES_SEGREDO.sub(MASCARA, obj)
        for c in creds:
            s = s.replace(c, MASCARA)
        return s
    return obj


# ------------------------------------------------------------------------- utilidades


def novo_id_experimento(prefixo: str = "exp") -> str:
    """``<prefixo>_<AAAAMMDD-HHMMSS>_<6 hex>`` (data/hora local + sufixo aleatório)."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", prefixo):
        raise ValueError("prefixo deve conter só letras, dígitos, _ ou -")
    return f"{prefixo}_{_dt.datetime.now():%Y%m%d-%H%M%S}_{secrets.token_hex(3)}"


def _agora() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def info_git(raiz: Path = RAIZ_PROJETO) -> dict[str, Any]:
    """Commit, ramo e estado (alterações não registradas) do repositório, se houver git."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(raiz), *args], capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
    try:
        return {"disponivel": True, "commit": git("rev-parse", "HEAD"),
                "ramo": git("rev-parse", "--abbrev-ref", "HEAD"),
                "alteracoes_pendentes": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"disponivel": False, "commit": None}


def _versoes() -> dict[str, str | None]:
    out = {}
    for p in ("numpy", "scipy", "pandas", "sympy", "pydantic", "physics-llm-sr", "requests"):
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = None
    return out


def _json_padrao(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _limpar_nan(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Mapping):
        return {k: _limpar_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_limpar_nan(v) for v in obj]
    return obj


def _escrever_json(caminho: Path, obj: Any) -> None:
    texto = json.dumps(_limpar_nan(sanear(obj)), ensure_ascii=False, indent=1, default=_json_padrao)
    tmp = caminho.with_suffix(caminho.suffix + ".tmp")
    tmp.write_text(texto, encoding="utf-8")
    tmp.replace(caminho)


def _escrever_csv(caminho: Path, df: pd.DataFrame) -> None:
    creds = _valores_credenciais()
    df = df.apply(lambda col: col.map(lambda v: sanear(v, creds) if isinstance(v, str) else v))
    tmp = caminho.with_suffix(caminho.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(caminho)


def _valor_csv(v: Any) -> Any:
    """Valor gravado no CSV: estruturas viram JSON."""
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(_limpar_nan(v), ensure_ascii=False, default=_json_padrao)
    return v


def _norm_chave(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        return format(float(v), ".10g")
    return str(v)


def chave_ensaio(linha: Mapping[str, Any]) -> str:
    """Chave do ensaio (``campo=valor`` de :data:`CAMPOS_CHAVE`, separados por ``|``)."""
    return "|".join(f"{k}={_norm_chave(linha.get(k))}" for k in CAMPOS_CHAVE)


def _ordenar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    extras = [c for c in df.columns if c not in COLUNAS_PADRAO]
    return df.reindex(columns=[*COLUNAS_PADRAO, *extras])


# ----------------------------------------------------------------------------- registro


class RegistroExperimento:
    """Diretório ``<raiz>/<id_experimento>`` com escrita incremental e retomada."""

    def __init__(
        self,
        raiz: Path | str = DIR_RESULTADOS,
        config: Mapping[str, Any] | None = None,
        *,
        prefixo: str = "exp",
        id_experimento: str | None = None,
        hash_templates: Mapping[str, Any] | None = None,
    ) -> None:
        """Cria o experimento (ou reabre, se ``id_experimento`` já existe em ``raiz``)."""
        self.diretorio = Path(raiz) / (id_experimento or novo_id_experimento(prefixo))
        existente = (self.diretorio / "config.json").is_file()
        self.diretorio.mkdir(parents=True, exist_ok=True)
        (self.diretorio / DIR_FIGURAS).mkdir(exist_ok=True)
        if not existente:
            _escrever_json(self.diretorio / "config.json", {
                "id_experimento": self.diretorio.name, "criado_em": _agora(),
                "config": dict(config or {})})
            _escrever_json(self.diretorio / "environment.json", {
                **relatorio_ambiente(), "git": info_git(), "pacotes": _versoes()})
            ht = dict(hash_templates) if hash_templates is not None else (
                _hash_templates(carregar_templates()))
            _escrever_json(self.diretorio / "generation_metadata.json",
                           {"hash_templates": ht, "ensaios": {}})
            (self.diretorio / "execution_log.jsonl").touch()
            self.log("experimento_criado")
        self._linhas: list[dict[str, Any]] = []
        self._candidatos: list[dict[str, Any]] = []
        m = self.diretorio / "metrics.csv"
        if m.is_file():
            self._linhas = pd.read_csv(m).to_dict("records")
        c = self.diretorio / "candidates.csv"
        if c.is_file():
            self._candidatos = pd.read_csv(c).to_dict("records")
        self._chaves = {chave_ensaio(x) for x in self._linhas}
        if existente:
            self.log("experimento_retomado", n_ensaios=len(self._linhas))

    @classmethod
    def abrir(cls, diretorio: Path | str) -> RegistroExperimento:
        """Reabre um experimento existente para retomada."""
        d = Path(diretorio)
        if not (d / "config.json").is_file():
            raise FileNotFoundError(f"{d} não contém config.json")
        return cls(d.parent, id_experimento=d.name)

    # ------------------------------------------------------------------ eventos

    def log(self, evento: str, **campos: Any) -> None:
        """Acrescenta um evento a ``execution_log.jsonl``."""
        reg = _limpar_nan(sanear({"ts": _agora(), "evento": evento, **campos}))
        with (self.diretorio / "execution_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(reg, ensure_ascii=False, default=_json_padrao) + "\n")

    def chave(self, linha: Mapping[str, Any]) -> str:
        """Chave de retomada do ensaio."""
        return chave_ensaio(linha)

    def concluido(self, campos: Mapping[str, Any]) -> bool:
        """``True`` se o ensaio de ``campos`` já foi registrado."""
        return chave_ensaio(campos) in self._chaves

    # --------------------------------------------------------------- gravação

    def registrar(
        self,
        linha: Mapping[str, Any],
        candidatos: list[Mapping[str, Any]] | None = None,
        metadados_geracao: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Grava a linha do ensaio (substitui outra de mesma chave) e seus candidatos."""
        linha = dict(linha)
        linha.setdefault("modo", "M1")
        ch = chave_ensaio(linha)
        linha["chave_ensaio"] = ch
        self._linhas = [x for x in self._linhas if chave_ensaio(x) != ch] + [linha]
        self._chaves.add(ch)
        _escrever_csv(self.diretorio / "metrics.csv",
                      _ordenar_colunas(pd.DataFrame([{k: _valor_csv(v) for k, v in x.items()}
                                                     for x in self._linhas])))
        cands = [dict(c) for c in (candidatos or [])]
        self._candidatos = [c for c in self._candidatos if c.get("chave_ensaio") != ch]
        self._candidatos += [{"chave_ensaio": ch, **{k: _valor_csv(v) for k, v in c.items()}}
                             for c in cands]
        if self._candidatos:
            _escrever_csv(self.diretorio / "candidates.csv", pd.DataFrame(self._candidatos))
        self._atualizar_equacoes(linha, ch, cands)
        meta_path = self.diretorio / "generation_metadata.json"
        meta = json.loads(meta_path.read_text("utf-8"))
        ens = dict(metadados_geracao or {})
        for k in ("hash_datasets", "hash_templates", "modelo", "metadados_modelo"):
            if k in linha and k not in ens and not (isinstance(linha[k], float) and math.isnan(linha[k])):
                ens[k] = linha[k]
        meta.setdefault("ensaios", {})[ch] = ens
        _escrever_json(meta_path, meta)
        self.log("ensaio_registrado", chave=ch, status=linha.get("status"),
                 tempo_s=linha.get("tempo_s"))
        return linha

    def _atualizar_equacoes(self, linha: Mapping[str, Any], ch: str,
                            cands: list[dict[str, Any]]) -> None:
        caminho = self.diretorio / "equations.json"
        eq = json.loads(caminho.read_text("utf-8")) if caminho.is_file() else []
        eq = [e for e in eq if e.get("chave_ensaio") != ch]
        campos = ("nivel", "modo", "condicao", "algoritmo", "modelo", "ruido", "semente",
                  "expressao", "expressao_referencia", "classe", "rmse_log10",
                  "rmse_log10_validacao", "status")
        eq.append({
            "chave_ensaio": ch, **{k: linha.get(k) for k in campos},
            "candidatos": [{k: c.get(k) for k in ("indice", "linhagem", "rodada", "status",
                                                  "equacao_proposta", "expressao",
                                                  "fisicamente_valido", "rmse_log10_validacao",
                                                  "selecionado", "categoria_erro")}
                           for c in cands],
        })
        _escrever_json(caminho, eq)

    def executar(
        self,
        campos: Mapping[str, Any],
        funcao: Callable[[], Any],
        repetir_erros: bool = False,
    ) -> dict[str, Any]:
        """Executa ``funcao`` (que devolve ``linha``, ``(linha, candidatos)`` ou
        ``(linha, candidatos, metadados)``) só se o ensaio de ``campos`` não foi registrado.

        ``campos`` completa a linha (ex.: ``modo``/``modelo`` no M1) e precisa gerar a mesma
        chave. ``repetir_erros=True`` reexecuta ensaios registrados com ``status != 'ok'``.
        """
        ch = chave_ensaio(campos)
        anterior = next((x for x in self._linhas if chave_ensaio(x) == ch), None)
        if anterior is not None and not (repetir_erros and anterior.get("status") != "ok"):
            self.log("ensaio_pulado", chave=ch)
            return anterior
        res = funcao()
        linha, cands, meta = (res, None, None) if isinstance(res, Mapping) else (
            tuple(res) + (None,) * (3 - len(res)))
        linha = {**{k: v for k, v in campos.items()}, **dict(linha)}
        for k, v in campos.items():
            if linha.get(k) is None or (isinstance(linha.get(k), float) and math.isnan(linha[k])):
                linha[k] = v
        if chave_ensaio(linha) != ch:
            raise ValueError(f"a linha produzida ({chave_ensaio(linha)}) não corresponde a {ch}")
        return self.registrar(linha, cands, meta)

    def finalizar(self) -> pd.DataFrame:
        """Grava ``metrics_summary.csv`` (estatistica.agregar) e devolve o resumo."""
        df = carregar_resultados(self.diretorio)
        resumo = agregar(df)
        _escrever_csv(self.diretorio / "metrics_summary.csv", resumo)
        self.log("experimento_finalizado", n_ensaios=len(df))
        return resumo

    def salvar_figura(self, figura: Any, nome: str, **kwargs: Any) -> Path:
        """Salva uma figura matplotlib em ``figures/<nome>``."""
        caminho = self.diretorio / DIR_FIGURAS / nome
        figura.savefig(caminho, **kwargs)
        return caminho


# ------------------------------------------------------------------------------ leitura


def _mapeamento_legado() -> dict[str, Any]:
    return json.loads(MAPEAMENTO_LEGADO.read_text(encoding="utf-8"))


def normalizar_resultados(df: pd.DataFrame, metadados: Mapping[str, Any] | None = None,
                          legado: bool = False) -> pd.DataFrame:
    """Renomeia colunas/valores antigos, completa :data:`COLUNAS_PADRAO` com NaN e ordena."""
    mapa = _mapeamento_legado()
    df = df.copy()
    for antiga, nova in mapa["colunas"].items():
        if antiga in df.columns:
            legado = True
            if nova in df.columns:
                df[nova] = df[nova].where(df[nova].notna(), df[antiga])
                df = df.drop(columns=antiga)
            else:
                df = df.rename(columns={antiga: nova})
    meta = dict(metadados or {})
    for antiga, nova in mapa["metadados"].items():
        if antiga in meta:
            legado = True
            meta.setdefault(nova, meta.pop(antiga))
    if "configuracao_busca" not in df.columns and isinstance(meta.get("configuracao_busca"), str):
        df["configuracao_busca"] = meta["configuracao_busca"]
    for col, valores in mapa["valores"].items():
        if col in df.columns:
            df[col] = df[col].replace(valores)
    if "modo" not in df.columns and (legado or "algoritmo" in df.columns):
        df["modo"] = "M1"  # o notebook 01 só executa o modo M1 (executar_condicao)
    return _ordenar_colunas(df)


def carregar_resultados(caminho: Path | str) -> pd.DataFrame:
    """Lê ``results/<id>/`` (ou o ``metrics.csv``), CSV/JSON antigos do notebook 01 ou JSON
    em lista de registros, devolvendo colunas normalizadas (ausentes = NaN)."""
    p = Path(caminho)
    if p.is_dir():
        p = p / "metrics.csv"
    if p.suffix == ".csv":
        return normalizar_resultados(pd.read_csv(p))
    if p.suffix == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(d, dict) and "linhas" in d:
            meta = {k: v for k, v in d.items() if k != "linhas"}
            return normalizar_resultados(pd.DataFrame(d["linhas"]), meta, legado=True)
        if isinstance(d, list):
            return normalizar_resultados(pd.DataFrame(d))
        raise ValueError(f"JSON sem 'linhas' nem lista de registros: {p}")
    raise ValueError(f"formato não reconhecido: {p}")
