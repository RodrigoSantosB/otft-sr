"""Persistência comum a M1/M2: estrutura de results/<id>, retomada, segredos e leitura legada."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from otft_sr import persistencia
from otft_sr.dados import construir_datasets, hash_datasets
from otft_sr.experimento import COLUNAS_M1
from otft_sr.niveis import nivel
from otft_sr.persistencia import (
    ARQUIVOS,
    COLUNAS_PADRAO,
    RegistroExperimento,
    carregar_resultados,
    novo_id_experimento,
)

RAIZ = Path(__file__).resolve().parents[1]
LEGADO = json.loads((RAIZ / "data" / "legado" / "colunas_notebook01.json").read_text("utf-8"))
COL_ALG_ANTIGA = next(k for k, v in LEGADO["colunas"].items() if v == "algoritmo")
COL_CFG_ANTIGA = next(k for k, v in LEGADO["colunas"].items() if v == "configuracao_busca")
VAL_VERIF_ANTIGO = next(
    k for k, v in LEGADO["valores"]["configuracao_busca"].items() if v == "verificacao")


def _linha(**kw):
    base = {c: math.nan for c in COLUNAS_M1 if c not in ("erro", "traceback")}
    base.update(nivel="N1", nivel_proposta=1, algoritmo="llm_gerador", condicao="G1", ruido=0.0,
                semente=0, status="ok", expressao="1.0 * V_GS", rmse_log10=0.1, n_nos=3,
                one_minus_r2=0.01, classe="não converge", modo="M2", modelo="fake/x")
    base.update(kw)
    return base


# ------------------------------------------------------------------------- dataset


def test_hash_datasets_deterministico_e_sensivel():
    n1 = nivel("N1")
    a = hash_datasets(construir_datasets(n1, 0.0, 0))
    assert re.fullmatch(r"[0-9a-f]{64}", a)
    assert a == hash_datasets(construir_datasets(n1, 0.0, 0))
    assert a != hash_datasets(construir_datasets(n1, 0.0, 1))
    assert a != hash_datasets(construir_datasets(n1, 0.05, 0))


# -------------------------------------------------------------------------- estrutura


def test_id_unico_com_data_hora():
    a, b = novo_id_experimento("m2"), novo_id_experimento("m2")
    assert re.fullmatch(r"m2_\d{8}-\d{6}_[0-9a-f]{6}", a)
    assert a != b


def test_estrutura_de_diretorio_e_metadados(tmp_path):
    reg = RegistroExperimento(tmp_path, {"modo": "M2", "niveis": ["N3"]}, prefixo="m2",
                              hash_templates={"conjunto": "ab" * 32})
    d = reg.diretorio
    assert d.parent == tmp_path and d.name.startswith("m2_")
    for nome in ("config.json", "environment.json", "generation_metadata.json",
                 "execution_log.jsonl"):
        assert (d / nome).is_file(), nome
    assert (d / "figures").is_dir()
    cfg = json.loads((d / "config.json").read_text("utf-8"))
    assert cfg["config"]["niveis"] == ["N3"] and cfg["id_experimento"] == d.name
    amb = json.loads((d / "environment.json").read_text("utf-8"))
    assert "python" in amb and "git" in amb
    assert re.fullmatch(r"[0-9a-f]{40}", amb["git"]["commit"])  # o projeto é um repositório git
    meta = json.loads((d / "generation_metadata.json").read_text("utf-8"))
    assert meta["hash_templates"]["conjunto"] == "ab" * 32
    assert set(ARQUIVOS) >= {"metrics.csv", "metrics_summary.csv", "candidates.csv",
                             "equations.json"}


def test_registrar_grava_todos_os_artefatos(tmp_path):
    reg = RegistroExperimento(tmp_path, {})
    cands = [{"indice": 0, "status": "valido", "expressao": "1.0 * V_GS", "selecionado": True,
              "checagens": [{"nome": "x", "aprovado": True}]},
             {"indice": 1, "status": "invalido", "categoria_erro": "inseguro"}]
    reg.registrar(_linha(), cands, {"hash_datasets": "cd" * 32, "prompts": ["p"]})
    reg.registrar(_linha(semente=1, rmse_log10=0.3))
    reg.finalizar()
    d = reg.diretorio
    m = pd.read_csv(d / "metrics.csv")
    assert len(m) == 2 and list(m.columns[: len(COLUNAS_PADRAO)]) == list(COLUNAS_PADRAO)
    c = pd.read_csv(d / "candidates.csv")
    assert len(c) == 2 and {"chave_ensaio", "indice", "status"} <= set(c.columns)
    eq = json.loads((d / "equations.json").read_text("utf-8"))
    assert len(eq) == 2 and eq[0]["expressao"] == "1.0 * V_GS"
    meta = json.loads((d / "generation_metadata.json").read_text("utf-8"))
    assert meta["ensaios"][reg.chave(_linha())]["hash_datasets"] == "cd" * 32
    log = [json.loads(x) for x in (d / "execution_log.jsonl").read_text("utf-8").splitlines()]
    assert [e["evento"] for e in log].count("ensaio_registrado") == 2
    s = pd.read_csv(d / "metrics_summary.csv")
    linha_rmse = s[(s["metrica"] == "rmse_log10")].iloc[0]
    assert linha_rmse["n"] == 2 and linha_rmse["media"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- retomada


def test_checkpoint_pula_ensaios_concluidos(tmp_path):
    chamadas = []

    def ensaio(semente):
        def f():
            chamadas.append(semente)
            return _linha(semente=semente), [{"indice": 0, "status": "valido"}]
        return f

    reg = RegistroExperimento(tmp_path, {})
    campos = dict(nivel="N1", modo="M2", condicao="G1", algoritmo="llm_gerador",
                  modelo="fake/x", ruido=0.0)
    reg.executar({**campos, "semente": 0}, ensaio(0))
    assert reg.concluido({**campos, "semente": 0})
    assert not reg.concluido({**campos, "semente": 1})
    reaberto = RegistroExperimento.abrir(reg.diretorio)
    assert reaberto.diretorio == reg.diretorio
    for s in (0, 1):
        reaberto.executar({**campos, "semente": s}, ensaio(s))
    assert chamadas == [0, 1]
    assert len(pd.read_csv(reg.diretorio / "metrics.csv")) == 2
    log = (reg.diretorio / "execution_log.jsonl").read_text("utf-8")
    assert "ensaio_pulado" in log


def test_chave_distingue_modelo_e_condicao():
    reg_chave = persistencia.chave_ensaio
    a = reg_chave(_linha())
    assert a != reg_chave(_linha(modelo="ollama/y"))
    assert a != reg_chave(_linha(condicao="G2"))
    assert a != reg_chave(_linha(semente=3))
    assert a == reg_chave(_linha(rmse_log10=9.0))


# ---------------------------------------------------------------------------- segredos

PADROES_CHAVE = re.compile(
    r"sk-[A-Za-z0-9_\-]{8,}|sk-ant-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{10,}|hf_[A-Za-z0-9]{10,}"
)


def test_nunca_grava_segredos(tmp_path, monkeypatch):
    chave_env = "sk-testeAmbiente1234567890abcdef"
    monkeypatch.setenv("OPENAI_API_KEY", chave_env)
    monkeypatch.setenv("HF_TOKEN", "hf_abcdefghijklmnopqrstu")
    config = {
        "provedor": {"provedor": "openai", "api_key": "sk-configuracao0123456789",
                     "variavel_credencial": "OPENAI_API_KEY"},
        "headers": {"Authorization": "Bearer AIzaSyQualquerCoisa123456"},
        "senha": "segredo-simples",
    }
    reg = RegistroExperimento(tmp_path, config)
    cands = [{"indice": 0, "resposta_bruta": "use a chave AIzaSyVazada12345678 e hf_vazado1234567890"}]
    reg.registrar(_linha(), cands, {"token": "sk-ant-vazamento123456789"})
    reg.finalizar()
    arquivos = [p for p in reg.diretorio.rglob("*") if p.is_file()]
    assert len(arquivos) >= 7
    for p in arquivos:
        texto = p.read_text("utf-8")
        assert not PADROES_CHAVE.search(texto), p.name
        for literal in (chave_env, "segredo-simples", "sk-configuracao0123456789"):
            assert literal not in texto, p.name
    cfg = json.loads((reg.diretorio / "config.json").read_text("utf-8"))
    assert cfg["config"]["provedor"]["variavel_credencial"] == "OPENAI_API_KEY"  # nome, não valor
    meta = json.loads((reg.diretorio / "generation_metadata.json").read_text("utf-8"))
    assert re.fullmatch(r"[0-9a-f]{64}", meta.get("hash_templates", {}).get("conjunto", "0" * 64))


def test_sanear_preserva_hashes_e_contadores():
    obj = {"hash_datasets": "ab" * 32, "tokens_entrada": 120, "chave_ensaio": "N1|M2",
           "api_key": "qualquer", "texto": "x sk-abcdefghijk y"}
    s = persistencia.sanear(obj)
    assert s["hash_datasets"] == "ab" * 32 and s["tokens_entrada"] == 120
    assert s["chave_ensaio"] == "N1|M2"
    assert s["api_key"] == "***" and "sk-abcdefghijk" not in s["texto"]


# --------------------------------------------------------------------- leitura legada


def test_carregar_csv_legado_normaliza_sem_inventar(tmp_path):
    antigo = pd.DataFrame({
        "nivel": ["N1", "N2"], COL_ALG_ANTIGA: ["pysr", "deap"], "condicao": ["A", "A"],
        "ruido": [0.0, 0.0], "semente": [0, 1], COL_CFG_ANTIGA: [VAL_VERIF_ANTIGO] * 2,
        "rmse_log10": [0.1, 0.2], "status": ["ok", "ok"], "coluna_extra": [1, 2],
    })
    caminho = tmp_path / "rs_otft_antigo.csv"
    antigo.to_csv(caminho, index=False)
    df = carregar_resultados(caminho)
    assert list(df["algoritmo"]) == ["pysr", "deap"]
    assert list(df["configuracao_busca"]) == ["verificacao"] * 2
    assert COL_ALG_ANTIGA not in df and COL_CFG_ANTIGA not in df
    assert df["nivel_proposta"].isna().all() and df["one_minus_r2"].isna().all()
    assert df["modelo"].isna().all()
    assert list(df["modo"]) == ["M1", "M1"]  # o notebook 01 só executa o modo M1
    assert list(df["coluna_extra"]) == [1, 2]
    assert set(COLUNAS_PADRAO) <= set(df.columns)


def test_carregar_json_legado_do_notebook(tmp_path):
    meta = {COL_CFG_ANTIGA: VAL_VERIF_ANTIGO, "backend_llm": "nenhum",
            "linhas": [{"nivel": "N1", COL_ALG_ANTIGA: "gplearn", "condicao": "A",
                        "semente": 0, "rmse_log10": 0.5}]}
    caminho = tmp_path / "antigo.json"
    caminho.write_text(json.dumps(meta), "utf-8")
    df = carregar_resultados(caminho)
    assert df.loc[0, "algoritmo"] == "gplearn"
    assert df.loc[0, "configuracao_busca"] == "verificacao"
    assert df.loc[0, "rmse_log10"] == 0.5 and np.isnan(df.loc[0, "n_nos"])


def test_carregar_resultados_reimporta_o_proprio_formato(tmp_path):
    reg = RegistroExperimento(tmp_path, {})
    reg.registrar(_linha(), [{"indice": 0}])
    reg.registrar(_linha(semente=1, modelo="ollama/mistral:7b-instruct"))
    for caminho in (reg.diretorio, reg.diretorio / "metrics.csv"):
        df = carregar_resultados(caminho)
        assert len(df) == 2
        assert list(df["modelo"]) == ["fake/x", "ollama/mistral:7b-instruct"]
        assert list(df["semente"]) == [0, 1]
