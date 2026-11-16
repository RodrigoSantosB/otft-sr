"""Configuração única, plano de ensaios, retomada e linha de base do notebook 02."""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import pytest

from otft_sr import campanha as cp
from otft_sr.campanha import ConfiguracaoExperimento
from otft_sr.niveis import nivel
from otft_sr.persistencia import RegistroExperimento, chave_ensaio


def _cfg(**kw):
    return ConfiguracaoExperimento("teste", **kw)


def test_padrao_valido_com_tres_modelos_de_revisao_fixada():
    cfg = _cfg()
    assert [m.modelo for m in cfg.modelos] == [
        "mistralai/Mistral-7B-Instruct-v0.1", "Qwen/Qwen2.5-3B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct"]
    assert all(len(m.revisao) == 40 for m in cfg.modelos)
    assert cfg.formulacao_perda == "linear" and not cfg.execucao_de_verificacao
    json.dumps(cfg.como_dict())


@pytest.mark.parametrize("kw, trecho", [
    ({"niveis": ["N5"]}, "niveis"),
    ({"niveis_llm": ["N1"], "niveis": ["N3"]}, "niveis_llm"),
    ({"condicoes_m1": ["D"]}, "condicoes_m1"),
    ({"condicoes_m2": ["G3"]}, "condicoes_m2"),
    ({"algoritmos_rs": ["xgb"]}, "algoritmos_rs"),
    ({"formulacao_perda": "ln"}, "formulacao_perda"),
    ({"sementes": [0, 0]}, "sementes"),
    ({"ruidos": [1.5]}, "ruidos"),
    ({"configuracao_busca": "rapida"}, "configuracao_busca"),
    ({"modelos": []}, "nenhum modelo"),
    ({"n_parametros_b": {}}, "n_parametros_b"),
    ({"n_curvas": 3}, "n_curvas"),
])
def test_validacao_recusa_com_mensagem(kw, trecho):
    with pytest.raises((ValueError, TypeError), match=trecho):
        _cfg(**kw)


def test_nome_invalido():
    with pytest.raises(ValueError):
        ConfiguracaoExperimento("com espaço")


def test_verificacao_minima_e_backends():
    v = cp.configuracao_de_verificacao(_cfg(), "simulado")
    assert v.execucao_de_verificacao and v.id_experimento == "teste_verificacao"
    assert v.niveis == ("N3",) and v.sementes == (0,) and v.algoritmos_rs == ("deap",)
    assert v.n_candidatos == 2 and v.n_iteracoes_refinamento == 1
    assert [m.provedor for m in v.modelos] == ["fake"]
    o = cp.configuracao_de_verificacao(_cfg(), "ollama")
    assert o.modelos[0].provedor == "ollama" and o.modelos[0].modelo == "mistral:7b-instruct"
    with pytest.raises(ValueError):
        cp.configuracao_de_verificacao(_cfg(), "gpu")


def test_quantizacao_automatica():
    cfg, tab = cp.aplicar_quantizacao(_cfg(), 15.0)  # T4
    q = {m.modelo: m.quantizacao for m in cfg.modelos}
    assert q["mistralai/Mistral-7B-Instruct-v0.1"] == "8bit"
    assert q["Qwen/Qwen2.5-3B-Instruct"] is None
    assert len(tab) == 3
    cfg80, _ = cp.aplicar_quantizacao(_cfg(), 80.0)
    assert all(m.quantizacao is None for m in cfg80.modelos)
    with pytest.raises(ValueError, match="sem GPU"):
        cp.aplicar_quantizacao(_cfg(), 0.0)


def test_plano_de_ensaios_tamanho_e_chaves_unicas():
    cfg = _cfg(niveis=["N1", "N3"], niveis_llm=["N3"], sementes=[0, 1], ruidos=[0.0, 0.05])
    plano = cp.plano_de_ensaios(cfg)
    n_a = 2 * 2 * 2 * 2          # níveis x algoritmos x ruídos x sementes
    n_m2 = 3 * 1 * 3 * 2 * 2     # modelos x níveis_llm x G x ruídos x sementes
    n_bc = 3 * 2 * 1 * 1 * 2 * 2  # modelos x B/C x níveis_llm x deap x ruídos x sementes
    assert len(plano) == n_a + n_m2 + n_bc
    chaves = plano.drop(columns="etapa").apply(lambda r: chave_ensaio(r.to_dict()), axis=1)
    assert chaves.is_unique


def test_plano_log10_muda_chaves_m1():
    p_lin = cp.plano_de_ensaios(_cfg(modos=["M1"], condicoes_m1=["A"]))
    p_log = cp.plano_de_ensaios(_cfg(modos=["M1"], condicoes_m1=["A"], formulacao_perda="log10"))
    k = [set(p.drop(columns="etapa").apply(lambda r: chave_ensaio(r.to_dict()), axis=1))
         for p in (p_lin, p_log)]
    assert not (k[0] & k[1])


def test_abrir_registro_retoma_e_recusa_configuracao_incompativel(tmp_path):
    cfg = _cfg()
    reg = cp.abrir_registro(tmp_path, cfg)
    assert reg.diretorio.name == "teste"
    cp.abrir_registro(tmp_path, cfg)  # mesma configuração: retoma
    with pytest.raises(ValueError, match="formulacao_perda"):
        cp.abrir_registro(tmp_path, _cfg(formulacao_perda="log10"))


def test_sincronizar(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "x.csv").write_text("1")
    assert cp.sincronizar(tmp_path / "a", None) is None
    cp.sincronizar(tmp_path / "a", tmp_path / "drive")
    assert (tmp_path / "drive" / "b" / "x.csv").read_text() == "1"


def test_importar_linha_de_base_filtra_referencia_atual(tmp_path):
    atual = nivel("N1").expressao
    comum = {"nivel": "N1", "algoritmo": "deap", "ruido": 0.0, "status": "ok",
             "classe": "não recupera", "configuracao_busca": "calibrada", "n_curvas": 20,
             "n_pontos": 50, "formulacao_perda": "linear"}
    pd.DataFrame([
        {**comum, "condicao": "A", "semente": 0, "expressao_referencia": atual},
        {**comum, "condicao": "A", "semente": 1, "expressao_referencia": "1.0 * V_GS"},
        {**comum, "condicao": "B", "semente": 0, "expressao_referencia": atual},
    ]).to_csv(tmp_path / "rs_otft_x.csv", index=False)
    df, rel = cp.importar_linha_de_base([str(tmp_path / "*.csv"), str(tmp_path / "nada*.json")],
                                        _cfg())
    assert len(df) == 1 and df["semente"].iloc[0] == 0 and df["modo"].iloc[0] == "M1"
    assert rel.loc[0, "linhas"] == 3 and rel.loc[0, "aceitas"] == 1


def test_importar_linha_de_base_filtra_busca_dados_e_formulacao(tmp_path):
    atual = nivel("N1").expressao
    ok = {"nivel": "N1", "algoritmo": "deap", "condicao": "A", "ruido": 0.0, "status": "ok",
          "expressao_referencia": atual, "configuracao_busca": "calibrada", "n_curvas": 20,
          "n_pontos": 50, "formulacao_perda": "linear"}
    linhas = [
        {**ok, "semente": 0},                                      # aceita
        {**ok, "semente": 1, "configuracao_busca": "verificacao"},  # outra busca
        {**ok, "semente": 2, "n_curvas": 10},                       # outros dados
        {**ok, "semente": 3, "n_pontos": 30},
        {**ok, "semente": 4, "formulacao_perda": "log10"},          # outra formulação
        # antigas sem n_curvas/n_pontos: conferidas por n_treino/n_teste (12 e 4 curvas x 50)
        {**ok, "semente": 5, "n_curvas": None, "n_pontos": None, "n_treino": 600, "n_teste": 200},
        {**ok, "semente": 6, "n_curvas": None, "n_pontos": None, "n_treino": 300, "n_teste": 100},
        {**ok, "semente": 7, "n_curvas": None, "n_pontos": None},  # sem como conferir
    ]
    pd.DataFrame(linhas).to_csv(tmp_path / "x.csv", index=False)
    df, rel = cp.importar_linha_de_base([str(tmp_path / "x.csv")], _cfg())
    assert sorted(df["semente"]) == [0, 5]
    assert rel.loc[0, "aceitas"] == 2
    log10 = cp.importar_linha_de_base([str(tmp_path / "x.csv")], _cfg(formulacao_perda="log10"))[0]
    assert sorted(log10["semente"]) == [4]
    verif = cp.importar_linha_de_base([str(tmp_path / "x.csv")], _cfg(), rotulo_busca="verificacao")[0]
    assert sorted(verif["semente"]) == [1]


def test_tipo_da_comparacao_A_com_LLM():
    assert cp.tipo_comparacao_a_llm(_cfg()) == "equivalente"
    assert cp.tipo_comparacao_a_llm(_cfg(configuracao_busca_llm="calibrada")) == "equivalente"
    assert cp.tipo_comparacao_a_llm(_cfg(configuracao_busca_llm="verificacao")) == "exploratória"
    assert cp.tipo_comparacao_a_llm(_cfg(configuracao_busca_llm={"sr.population": 8})) == "exploratória"


def test_conclusoes_em_verificacao_declaram_ausencia_de_resultado():
    v = cp.configuracao_de_verificacao(_cfg())
    texto = cp.texto_conclusoes(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), v)
    assert "não constitui resultado" in texto


def test_conclusoes_a_partir_das_tabelas():
    taxa = pd.DataFrame([{"nivel": "N1", "modo": "M1", "condicao": "A", "modelo": None, "n": 3,
                          "n_recuperados": 2, "taxa": 2 / 3, "ic95_inf": 0.2, "ic95_sup": 0.94}])
    aud = pd.DataFrame([{"modelo": "m", "condicao": "C", "nivel": "N1", "c_referencia": 0.1,
                         "c_errada_min": 0.1, "discrimina_erradas": False}])
    comp = pd.DataFrame([{"comparacao": "A vs C", "nivel": "N1", "n_pares": 3,
                          "diferenca_media": 0.01, "p_valor": 0.5, "p_mcnemar": 1.0,
                          "tipo": "equivalente"}])
    texto = cp.texto_conclusoes(taxa, aud, comp, _cfg())
    assert "2/3" in texto and "NÃO discrimina" in texto and "A vs C" in texto


# --------------------------------------------------- identidade do modelo na retomada


def _hf(quant):
    return dataclasses.replace(cp.QWEN25_3B, quantizacao=quant)


def test_chave_do_ensaio_distingue_quantizacao(tmp_path):
    """Retomar com outra quantização não reaproveita ensaios gravados com a anterior."""
    kw = dict(modos=("M1", "M2"), niveis=("N3",), sementes=(0,), n_parametros_b=cp.N_PARAMETROS_B)
    p4 = cp.plano_de_ensaios(_cfg(modelos=(_hf("4bit"),), **kw))
    p8 = cp.plano_de_ensaios(_cfg(modelos=(_hf("8bit"),), **kw))
    llm4 = p4[p4["etapa"] != "M1 linha de base"].drop(columns="etapa")
    llm8 = p8[p8["etapa"] != "M1 linha de base"].drop(columns="etapa")
    k4 = {chave_ensaio(r) for r in llm4.to_dict("records")}
    k8 = {chave_ensaio(r) for r in llm8.to_dict("records")}
    assert k4 and not (k4 & k8)
    reg = RegistroExperimento(tmp_path, id_experimento="x")
    for r in llm4.to_dict("records"):
        reg.registrar({**r, "status": "ok"})
    assert all(reg.concluido(r) for r in llm4.to_dict("records"))
    assert not any(reg.concluido(r) for r in llm8.to_dict("records"))


def test_retomada_com_quantizacao_diferente_acusa_incompatibilidade(tmp_path):
    cfg = _cfg(modelos=(_hf("4bit"),))
    cp.abrir_registro(tmp_path, cfg)
    cp.abrir_registro(tmp_path, cfg)  # mesma identidade: retoma
    with pytest.raises(ValueError, match="modelos"):
        cp.abrir_registro(tmp_path, _cfg(modelos=(_hf("8bit"),)))
    with pytest.raises(ValueError, match="modelos"):
        cp.abrir_registro(tmp_path, _cfg(modelos=(dataclasses.replace(_hf("4bit"), revisao="b" * 40),)))


def test_conclusoes_reportam_excluidos_por_erro():
    taxa = pd.DataFrame([{"nivel": "N1", "modo": "M1", "condicao": "A", "modelo": None, "n": 2,
                          "n_excluido_erro": 1, "n_recuperados": 1, "taxa": 0.5,
                          "ic95_inf": 0.1, "ic95_sup": 0.9}])
    comp = pd.DataFrame([{"comparacao": "A vs C", "nivel": "N1", "n_pares": 2,
                          "n_excluido_erro": 1, "diferenca_media": 0.01, "p_valor": 0.5,
                          "p_mcnemar": 1.0, "tipo": "equivalente"}])
    texto = cp.texto_conclusoes(taxa, pd.DataFrame(), comp, _cfg())
    assert texto.count("1 excluído(s) por erro de execução") == 2
