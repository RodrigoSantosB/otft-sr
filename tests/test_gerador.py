"""Modo M2: LLM como gerador de equações (G0, G1, G2) com FakeBackend."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from physics_llm_sr.llm.inference import FakeBackend
from physics_llm_sr.symbolic.expression import parse_infix

from otft_sr import gerador
from otft_sr.dados import construir_datasets
from otft_sr.experimento import COLUNAS_M1
from otft_sr.gerador import (
    RespostaInvalida,
    ajustar_parametros,
    checagens_fisicas,
    executar_geracao,
    extrair_primeiro_json,
    interpretar_equacao,
    montar_prompt,
    selecionar_melhor,
    validar_resposta,
)
from otft_sr.niveis import FEATURES, nivel
from otft_sr.provedores_llm import ConfiguracaoLLM

N1, N2, N3, N4 = (nivel(i) for i in ("N1", "N2", "N3", "N4"))


def _json(equation, parameters=None, variables=("V_GS",), **extra):
    d = {
        "equation": equation,
        "variables": list(variables),
        "parameters": parameters if parameters is not None else {},
        "assumptions": ["saturation"],
        "physical_constraints": ["I_D > 0"],
        **extra,
    }
    return "Here is my proposal:\n```json\n" + json.dumps(d) + "\n```"


CORRETA_N3 = _json("K * (V_GS - V0) ** l", {"K": 1e-4, "V0": 8.0, "l": 3.0})
QUADRATICA = _json("K * (V_GS - V0) ** 2", {"K": 1e-3, "V0": 10.0})
ABSURDA = _json("a * sin(b * V_DS) + c", {"a": 1.0, "b": 1.0, "c": 0.0}, variables=("V_DS",))
MALICIOSA = _json("__import__('os').system('echo invadido')")


@pytest.fixture(scope="module")
def ds_n3():
    return construir_datasets(N3, 0.0, 0)


# ----------------------------------------------------------------------- templates


def test_templates_modulares_com_hash():
    t = gerador.carregar_templates()
    assert set(gerador.SECOES_PROMPT) <= set(t)
    h = gerador.hash_templates(t)
    assert set(t) <= set(h["arquivos"])
    assert re.fullmatch(r"[0-9a-f]{64}", h["conjunto"])
    t2 = dict(t, instrucoes=t["instrucoes"] + " ")
    assert gerador.hash_templates(t2)["conjunto"] != h["conjunto"]
    assert gerador.hash_templates(dict(t))["conjunto"] == h["conjunto"]


def test_prompt_G0_so_variaveis_unidades_e_resumo(ds_n3):
    sistema, usuario = montar_prompt(N3, "G0", ds_n3["treino"], 1, 3)
    assert "V_GS [V]" in usuario and "I_D [uA]" in usuario
    assert "Training data summary" in usuario
    for termo in ("OTFT", "threshold", "mobility", "saturation", "C_i", "W/L"):
        assert termo not in usuario, termo


@pytest.mark.parametrize("nv", [N1, N2, N3, N4], ids=lambda n: n.id)
@pytest.mark.parametrize("condicao", ["G0", "G1", "G2"])
def test_prompt_nao_revela_equacao_nem_constantes(nv, condicao):
    ds = construir_datasets(nv, 0.0, 0)
    sistema, usuario = montar_prompt(nv, condicao, ds["treino"], 1, 3)
    texto = sistema + usuario
    assert nv.expressao not in texto
    for forma in nv.formas_equivalentes:
        assert forma not in texto
    for nome, valor in nv.constantes.items():
        if nome.startswith("S_teorico"):
            continue
        assert not re.search(rf"(?<![\d.]){re.escape(repr(valor))}(?![\d])", texto), nome


def test_prompt_G1_tem_contexto_fisico_estruturado(ds_n3):
    _, usuario = montar_prompt(N3, "G1", ds_n3["treino"], 1, 3)
    for termo in ("OTFT", "V_Th", "C_i", "W/L", "mobility", "saturation", "triode", "V", "uA"):
        assert termo in usuario, termo


def test_condicao_desconhecida(ds_n3):
    with pytest.raises(ValueError):
        montar_prompt(N3, "C", ds_n3["treino"], 1, 1)


# ------------------------------------------------------------------ resposta e schema


def test_extrai_primeiro_objeto_json():
    texto = 'bla {"equation": "a * V_GS", "parameters": {"a": {"x": 1}}} depois {"equation": "b"}'
    assert extrair_primeiro_json(texto)["equation"] == "a * V_GS"
    assert extrair_primeiro_json('{quebrado} {"equation": "c"}')["equation"] == "c"
    with pytest.raises(RespostaInvalida) as e:
        extrair_primeiro_json("sem json aqui")
    assert e.value.categoria == "json_ausente"


def test_schema_valida_e_normaliza():
    r = validar_resposta(_json("K * V_GS", ["K"]))
    assert r.parameters == {"K": None}
    r = validar_resposta(_json("K * V_GS", {"K": "transconductance"}))
    assert r.parameters == {"K": None}
    r = validar_resposta(_json("K * V_GS", [{"name": "K", "initial_value": 2.0}]))
    assert r.parameters == {"K": 2.0}
    for ruim in (
        json.dumps({"variables": ["V_GS"], "parameters": {}}),  # sem equation
        _json("K * T", {"K": 1}, variables=("T",)),  # variável não permitida
        _json("_x * V_GS", {"_x": 1}),  # nome de parâmetro inválido
        _json("V_GS * a", {"V_GS": 1}),  # parâmetro com nome de variável
        _json("a*V_GS", {f"p{i}": 1 for i in range(gerador.MAX_PARAMETROS + 1)}),
    ):
        with pytest.raises(RespostaInvalida) as e:
            validar_resposta(ruim)
        assert e.value.categoria == "schema", ruim


@pytest.mark.parametrize(
    "texto",
    [
        "__import__('os').system('echo invadido')",
        "os.system('ls')",
        "(lambda: 1)()",
        "V_GS.__class__",
        "[x for x in (1, 2)]",
        "open('arquivo')",
        "eval('1')",
        "a if V_GS > 1 else b",
        "V_GS > a",
        "'texto' * 2",
    ],
)
def test_rejeita_codigo_malicioso_sem_executar(texto):
    with pytest.raises(RespostaInvalida) as e:
        interpretar_equacao(texto, {})
    assert e.value.categoria in {"inseguro", "sintaxe"}


def test_codigo_fonte_sem_eval_exec_sympify():
    fonte = Path(gerador.__file__).read_text(encoding="utf-8")
    for proibido in (r"(?<![\w.])eval\(", r"(?<![\w.])exec\(", r"sympify\(",
                     r"(?<![\w.])compile\(", r"__import__\("):
        assert not re.search(proibido, fonte), proibido


def test_limites_de_complexidade_e_operacoes():
    profunda = "a * V_GS"
    for _ in range(gerador.MAX_PROFUNDIDADE):
        profunda = f"exp({profunda})"
    with pytest.raises(RespostaInvalida) as e:
        interpretar_equacao(profunda, {"a": None})
    assert e.value.categoria == "complexidade"
    longa = " + ".join(f"V_GS * {i}" for i in range(12))
    with pytest.raises(RespostaInvalida) as e:
        interpretar_equacao(longa, {})
    assert e.value.categoria == "complexidade"
    with pytest.raises(RespostaInvalida) as e:
        interpretar_equacao("sqrt(V_GS)", {})
    assert e.value.categoria == "operacao_nao_permitida"
    with pytest.raises(RespostaInvalida) as e:
        interpretar_equacao("a + b", {"a": None, "b": None})
    assert e.value.categoria == "sem_variavel"


def test_normalizacao_textual_e_parametros_inferidos():
    i = interpretar_equacao("I_D = K*(V_GS - Vt)^2 + ln(c) * 0", {"K": 1.0})
    assert i.parametros == ("K", "Vt", "c")
    assert i.parametros_inferidos == ("Vt", "c")
    assert "**" in i.texto_normalizado and "log(" in i.texto_normalizado
    assert i.chutes == {"K": 1.0, "Vt": None, "c": None}


# -------------------------------------------------------------------------- ajuste


def test_ajuste_recupera_constantes_N3_com_precisao(ds_n3):
    interp = interpretar_equacao("K * (V_GS - V0) ** l", {"K": 1e-4, "V0": 8.0, "l": 3.0})
    tr = ds_n3["treino"]
    aj = ajustar_parametros(interp, tr.X, tr.y_observed, semente=0)
    assert aj.convergiu
    c = N3.constantes
    assert aj.valores["K"] == pytest.approx(c["K"], rel=1e-4)
    assert aj.valores["V0"] == pytest.approx(c["V_th0"], rel=1e-5)
    assert aj.valores["l"] == pytest.approx(c["l"], rel=1e-5)
    assert aj.n_inicios == gerador.N_INICIOS_PADRAO


def test_ajuste_sem_chute_recupera_N1_e_e_deterministico():
    ds = construir_datasets(N1, 0.0, 0)
    interp = interpretar_equacao("K * (V_GS - V_T) ** 2", {})
    a = ajustar_parametros(interp, ds["treino"].X, ds["treino"].y_observed, semente=3)
    b = ajustar_parametros(interp, ds["treino"].X, ds["treino"].y_observed, semente=3)
    assert a.valores == b.valores
    assert a.valores["K"] == pytest.approx(N1.constantes["K"], rel=1e-4)
    assert a.valores["V_T"] == pytest.approx(N1.constantes["V_Th"], rel=1e-5)


# ----------------------------------------------------------------------- checagens


@pytest.mark.parametrize("nv", [N1, N2, N3, N4], ids=lambda n: n.id)
def test_referencia_passa_todas_as_checagens_obrigatorias(nv):
    ds = construir_datasets(nv, 0.05, 0)
    ch = checagens_fisicas(parse_infix(nv.expressao, FEATURES), nv, ds["treino"])
    nomes = {c.nome for c in ch}
    assert {"positiva_finita", "monotonica_V_GS", "dimensional"} <= nomes
    assert any(c.nome.startswith("saturacao") for c in ch)
    reprovadas = [c for c in ch if c.obrigatoria and c.aprovado is False]
    assert reprovadas == []
    dim = next(c for c in ch if c.nome == "dimensional")
    assert dim.obrigatoria is False and "qualitativ" in dim.nota


def test_checagens_reprovam_formas_nao_fisicas():
    ds = construir_datasets(N1, 0.0, 0)

    def reprovadas(texto, nv=N1, d=ds):
        ch = checagens_fisicas(parse_infix(texto, FEATURES), nv, d["treino"])
        return {c.nome for c in ch if c.obrigatoria and c.aprovado is False}

    assert {"positiva_finita", "monotonica_V_GS"} <= reprovadas("-0.01 * V_GS")
    assert "saturacao_V_DS" in reprovadas("0.008 * (V_GS - 20.0) ** 2 * V_DS / 45")
    assert "anulacao_no_limiar" in reprovadas("0.05 * exp(0.06 * V_GS)")
    d4 = construir_datasets(N4, 0.0, 0)
    assert "patamar_de_fuga" in reprovadas("0.0095 * exp(0.9882 * V_GS)", N4, d4)
    d2 = construir_datasets(N2, 0.0, 0)
    assert "anulacao_em_V_DS_nulo" in reprovadas("0.01 * (V_GS - 20.0) * (V_DS + 5.0)", N2, d2)
    assert "saturacao_na_constricao" in reprovadas("0.01 * (V_GS - 25.0) * V_DS", N2, d2)


def test_dimensional_qualitativa_marca_exp_de_tensao_sem_escala():
    ds = construir_datasets(N4, 0.0, 0)
    ch = checagens_fisicas(parse_infix("65.0 + exp(V_GS)", FEATURES), N4, ds["treino"])
    dim = next(c for c in ch if c.nome == "dimensional")
    assert dim.aprovado is False and dim.obrigatoria is False


# ------------------------------------------------------------------------- seleção


def test_selecao_por_validacao_entre_fisicamente_validos():
    c = [
        {"status": "valido", "fisicamente_valido": False, "rmse_log10_validacao": 0.001, "n_nos": 3},
        {"status": "valido", "fisicamente_valido": True, "rmse_log10_validacao": 0.20, "n_nos": 9},
        {"status": "valido", "fisicamente_valido": True, "rmse_log10_validacao": 0.10, "n_nos": 9},
        {"status": "valido", "fisicamente_valido": True, "rmse_log10_validacao": 0.10, "n_nos": 7},
        {"status": "invalido", "fisicamente_valido": False, "rmse_log10_validacao": math.nan},
    ]
    assert selecionar_melhor(c) == 3
    assert selecionar_melhor([c[0], c[4]]) == 0  # nenhum válido: melhor ajustado, marcado
    assert selecionar_melhor([c[4]]) is None


# ------------------------------------------------------------------- ponta a ponta


def test_executar_geracao_G1_com_respostas_invalidas_maliciosa_absurda_e_correta(ds_n3):
    roteiro = [
        "Desculpe, não sei.",  # sem JSON -> retentativa
        CORRETA_N3,
        MALICIOSA,  # inseguro -> retentativa
        MALICIOSA,  # esgota retentativas: candidato inválido
        ABSURDA,  # interpreta, mas reprova checagens
        QUADRATICA,
    ]
    fb = FakeBackend(roteiro)
    linha, cands = executar_geracao(
        N3, "G1", fb, semente=0, n_candidatos=4, n_iteracoes_refinamento=0,
        max_retentativas=1, datasets=ds_n3,
    )
    assert fb.prompt_count == 6
    assert set(COLUNAS_M1) - {"erro", "traceback"} <= set(linha)
    assert linha["modo"] == "M2" and linha["condicao"] == "G1"
    assert linha["algoritmo"] == gerador.ALGORITMO_M2
    assert linha["n_candidatos"] == 4 and linha["n_candidatos_avaliados"] == 4
    assert linha["n_candidatos_validos"] == 3
    assert linha["taxa_validos"] == pytest.approx(0.75)
    assert linha["n_candidatos_fisicamente_validos"] == 2
    assert linha["status"] == "ok"
    assert linha["forma_funcional_sympy"] is True
    assert linha["rmse_log10"] < 1e-6 and linha["rmse_log10_validacao"] < 1e-6
    assert linha["classe"].startswith("recupera estrutura")
    assert linha["chamadas_llm"] == 6
    for col in ("tokens_entrada", "tokens_saida", "tempo_inferencia_s", "custo_api_estimado",
                "hash_templates", "modelo", "hash_datasets"):
        assert col in linha
    assert len(cands) == 4
    sel = [c for c in cands if c["selecionado"]]
    assert len(sel) == 1 and sel[0]["indice"] == 0
    assert cands[0]["n_tentativas"] == 2 and cands[0]["tentativas"][0]["categoria"] == "json_ausente"
    assert cands[1]["status"] == "invalido"
    assert [t["categoria"] for t in cands[1]["tentativas"]] == ["inseguro", "inseguro"]
    assert cands[2]["status"] == "valido" and cands[2]["fisicamente_valido"] is False
    assert "invadido" not in json.dumps(linha, default=str)
    assert all("rmse_log10_teste" not in c for c in cands)


def test_retentativa_envia_motivo_da_rejeicao(ds_n3):
    fb = FakeBackend(["nada", CORRETA_N3])
    executar_geracao(N3, "G0", fb, 0, 1, 0, max_retentativas=1, datasets=ds_n3)
    assert "rejected" in fb.calls[1][1] and "json_ausente" in fb.calls[1][1]


def test_G2_refinamento_com_feedback_de_treino_e_validacao(ds_n3):
    def responder(sistema, usuario):
        return CORRETA_N3 if "Feedback on your previous proposal" in usuario else QUADRATICA

    fb = FakeBackend(responder)
    linha, cands = executar_geracao(N3, "G2", fb, 0, n_candidatos=2, n_iteracoes_refinamento=2,
                                    datasets=ds_n3)
    assert len(cands) == 6 and linha["n_candidatos_avaliados"] == 6
    assert [(c["linhagem"], c["rodada"]) for c in cands] == [
        (0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
    feedbacks = [u for _, u in fb.calls if "Feedback on your previous proposal" in u]
    assert len(feedbacks) == 4
    secao = feedbacks[0].split("Feedback on your previous proposal")[1].split("Propose candidate")[0]
    assert "validation" in secao and "training" in secao
    assert "test" not in secao.lower()
    assert cands[2]["prompt_feedback_de"] == 0
    assert linha["selecionado_rodada"] >= 1
    assert linha["forma_funcional_sympy"] is True


def test_G0_G1_sem_feedback(ds_n3):
    fb = FakeBackend(QUADRATICA)
    executar_geracao(N3, "G1", fb, 0, 2, n_iteracoes_refinamento=3, datasets=ds_n3)
    assert fb.prompt_count == 2  # k só vale em G2
    assert all("Feedback" not in u for _, u in fb.calls)


def test_teste_nunca_usado_para_ajuste_selecao_ou_feedback(ds_n3):
    def responder(sistema, usuario):
        return CORRETA_N3 if "Feedback" in usuario else QUADRATICA

    rng = np.random.default_rng(99)
    te = ds_n3["teste"]
    corrompido = dataclasses.replace(
        te,
        y_true=rng.uniform(1e3, 1e4, te.y_true.shape),
        y_observed=rng.uniform(1e3, 1e4, te.y_observed.shape),
    )
    ds_c = {**ds_n3, "teste": corrompido}
    fa, fb_ = FakeBackend(responder), FakeBackend(responder)
    la, ca = executar_geracao(N3, "G2", fa, 0, 2, 1, datasets=ds_n3)
    lb, cb = executar_geracao(N3, "G2", fb_, 0, 2, 1, datasets=ds_c)
    assert fa.calls == fb_.calls
    assert la["expressao"] == lb["expressao"]
    assert [c["selecionado"] for c in ca] == [c["selecionado"] for c in cb]
    assert [c["rmse_log10_validacao"] for c in ca] == [c["rmse_log10_validacao"] for c in cb]
    assert [c["parametros_ajustados"] for c in ca] == [c["parametros_ajustados"] for c in cb]
    assert la["rmse_log10"] != lb["rmse_log10"]  # só a métrica de teste muda


def test_provedor_por_configuracao_e_sem_candidato_valido(ds_n3):
    cfg = ConfiguracaoLLM("fake", "roteiro-m2", resposta_fake="sem json")
    linha, cands = executar_geracao(N3, "G0", cfg, 0, 2, 0, max_retentativas=0, datasets=ds_n3)
    assert linha["modelo"] == "fake/roteiro-m2"
    assert linha["status"] == "erro" and linha["n_candidatos_validos"] == 0
    assert linha["classe"] == "erro de execução"
    assert all(c["status"] == "invalido" for c in cands)
