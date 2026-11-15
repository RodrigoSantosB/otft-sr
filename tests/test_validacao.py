"""Validação da implementação (R2): métrica rmse_log10 e execução de um ensaio."""

import math

import numpy as np
import pytest

from otft_sr import experimento
from otft_sr.experimento import executar_condicao, rmse_log10
from otft_sr.niveis import escada

N1 = escada()[0]


def test_rmse_log10_contra_valor_calculado_a_mao():
    # resíduos em décadas: +1, 0, -3  ->  RMSE = sqrt((1 + 0 + 9) / 3)
    y = np.array([1e-3, 1e-2, 1e-1])
    yhat = np.array([1e-2, 1e-2, 1e-4])
    rmse, nao_pos = rmse_log10(y, yhat)
    assert rmse == pytest.approx(math.sqrt(10 / 3), rel=1e-12)
    assert nao_pos == 0.0


def test_rmse_log10_predicao_nao_positiva_usa_piso_e_conta_fracao():
    # piso = 1e-6 * min(y>0) = 1e-6; yhat=-5 -> log10(piso) = -6 contra log10(1) = 0
    y = np.array([1.0, 10.0])
    yhat = np.array([-5.0, 10.0])
    rmse, nao_pos = rmse_log10(y, yhat)
    assert rmse == pytest.approx(math.sqrt(36 / 2), rel=1e-12)
    assert nao_pos == 0.5


CHAVES = {
    "nivel", "algoritmo", "condicao", "ruido", "semente", "configuracao_busca", "expressao",
    "one_minus_r2", "rmse_log10", "tree_distance_algebraic", "forma_funcional_sympy",
    "classe", "chamadas_llm", "status",
}


def test_executar_condicao_deap_verificacao_devolve_linha_completa():
    linha = executar_condicao(N1, "deap", "A", 0.0, 0, "verificacao", n_curvas=4, n_pontos=20)
    assert CHAVES <= set(linha)
    assert linha["status"] == "ok"
    assert linha["nivel"] == N1.id
    assert linha["algoritmo"] == "deap"
    assert linha["configuracao_busca"] == "verificacao"
    assert linha["chamadas_llm"] == 0  # condição A nunca chama o LLM
    assert linha["rmse_log10"] >= 0


def test_executar_condicao_com_erro_registra_status_erro_sem_excecao(monkeypatch):
    def falha(_algoritmo):
        raise RuntimeError("falha forçada")

    monkeypatch.setattr(experimento, "make_regressor", falha)
    linha = executar_condicao(N1, "deap", "A", 0.0, 0, "verificacao", n_curvas=4, n_pontos=20)
    assert linha["status"] == "erro"
    assert "RuntimeError: falha forçada" in linha["erro"]
    assert linha["classe"] != "recupera estrutura"
