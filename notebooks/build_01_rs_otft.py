"""Gera ``notebooks/01_rs_otft.ipynb`` (edite este arquivo, não o .ipynb).

Uso: ``<venv>/bin/python notebooks/build_01_rs_otft.py``
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

AQUI = Path(__file__).resolve().parent
SAIDA = AQUI / "01_rs_otft.ipynb"

celulas: list[tuple[str, str]] = []


def md(texto: str) -> None:
    celulas.append(("md", texto.strip("\n")))


def code(texto: str) -> None:
    celulas.append(("code", texto.strip("\n")))


# ============================================================================ 0
md(r"""
# Regressão simbólica guiada por LLM para OTFTs (Org1, OVSED)

Adaptação de Taskin, Xie & Lazebnik (Sci Rep 2026): perda $L = w_1 e + w_2 s + w_3 c$, com
$c$ = penalidade de plausibilidade física atribuída por LLM. Equações-alvo: os quatro
níveis da proposta (Seção 5.1), ancorados no modelo OVSED com parâmetros da Org1.

| Condição experimental | Guia do LLM | $w_3$ |
|---|---|---|
| A | nenhum (RS pura, linha de base) | 0 |
| B | prompt genérico (física clássica, artigo) | 0,1 |
| C | prompt de física de dispositivo (OTFT) | 0,1 |

Premissas: `references/premissas_cientificas.md` (H-OTFT-01…18).

## 0. Ambiente e parâmetros
""")

code(r"""
# ---- PARÂMETROS EDITÁVEIS -------------------------------------------------------------
CONFIGURACAO_BUSCA = "verificacao"  # "verificacao" (integridade do pipeline) | "calibrada" (best_config)
NIVEIS = ["N1", "N2", "N3", "N4"]
ALGORITMOS_RS = ["pysr", "deap", "gplearn"]
CONDICOES = ["A", "B", "C"]
RUIDOS = [0.0]                  # proposta §4.5: [0.0, 0.01, 0.05, 0.10]
SEMENTES = [0]
N_CURVAS, N_PONTOS = 20, 50     # curvas por nível, pontos por curva
BACKEND_LLM = "ollama"          # "nenhum" | "ollama" (mistral:7b-instruct, temperatura 0)
# Ensaios B/C (LLM no laço): subconjunto do desenho experimental; em verificação, DEAP mínimo.
NIVEIS_LLM, ALGORITMOS_RS_LLM = ["N3"], ["deap"]
CONFIGURACAO_BUSCA_LLM_VERIFICACAO = {"sr.population": 8, "sr.generations": 1, "sr.early_stopping.enabled": False}
# ---------------------------------------------------------------------------------------
""")

code(r"""
import json, sys, time, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(RAIZ / "src"))
warnings.filterwarnings("ignore")

from otft_sr import ovsed
from otft_sr.contexto import contexto_fisico, criar_avaliador, ollama_disponivel
from otft_sr.dados import construir_datasets
from otft_sr.experimento import CRITERIO, PESOS_LINHA_DE_BASE, PESOS_LLM, controle_positivo, executar_condicao
from otft_sr.niveis import (N5_DOCUMENTACAO, discriminabilidade_quadratica, erro_vs_medida_org1,
                            erro_vs_ovsed, escada, nivel)

# Rótulo científico "N<k> — nome" de cada nível (índice = nível da proposta, Seção 5.1).
ROTULO = {n.id: f"{n.id} — {n.nome[0].lower()}{n.nome[1:]}" for n in escada()}

EXECUCAO_DE_VERIFICACAO = CONFIGURACAO_BUSCA == "verificacao"
DIR_RESULTADOS = RAIZ / "results"
DIR_RESULTADOS.mkdir(exist_ok=True)
print(f"python: {sys.executable}")
print(f"configuração de busca: {CONFIGURACAO_BUSCA}; pesos A = {tuple(round(w, 3) for w in PESOS_LINHA_DE_BASE)}, B/C = {PESOS_LLM}")
if EXECUCAO_DE_VERIFICACAO:
    print("EXECUÇÃO DE VERIFICAÇÃO — valida a integridade do pipeline; não constitui resultado.")
""")

# ============================================================================ 1
md(r"""
## 1. Modelo OVSED e escada de equações

| Id | Nome científico | Equação-alvo |
|---|---|---|
| N1 | saturação quadrática ideal | $I_D = K (V_{GS} - V_{Th})^2$ |
| N2 | regime linear (triodo) com termo quadrático em $V_{DS}$ | $I_D = K (2 (V_{GS} - V_{Th}) V_{DS} - V_{DS}^2)$ |
| N3 | saturação com mobilidade em lei de potência (perto do limiar) | $I_D = K (V_{GS} - V_{th0})^{l}$ |
| N4 | corrente sublimiar exponencial com fuga aditiva | $I_D = I_{leak} + I_0 \exp(\ln 10\, V_{GS}/S)$ |
| N5 | saturação com resistência de contato | implícita em $I_D$; só documentado (H-OTFT-08) |

OVSED (canal p em módulo): $\theta = (V_{GS} - V_{th0} - \delta V_{DS})/(n V_T)$,
$q_{tot} = \mathrm{softplus}(\theta)$, $I_D = I_{leak} + W J_{th} q_{tot}^{l} F_{sat}$, com
$R_S$ por ponto fixo. $V_{th0} = 8{,}57$ V (H-OTFT-02).

Regiões de N3 e N4 (H-OTFT-17, H-OTFT-18): o N3 desce até perto do limiar
($V_{GS}$ de 10,5 a 28 V), onde a lei de potência se separa da lei quadrática; o critério é a
discriminabilidade, isto é, o erro do melhor ajuste $K (V_{GS} - V_T)^2$ aos dados do N3. O N4
inclui a fuga $I_{leak}$ como termo aditivo e cobre o sublimiar observável na Org1 medida
($V_{GS}$ de 0 a 10 V, $V_{DS}$ de 1 a 5 V); seu $S$ é um swing efetivo da transição (H-OTFT-12).
""")

code(r"""
p = ovsed.carregar_parametros_org1()
curvas = ovsed.carregar_curvas_org1()
print("Parâmetros Org1:", p.como_dict())
print(f"RMSE log10|I_D| OVSED vs Org1 medida: V_th0 = 8,57 V -> {ovsed.erro_log_medido(p):.3f} déc; "
      f"V_th0 = 18,57 V -> {ovsed.erro_log_medido(ovsed.carregar_parametros_org1(V_th0=None)):.3f} déc")

linhas = []
for n in escada():
    e, m = erro_vs_ovsed(n), erro_vs_medida_org1(n, curvas)
    linhas.append({"nível": ROTULO[n.id], "nome": n.nome, "expressão de referência": n.expressao,
                   "I_D": n.unidades["I_D"], "região": n.regiao.descricao,
                   "erro rel. médio vs OVSED": round(e["erro_rel_medio"], 3),
                   "RMSE log10 vs OVSED (déc)": round(e["rmse_log10"], 3),
                   "RMSE log10 vs Org1 medida (déc)": round(m["rmse_log10"], 3),
                   "pontos medidos na região": m["n_pontos"]})
tabela_niveis = pd.DataFrame(linhas).set_index("nível")
display(tabela_niveis)

# Discriminabilidade do N3 (H-OTFT-18): melhor K (V_GS - V_T)^2 ajustado aos dados do N3.
disc = discriminabilidade_quadratica(nivel("N3"))
print(f"N3 vs melhor ajuste quadrático: RMSE log10 = {disc['rmse_log10_quadratica']:.3f} déc "
      f"(V_T = {disc['V_T_quadratica_log']:.2f} V); 1 - R² = {disc['one_minus_r2_quadratica']:.2e} "
      f"(V_T = {disc['V_T_quadratica']:.2f} V). Para o próprio N3 ambos são 0.")
for n in escada():
    print(f"{ROTULO[n.id]}: {n.derivacao}")
print(N5_DOCUMENTACAO)
""")

code(r"""
med = {c.nome: c for c in curvas}
sat_quad, triodo, sat_potencia, sublimiar = escada()
fig, ax = plt.subplots(2, 3, figsize=(16, 8.4))
c50, c2 = med["org1_50VDS"], med["org1_2VDS"]
cor_n = {"N1": "C0", "N2": "C2", "N3": "C1", "N4": "C3", "quad": "C4"}
r3 = sat_potencia.regiao.V_GS

# (a) saturação: N1 e N3 nas respectivas regiões
a = ax[0, 0]; vg = np.linspace(0.3, 50, 300)
a.semilogy(c50.V_GS, c50.I_D_uA, "k.", label="Org1 medida (V_DS = 50 V)")
a.semilogy(vg, ovsed.corrente_ovsed_uA(vg, 50, p), "k-", lw=1, label="OVSED completo")
for n in (sat_quad, sat_potencia):
    v = np.linspace(*n.regiao.V_GS, 100)
    a.semilogy(v, n.gerador(v, np.full_like(v, 50.0)), cor_n[n.id], lw=2, label=ROTULO[n.id])
a.axvspan(*r3, color=cor_n["N3"], alpha=0.08)
a.set(xlabel="V_GS (V)", ylabel="I_D (µA)", title="(a) Saturação: N1 (25–50 V) e N3 (10,5–28 V)", ylim=(1e-5, 30))
a.legend(fontsize=7)

# (b) região do N3: lei de potência contra a forma quadrática de N1
a = ax[0, 1]; v = np.linspace(*r3, 200); vd50 = np.full_like(v, 50.0)
y3 = sat_potencia.gerador(v, vd50)
quad = disc["K_quadratica_log"] * (v - disc["V_T_quadratica_log"]) ** 2
m50 = (c50.V_GS >= r3[0]) & (c50.V_GS <= r3[1])
a.semilogy(c50.V_GS[m50], c50.I_D_uA[m50], "k.", label="Org1 medida (V_DS = 50 V)")
a.semilogy(v, ovsed.corrente_ovsed_uA(v, 50, p), "k-", lw=1, label="OVSED completo")
a.semilogy(v, y3, cor_n["N3"], lw=2, label=ROTULO["N3"])
a.semilogy(v, quad, "--", color=cor_n["quad"], lw=2,
           label=f"melhor K (V_GS − V_T)² ajustado ao N3 (V_T = {disc['V_T_quadratica_log']:.1f} V)")
vn1 = v[v > sat_quad.constantes["V_Th"]]
a.semilogy(vn1, sat_quad.gerador(vn1, np.full_like(vn1, 50.0)), ":", color=cor_n["N1"], lw=2,
           label="N1 extrapolado (V_GS > V_Th)")
a.set(xlabel="V_GS (V)", ylabel="I_D (µA)", title="(b) Região do N3: lei de potência × quadrática")
a.legend(fontsize=7)

# (c) desvio em relação ao N3, em década
a = ax[0, 2]
a.axhline(0, color=cor_n["N3"], lw=2, label="N3 (referência)")
a.plot(v, np.log10(quad / y3), "--", color=cor_n["quad"], lw=2,
       label=f"melhor quadrática (RMSE {disc['rmse_log10_quadratica']:.3f} déc)")
a.plot(v, np.log10(ovsed.corrente_ovsed_uA(v, 50, p) / y3), "k-", lw=1, label="OVSED completo (V_DS = 50 V)")
a.plot(c50.V_GS[m50], np.log10(c50.I_D_uA[m50] / sat_potencia.gerador(c50.V_GS[m50], c50.V_DS[m50])), "k.",
       label="Org1 medida")
a.set(xlabel="V_GS (V)", ylabel="log10(I_D / I_D,N3) (déc)", title="(c) Desvio em relação ao N3")
a.legend(fontsize=7)

# (d) N2, triodo
a = ax[1, 0]
for vgs, cor in ((30, "C2"), (40, "C3"), (50, "C4")):
    c = med[f"org1_{vgs}VGS"]; vd = np.linspace(0, 50, 200)
    a.plot(c.V_DS, c.I_D_uA, ".", color=cor, ms=4, label=f"Org1 medida V_GS = {vgs} V")
    a.plot(vd, ovsed.corrente_ovsed_uA(vgs, vd, p), "-", color=cor, lw=1)
    vdt = np.linspace(0, vgs - triodo.constantes["V_Th"], 50)
    a.plot(vdt, triodo.gerador(np.full_like(vdt, vgs), vdt), "--", color=cor, lw=2)
a.set(xlabel="V_DS (V)", ylabel="I_D (µA)", title="(d) Saída: OVSED (—) e N2, triodo (--)")
a.legend(fontsize=7)

# (e) N4 com fuga aditiva sobre a medida a V_DS = 2 V
a = ax[1, 1]; vg = np.linspace(0, 20, 300); r4 = sublimiar.regiao.V_GS
I_leak = sublimiar.constantes["I_leak"]
a.semilogy(c2.V_GS, c2.I_D_uA * 1e6, "k.", label="Org1 medida (V_DS = 2 V)")
a.semilogy(vg, ovsed.corrente_ovsed_uA(vg, 2, p) * 1e6, "k-", lw=1, label="OVSED completo")
a.semilogy(vg, ovsed.corrente_ovsed_uA(vg, 2, p, com_leak=False) * 1e6, "k:", label="OVSED sem I_leak (canal)")
v = np.linspace(*r4, 100)
a.semilogy(v, sublimiar.gerador(v, np.full_like(v, 2.0)), cor_n["N4"], lw=2, label=ROTULO["N4"])
a.semilogy(v, sublimiar.gerador(v, np.full_like(v, 2.0)) - I_leak, "--", color=cor_n["N4"], lw=1,
           label="termo exponencial do N4 (sem fuga)")
a.axhline(I_leak, color="grey", lw=0.8, ls="-.", label=f"I_leak = {I_leak:g} pA")
a.axvspan(*r4, color=cor_n["N4"], alpha=0.08)
a.set(xlabel="V_GS (V)", ylabel="I_D (pA)", title=f"(e) N4 com fuga aditiva (S = {sublimiar.constantes['S']:.3g} V/déc)",
      xlim=(0, 20), ylim=(1e-3, 3e4))
a.legend(fontsize=7)

# (f) N4 e OVSED contra a medida na região do N4
a = ax[1, 2]; m2 = (c2.V_GS >= r4[0]) & (c2.V_GS <= r4[1])
vgm, Im = c2.V_GS[m2], c2.I_D_uA[m2] * 1e6
a.axhline(0, color="k", lw=0.5)
a.plot(vgm, np.log10(sublimiar.gerador(vgm, c2.V_DS[m2]) / Im), "o", color=cor_n["N4"],
       label=f"N4 (RMSE {erro_vs_medida_org1(sublimiar, curvas)['rmse_log10']:.3f} déc)")
a.plot(vgm, np.log10(ovsed.corrente_ovsed_uA(vgm, 2, p) * 1e6 / Im), "ks", mfc="none", label="OVSED completo")
a.set(xlabel="V_GS (V)", ylabel="log10(I_D,modelo / I_D,medida) (déc)", title="(f) Região do N4: desvio contra a medida",
      xlim=(4, r4[1]), ylim=(-0.3, 0.3))
a.legend(fontsize=7)

fig.tight_layout(); fig.savefig(DIR_RESULTADOS / "fig_niveis_vs_ovsed_org1.png", dpi=150)
plt.show()
""")

# ============================================================================ 2
md(r"""
## 2. Dados

Grade por curvas (transferência a $V_{DS}$ fixo ou saída a $V_{GS}$ fixo), pontos sorteados
na região do nível, partição treino/validação/teste por curva (60/20/20 %), ruído gaussiano
multiplicativo $I_{obs} = I_D (1 + \sigma \mathcal{N}(0,1))$.
""")

code(r"""
resumo = []
for nid in NIVEIS:
    ds = construir_datasets(nivel(nid), RUIDOS[-1], SEMENTES[0], N_CURVAS, N_PONTOS)
    ids = {k: set(np.unique(d.simulation_ids)) for k, d in ds.items()}
    assert not (ids["treino"] & ids["teste"]) and not (ids["treino"] & ids["validacao"])
    assert not (ids["validacao"] & ids["teste"])
    resumo.append({"nível": ROTULO[nid], "tipo de curva": ds["treino"].metadata["tipo_curva"],
                   **{f"curvas {k}": len(v) for k, v in ids.items()},
                   **{f"pontos {k}": d.y_observed.size for k, d in ds.items()},
                   "σ": RUIDOS[-1]})
display(pd.DataFrame(resumo).set_index("nível"))

fig, ax = plt.subplots(1, len(NIVEIS), figsize=(4 * len(NIVEIS), 3.6))
for a, nid in zip(np.atleast_1d(ax), NIVEIS):
    ds = construir_datasets(nivel(nid), RUIDOS[-1], SEMENTES[0], N_CURVAS, N_PONTOS)
    for k, m in (("treino", "."), ("validacao", "x"), ("teste", "+")):
        d = ds[k]
        a.scatter(d.X[:, 0], d.X[:, 1], s=4, marker=m, label=k)
    a.set(title=ROTULO[nid].replace(" — ", "\n"), xlabel="V_GS (V)", ylabel="V_DS (V)")
np.atleast_1d(ax)[0].legend(fontsize=7)
fig.tight_layout(); plt.show()
""")

# ============================================================================ 3
md(r"""
## 3. Controle positivo do pipeline

A expressão de referência, avaliada pelo mesmo objetivo e pelas mesmas métricas da busca,
deve dar $1 - R^2 \approx 0$, $e \approx 0$ e distância de árvore 0.
""")

code(r"""
cp = pd.DataFrame([controle_positivo(nivel(nid)) for nid in NIVEIS]).set_index("nivel")
display(cp.assign(rotulo=[ROTULO[i] for i in cp.index]).set_index("rotulo")[["expressao", "n_nos", "e_normalizado", "one_minus_r2", "rmse_log10",
            "tree_distance", "tree_distance_algebraic"]])
assert (cp["one_minus_r2"] < 1e-12).all() and (cp["tree_distance"] == 0).all()
assert (cp["tree_distance_algebraic"] == 0).all()
print("controle positivo: OK")
""")

# ============================================================================ 4
md(r"""
## 4. Linha de base sem LLM (condição A)
""")

code(r"""
resultados = []
if "A" in CONDICOES:
    for nid in NIVEIS:
        for algoritmo in ALGORITMOS_RS:
            for sigma in RUIDOS:
                for sem in SEMENTES:
                    r = executar_condicao(nivel(nid), algoritmo, "A", sigma, sem, CONFIGURACAO_BUSCA,
                                          n_curvas=N_CURVAS, n_pontos=N_PONTOS)
                    resultados.append(r)
                    print(f"A {ROTULO[nid]} {algoritmo:7s} σ={sigma:<4} s={sem} [{r['status']}] "
                          f"{r.get('expressao', r.get('erro'))!s:.70} | 1-R²={r.get('one_minus_r2', float('nan')):.2e} "
                          f"| {r['classe']} | {r['tempo_s']:.1f}s")
""")

# ============================================================================ 5
md(r"""
## 5. Condições B e C (LLM no laço)

Variante de prompt E (variáveis + descrição do experimento, sem a equação-alvo). B usa
`prompts/generico`, C usa `prompts/dispositivo`; o contexto físico é o mesmo (H-OTFT-09).
""")

code(r"""
condicoes_llm = [c for c in CONDICOES if c in ("B", "C")]
motivo = None
if not condicoes_llm:
    motivo = "nenhuma condição com LLM selecionada"
elif BACKEND_LLM == "nenhum":
    motivo = "BACKEND_LLM = 'nenhum'"
elif not ollama_disponivel():
    motivo = "Ollama indisponível ou sem mistral:7b-instruct"
if motivo:
    print(f"Condições B/C PULADAS: {motivo}.")
else:
    busca_llm = (
        CONFIGURACAO_BUSCA_LLM_VERIFICACAO if EXECUCAO_DE_VERIFICACAO else CONFIGURACAO_BUSCA
    )
    for cond in condicoes_llm:
        avaliador = criar_avaliador(cond)
        for nid in NIVEIS_LLM:
            for algoritmo in ALGORITMOS_RS_LLM:
                for sigma in RUIDOS:
                    for sem in SEMENTES:
                        r = executar_condicao(nivel(nid), algoritmo, cond, sigma, sem, busca_llm,
                                              avaliador=avaliador, n_curvas=N_CURVAS, n_pontos=N_PONTOS)
                        r["configuracao_busca"] = "verificacao" if EXECUCAO_DE_VERIFICACAO else r["configuracao_busca"]
                        resultados.append(r)
                        print(f"{cond} {ROTULO[nid]} {algoritmo:7s} σ={sigma:<4} s={sem} [{r['status']}] "
                              f"{r.get('expressao', r.get('erro'))!s:.60} | 1-R²={r.get('one_minus_r2', float('nan')):.2e} "
                              f"| chamadas LLM={r['chamadas_llm']} ({r['tempo_llm_s']:.0f}s) | {r['tempo_s']:.1f}s")
""")

# ============================================================================ 6
md(r"""
## 6. Resultados

Critério de recuperação simbólica: $1 - R^2 \le 10^{-3}$ (teste limpo) e
$\min$(distância primária, equivalente, algébrica) $\le 0{,}05$, ou mesma forma funcional
por manipulação simbólica (sympy, H-OTFT-07b).
""")

code(r"""
df = pd.DataFrame(resultados)
if EXECUCAO_DE_VERIFICACAO:
    print("EXECUÇÃO DE VERIFICAÇÃO — a tabela demonstra apenas o formato; não constitui resultado.")
cols = ["nivel", "nivel_proposta", "algoritmo", "condicao", "ruido", "semente", "status", "expressao", "n_nos",
        "one_minus_r2", "rmse_log10", "tree_distance", "tree_distance_algebraic",
        "forma_funcional_sympy", "classe",
        "tempo_s", "chamadas_llm", "tempo_llm_s"]
display(df.reindex(columns=cols))
if len(df):
    ok = df[df["status"] == "ok"]
    resumo = ok.groupby(["nivel_proposta", "nivel", "algoritmo", "condicao"]).agg(
        recupera=("classe", lambda s: float(np.mean(s.str.startswith("recupera estrutura")))),
        one_minus_r2_mediana=("one_minus_r2", "median"),
        rmse_log10_mediana=("rmse_log10", "median"),
        dist_algebrica_mediana=("tree_distance_algebraic", "median"),
        nos_mediana=("n_nos", "median"),
        tempo_s=("tempo_s", "median"),
        chamadas_llm=("chamadas_llm", "sum"),
    )
    display(resumo)
""")

code(r"""
if len(df) and (df["status"] == "ok").any():
    ok = df[df["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    marcadores = {"pysr": "o", "deap": "s", "gplearn": "^"}
    cores = {"A": "C0", "B": "C1", "C": "C2"}
    for (algoritmo, cond), g in ok.groupby(["algoritmo", "condicao"]):
        ax.scatter(g["n_nos"], np.maximum(g["one_minus_r2"], 1e-16), marker=marcadores[algoritmo],
                   color=cores[cond], label=f"{algoritmo} / {cond}", alpha=0.8)
    for nid in sorted(ok["nivel"].unique()):
        ax.axvline(cp.loc[nid, "n_nos"], color="grey", lw=0.5, ls=":")
    ax.set(xscale="linear", yscale="log", xlabel="complexidade (nós)", ylabel="1 − R² (teste)",
           title="Erro × complexidade" + (" — EXECUÇÃO DE VERIFICAÇÃO" if EXECUCAO_DE_VERIFICACAO else ""))
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(DIR_RESULTADOS / f"fig_pareto_{CONFIGURACAO_BUSCA}.png", dpi=150); plt.show()
""")

# ============================================================================ 7
md(r"""
## 7. Exportação
""")

code(r"""
marca_temporal = time.strftime("%Y%m%d-%H%M%S")
base = DIR_RESULTADOS / f"rs_otft_{CONFIGURACAO_BUSCA}_{marca_temporal}"
df.to_csv(base.with_suffix(".csv"), index=False)
meta = {"configuracao_busca": CONFIGURACAO_BUSCA, "execucao_de_verificacao": EXECUCAO_DE_VERIFICACAO,
        "niveis": NIVEIS, "algoritmos_rs": ALGORITMOS_RS,
        "condicoes": CONDICOES, "ruidos": RUIDOS, "sementes": SEMENTES, "backend_llm": BACKEND_LLM,
        "criterio": {"max_one_minus_r2": CRITERIO.max_one_minus_r2,
                     "max_tree_distance": CRITERIO.max_tree_distance},
        "niveis_referencia": {n.id: n.expressao for n in escada()},
        "parametros_org1": p.como_dict(),
        "linhas": json.loads(df.to_json(orient="records"))}
base.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
print("exportado:", base.with_suffix(".csv").name, base.with_suffix(".json").name)
""")


def construir() -> nbf.NotebookNode:
    """Monta o notebook a partir da lista ``celulas`` declarada acima."""
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [
        nbf.v4.new_markdown_cell(t) if k == "md" else nbf.v4.new_code_cell(t) for k, t in celulas
    ]
    return nb


def main(saida: Path = SAIDA) -> None:
    nb = construir()
    nbf.write(nb, saida)
    print(f"escrito: {saida} ({len(nb.cells)} células)")


if __name__ == "__main__":
    main()
