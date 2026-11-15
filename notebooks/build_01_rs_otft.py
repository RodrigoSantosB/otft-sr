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
$c$ = penalidade de plausibilidade física atribuída por LLM. Equações-alvo: escada N1–N4
ancorada no modelo OVSED com parâmetros da Org1.

| Condição experimental | Guia do LLM | $w_3$ |
|---|---|---|
| A | nenhum (RS pura, linha de base) | 0 |
| B | prompt genérico (física clássica, artigo) | 0,1 |
| C | prompt de física de dispositivo (OTFT) | 0,1 |

Premissas: `references/premissas_cientificas.md` (H-OTFT-01…15).

## 0. Ambiente e parâmetros
""")

code(r"""
# ---- PARÂMETROS EDITÁVEIS -------------------------------------------------------------
ORCAMENTO = "fumaca"            # "fumaca" (teste de encadeamento) | "completo" (best_config)
NIVEIS = ["N1", "N2", "N3", "N4"]
MOTORES = ["pysr", "deap", "gplearn"]
CONDICOES = ["A", "B", "C"]
RUIDOS = [0.0]                  # proposta §4.5: [0.0, 0.01, 0.05, 0.10]
SEMENTES = [0]
N_CURVAS, N_PONTOS = 20, 50     # curvas por nível, pontos por curva
BACKEND_LLM = "ollama"          # "nenhum" | "ollama" (mistral:7b-instruct, temperatura 0)
# Células B/C (LLM no laço): subconjunto da matriz; em fumaça, orçamento mínimo de DEAP.
NIVEIS_LLM, MOTORES_LLM = ["N2"], ["deap"]
ORCAMENTO_LLM_FUMACA = {"sr.population": 8, "sr.generations": 1, "sr.early_stopping.enabled": False}
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
from otft_sr.experimento import CRITERIO, PESOS_BASELINE, PESOS_LLM, controle_positivo, executar_condicao
from otft_sr.niveis import N5_DOCUMENTACAO, erro_vs_ovsed, escada, nivel

FUMACA = ORCAMENTO == "fumaca"
DIR_RESULTADOS = RAIZ / "results"
DIR_RESULTADOS.mkdir(exist_ok=True)
print(f"python: {sys.executable}")
print(f"orçamento: {ORCAMENTO}; pesos A = {tuple(round(w, 3) for w in PESOS_BASELINE)}, B/C = {PESOS_LLM}")
if FUMACA:
    print("ATENÇÃO: execução de FUMAÇA — valida o encadeamento; os números NÃO são resultados.")
""")

# ============================================================================ 1
md(r"""
## 1. Modelo OVSED e escada de equações

OVSED (canal p em módulo): $\theta = (V_{GS} - V_{th0} - \delta V_{DS})/(n V_T)$,
$q_{tot} = \mathrm{softplus}(\theta)$, $I_D = I_{leak} + W J_{th} q_{tot}^{l} F_{sat}$, com
$R_S$ por ponto fixo. $V_{th0} = 8{,}57$ V (H-OTFT-02).
""")

code(r"""
p = ovsed.carregar_parametros_org1()
curvas = ovsed.carregar_curvas_org1()
print("Parâmetros Org1:", p.como_dict())
print(f"RMSE log10|I_D| OVSED vs Org1 medida: V_th0 = 8,57 V -> {ovsed.erro_log_medido(p):.3f} déc; "
      f"V_th0 = 18,57 V -> {ovsed.erro_log_medido(ovsed.carregar_parametros_org1(V_th0=None)):.3f} déc")

linhas = []
for n in escada():
    e = erro_vs_ovsed(n)
    linhas.append({"nível": n.id, "nome": n.nome, "expressão de referência": n.expressao,
                   "I_D": n.unidades["I_D"], "região": n.regiao.descricao,
                   "erro rel. médio vs OVSED": round(e["erro_rel_medio"], 3),
                   "RMSE log10 vs OVSED (déc)": round(e["rmse_log10"], 3)})
tabela_niveis = pd.DataFrame(linhas).set_index("nível")
display(tabela_niveis)
for n in escada():
    print(f"{n.id}: {n.derivacao}")
print("N5:", N5_DOCUMENTACAO)
""")

code(r"""
med = {c.nome: c for c in curvas}
N1, N2, N3, N4 = escada()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

c50 = med["org1_50VDS"]; vg = np.linspace(0.3, 50, 300)
ax[0].semilogy(c50.V_GS, c50.I_D_uA, "k.", label="Org1 medida (V_DS = 50 V)")
ax[0].semilogy(vg, ovsed.corrente_ovsed_uA(vg, 50, p), "k-", lw=1, label="OVSED completo")
for n, cor in ((N1, "C0"), (N2, "C1")):
    v = np.linspace(*n.regiao.V_GS, 100)
    ax[0].semilogy(v, n.gerador(v, np.full_like(v, 50.0)), cor, lw=2, label=f"{n.id}")
c2 = med["org1_2VDS"]
ax[0].set(xlabel="V_GS (V)", ylabel="I_D (µA)", title="Transferência, saturação: N1, N2")
ax[0].set_ylim(1e-5, 30); ax[0].legend(fontsize=8)

for vgs, cor in ((30, "C2"), (40, "C3"), (50, "C4")):
    c = med[f"org1_{vgs}VGS"]; vd = np.linspace(0, 50, 200)
    ax[1].plot(c.V_DS, c.I_D_uA, ".", color=cor, ms=4, label=f"Org1 medida V_GS = {vgs} V")
    ax[1].plot(vd, ovsed.corrente_ovsed_uA(vgs, vd, p), "-", color=cor, lw=1)
    vdt = np.linspace(0, vgs - N3.constantes["V_Th"], 50)
    ax[1].plot(vdt, N3.gerador(np.full_like(vdt, vgs), vdt), "--", color=cor, lw=2)
ax[1].set(xlabel="V_DS (V)", ylabel="I_D (µA)", title="Saída: OVSED (—) e N3 no triodo (--)")
ax[1].legend(fontsize=7)

vg = np.linspace(0, 20, 200)
ax[2].semilogy(c2.V_GS, c2.I_D_uA * 1e6, "k.", label="Org1 medida (V_DS = 2 V)")
ax[2].semilogy(vg, ovsed.corrente_ovsed_uA(vg, 2, p) * 1e6, "k-", lw=1, label="OVSED completo")
ax[2].semilogy(vg, ovsed.corrente_ovsed_uA(vg, 2, p, com_leak=False) * 1e6, "k:", label="OVSED sem I_leak")
v = np.linspace(*N4.regiao.V_GS, 50)
ax[2].semilogy(v, N4.gerador(v, np.full_like(v, 2.0)), "C5", lw=2, label="N4")
ax[2].set(xlabel="V_GS (V)", ylabel="I_D (pA)", title="Sublimiar: N4 (I_leak removido)")
ax[2].legend(fontsize=8)
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
    resumo.append({"nível": nid, "tipo de curva": ds["treino"].metadata["tipo_curva"],
                   **{f"curvas {k}": len(v) for k, v in ids.items()},
                   **{f"pontos {k}": d.y_observed.size for k, d in ds.items()},
                   "σ": RUIDOS[-1]})
display(pd.DataFrame(resumo).set_index("nível"))

fig, ax = plt.subplots(1, len(NIVEIS), figsize=(4 * len(NIVEIS), 3.2))
for a, nid in zip(np.atleast_1d(ax), NIVEIS):
    ds = construir_datasets(nivel(nid), RUIDOS[-1], SEMENTES[0], N_CURVAS, N_PONTOS)
    for k, m in (("treino", "."), ("validacao", "x"), ("teste", "+")):
        d = ds[k]
        a.scatter(d.X[:, 0], d.X[:, 1], s=4, marker=m, label=k)
    a.set(title=nid, xlabel="V_GS (V)", ylabel="V_DS (V)")
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
display(cp[["expressao", "n_nos", "e_normalizado", "one_minus_r2", "rmse_log10",
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
        for motor in MOTORES:
            for sigma in RUIDOS:
                for sem in SEMENTES:
                    r = executar_condicao(nivel(nid), motor, "A", sigma, sem, ORCAMENTO,
                                          n_curvas=N_CURVAS, n_pontos=N_PONTOS)
                    resultados.append(r)
                    print(f"A {nid} {motor:7s} σ={sigma:<4} s={sem} [{r['status']}] "
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
    orc_llm = ORCAMENTO_LLM_FUMACA if FUMACA else ORCAMENTO
    for cond in condicoes_llm:
        avaliador = criar_avaliador(cond)
        for nid in NIVEIS_LLM:
            for motor in MOTORES_LLM:
                for sigma in RUIDOS:
                    for sem in SEMENTES:
                        r = executar_condicao(nivel(nid), motor, cond, sigma, sem, orc_llm,
                                              avaliador=avaliador, n_curvas=N_CURVAS, n_pontos=N_PONTOS)
                        r["orcamento"] = "fumaca" if FUMACA else r["orcamento"]
                        resultados.append(r)
                        print(f"{cond} {nid} {motor:7s} σ={sigma:<4} s={sem} [{r['status']}] "
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
if FUMACA:
    print("FUMAÇA — tabela apenas demonstra o formato; NÃO é resultado.")
cols = ["nivel", "motor", "condicao", "ruido", "semente", "status", "expressao", "n_nos",
        "one_minus_r2", "rmse_log10", "tree_distance", "tree_distance_algebraic",
        "forma_funcional_sympy", "classe",
        "tempo_s", "chamadas_llm", "tempo_llm_s"]
display(df.reindex(columns=cols))
if len(df):
    ok = df[df["status"] == "ok"]
    resumo = ok.groupby(["nivel", "motor", "condicao"]).agg(
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
    for (motor, cond), g in ok.groupby(["motor", "condicao"]):
        ax.scatter(g["n_nos"], np.maximum(g["one_minus_r2"], 1e-16), marker=marcadores[motor],
                   color=cores[cond], label=f"{motor} / {cond}", alpha=0.8)
    for nid in sorted(ok["nivel"].unique()):
        ax.axvline(cp.loc[nid, "n_nos"], color="grey", lw=0.5, ls=":")
    ax.set(xscale="linear", yscale="log", xlabel="complexidade (nós)", ylabel="1 − R² (teste)",
           title="Erro × complexidade" + (" — FUMAÇA" if FUMACA else ""))
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(DIR_RESULTADOS / f"fig_pareto_{ORCAMENTO}.png", dpi=150); plt.show()
""")

# ============================================================================ 7
md(r"""
## 7. Exportação
""")

code(r"""
carimbo = time.strftime("%Y%m%d-%H%M%S")
base = DIR_RESULTADOS / f"rs_otft_{ORCAMENTO}_{carimbo}"
df.to_csv(base.with_suffix(".csv"), index=False)
meta = {"orcamento": ORCAMENTO, "fumaca": FUMACA, "niveis": NIVEIS, "motores": MOTORES,
        "condicoes": CONDICOES, "ruidos": RUIDOS, "sementes": SEMENTES, "backend_llm": BACKEND_LLM,
        "criterio": {"max_one_minus_r2": CRITERIO.max_one_minus_r2,
                     "max_tree_distance": CRITERIO.max_tree_distance},
        "niveis_referencia": {n.id: n.expressao for n in escada()},
        "parametros_org1": p.como_dict(),
        "linhas": json.loads(df.to_json(orient="records"))}
base.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
print("exportado:", base.with_suffix(".csv").name, base.with_suffix(".json").name)
""")


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [
        nbf.v4.new_markdown_cell(t) if k == "md" else nbf.v4.new_code_cell(t) for k, t in celulas
    ]
    nbf.write(nb, SAIDA)
    print(f"escrito: {SAIDA} ({len(nb.cells)} células)")


if __name__ == "__main__":
    main()
