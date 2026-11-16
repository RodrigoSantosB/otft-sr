"""Gera ``notebooks/02_multi_llm_descoberta_otft_colab.ipynb`` (edite este arquivo, não o .ipynb).

Uso: ``<venv>/bin/python notebooks/build_02_multi_llm_colab.py``
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

AQUI = Path(__file__).resolve().parent
SAIDA = AQUI / "02_multi_llm_descoberta_otft_colab.ipynb"

celulas: list[tuple[str, str]] = []


def md(texto: str) -> None:
    celulas.append(("md", texto.strip("\n")))


def code(texto: str) -> None:
    celulas.append(("code", texto.strip("\n")))


# ============================================================================ 1
md(r"""
# Descoberta de equações de OTFT com múltiplos LLMs (Colab)

## 1. Introdução e objetivos

**Pergunta científica.** Um LLM melhora a descoberta das equações de corrente de dreno de
OTFTs (níveis N1–N4, ancorados no OVSED com parâmetros da Org1), seja como avaliador de
plausibilidade dentro da regressão simbólica (RS), seja como gerador de equações?

| Modo | Papel do LLM | Condições | Estatuto |
|---|---|---|---|
| M1 avaliador | termo $c$ da perda $L = w_1 e + w_2 s + w_3 c$ (método do artigo) | A = RS sem LLM; B = prompt genérico; C = prompt de física de dispositivo | comparável ao artigo |
| M2 gerador | propõe a equação; parâmetros ajustados numericamente | G0 = só dados; G1 = + contexto físico; G2 = G1 + refinamento iterativo | exploratório |

**O que cada comparação permite concluir.** A×B e A×C (mesma configuração de busca, pareadas
por semente): efeito do avaliador sobre a recuperação. B×C: efeito do conteúdo físico do
prompt. G0×G1: efeito do contexto físico na geração (mesmo número de candidatos). G1×G2:
efeito do refinamento, com esforço maior (exploratória). M1×M2: só sob custo efetivo
(exploratória). B/C só são interpretáveis se a auditoria da seção 9 mostrar que o modelo
discrimina a referência de formas erradas.

**Uso.** Edite as seções 2 (ambiente) e 4 (configuração única) e execute em ordem. Toda etapa
longa é retomável: após queda da sessão, execute de novo as seções 2–4 e continue; ensaios já
gravados em `results/<id>/` com `status = "ok"` são pulados e ensaios com `erro_execucao`
(falha de infraestrutura: exceção, memória, tempo esgotado, API) são repetidos. Premissas: `references/premissas_cientificas.md`
(H-OTFT-01…20).
""")

# ============================================================================ 2
md(r"""
## 2. Ambiente (Colab ou local)

**Finalidade:** preparar código, dados e caminhos. **Método:** no Colab, monta o Drive, clona o
`physics-llm-sr` no commit fixado, extrai o pacote `otft-sr-<commit>.zip` de
`MyDrive/otft-sr/` para o disco local e define `HF_HOME` no disco local; fora do Colab usa o
checkout atual (nada é instalado). **Resultado esperado:** caminhos impressos e relatório de
CPU/GPU/memória.
""")

code(r"""
# ---- PARÂMETROS DE AMBIENTE (o restante da configuração fica na seção 4) ---------------
PHYSICS_LLM_SR_URL = "https://github.com/RodrigoSantosB/physics-llm-sr.git"
PHYSICS_LLM_SR_COMMIT = "0436128b7968b1251a445241f01add21d86da6a1"  # commit validado
DIR_DRIVE = "/content/drive/MyDrive/otft-sr"  # pacote otft-sr-<commit>.zip e cópia dos resultados
PACOTE_ZIP = None              # None = o otft-sr-*.zip mais recente em DIR_DRIVE
DIR_TRABALHO = "/content/otft-sr"            # disco local do Colab (rápido)
DIR_PHYSICS_LLM_SR = "/content/physics-llm-sr"
HF_HOME_DIR = "/content/hf_home"             # pesos no disco local (o Drive trunca arquivos grandes)
USAR_PYSR = False              # instala pysr/juliacall (obrigatório se "pysr" estiver em algoritmos_rs)
FORCAR_TORCH_FIXADO = False    # True reinstala o torch fixado (pode romper o CUDA da imagem do Colab)
# ---------------------------------------------------------------------------------------
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    EM_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:  # sem o pacote "google"
    EM_COLAB = False


def extrair_pacote(zip_path: Path, destino: Path) -> None:
    # Recusa membros com caminho fora do destino antes de extrair.
    destino.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for nome in z.namelist():
            if not (destino / nome).resolve().is_relative_to(destino.resolve()):
                raise ValueError(f"caminho inseguro no pacote: {nome}")
        z.extractall(destino)


if EM_COLAB:
    os.environ["HF_HOME"] = HF_HOME_DIR  # antes de qualquer import de transformers/huggingface_hub
    from google.colab import drive

    drive.mount("/content/drive")
    repo = Path(DIR_PHYSICS_LLM_SR)
    if not (repo / ".git").is_dir():
        subprocess.run(["git", "clone", "--quiet", PHYSICS_LLM_SR_URL, str(repo)], check=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    if head != PHYSICS_LLM_SR_COMMIT:
        subprocess.run(["git", "-C", str(repo), "fetch", "--quiet", "origin"], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", PHYSICS_LLM_SR_COMMIT], check=True)
    RAIZ = Path(DIR_TRABALHO)
    if not (RAIZ / "src" / "otft_sr").is_dir():
        zips = ([Path(PACOTE_ZIP)] if PACOTE_ZIP else
                sorted(Path(DIR_DRIVE).glob("otft-sr-*.zip"), key=lambda p: p.stat().st_mtime))
        if not zips:
            raise FileNotFoundError(
                f"nenhum otft-sr-*.zip em {DIR_DRIVE}: gere com scripts/empacotar_para_colab.py "
                "e copie o zip para MyDrive/otft-sr/")
        extrair_pacote(zips[-1], RAIZ)
        print("pacote extraído:", zips[-1].name)
    os.environ["PHYSICS_LLM_SR_REPO_DIR"] = str(repo)
    CAMINHOS = [repo / "src", RAIZ / "src"]
    DIR_COPIA_DRIVE = Path(DIR_DRIVE)
else:
    RAIZ = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
    CAMINHOS = [RAIZ / "src"]
    DIR_COPIA_DRIVE = None
for c in CAMINHOS:
    if str(c) not in sys.path:
        sys.path.insert(0, str(c))
arq_versao = RAIZ / "VERSAO_PACOTE.json"
VERSAO_PACOTE = json.loads(arq_versao.read_text()) if arq_versao.is_file() else {"origem": "checkout local"}

from otft_sr.provedores_llm import relatorio_ambiente

amb = relatorio_ambiente()
print(f"Colab: {EM_COLAB} | raiz: {RAIZ} | pacote: {VERSAO_PACOTE}")
print(f"physics-llm-sr: {PHYSICS_LLM_SR_COMMIT[:12] if EM_COLAB else 'instalação local'}")
print(f"CPU: {amb['n_cpus']} núcleos, RAM {amb['memoria_ram_gb']} GB | GPU: {amb['gpus'] or 'nenhuma'}")
""")

# ============================================================================ 3
md(r"""
## 3. Dependências com versões fixadas

**Finalidade:** garantir as versões de `requirements-colab.txt`. **Método:** compara a versão
instalada com a fixada e instala só o que diverge (no Colab); `pysr`/`juliacall` só com
`USAR_PYSR = True`; o `torch` da imagem do Colab é mantido salvo `FORCAR_TORCH_FIXADO = True`
(a versão efetiva fica registrada em `environment.json`). Fora do Colab nada é instalado.
**Resultado esperado:** tabela de versões e diagnóstico de PyTorch/CUDA.
""")

code(r"""
import importlib.metadata as metadata

import pandas as pd
from IPython.display import Markdown, display


def ler_requisitos(arquivo: Path) -> dict[str, tuple[str, bool]]:
    # "pacote==versão  # pysr" -> {pacote: (versão, só com PySR)}
    req = {}
    for linha in arquivo.read_text(encoding="utf-8").splitlines():
        spec, _, comentario = linha.partition("#")
        if spec.strip():
            nome, versao = spec.strip().split("==")
            req[nome.strip()] = (versao.strip(), "pysr" in comentario)
    return req


def versao_instalada(nome: str) -> str | None:
    try:
        return metadata.version(nome)
    except metadata.PackageNotFoundError:
        return None


REQUISITOS = ler_requisitos(RAIZ / "requirements-colab.txt")
situacao, instalar = [], []
for nome, (fixada, so_pysr) in REQUISITOS.items():
    atual = versao_instalada(nome)
    if so_pysr and not USAR_PYSR:
        acao = "não necessário (USAR_PYSR = False)"
    elif atual == fixada:
        acao = "ok"
    elif nome == "torch" and atual is not None and not FORCAR_TORCH_FIXADO:
        acao = "mantido (versão da imagem do Colab)"
    else:
        acao = "instalar" if EM_COLAB else "divergente (local: nada é instalado)"
        if EM_COLAB:
            instalar.append(f"{nome}=={fixada}")
    situacao.append({"pacote": nome, "fixada": fixada, "instalada": atual, "ação": acao})
display(pd.DataFrame(situacao).set_index("pacote"))

if instalar:
    ja_importados = [m for m in ("torch", "transformers", "accelerate", "bitsandbytes") if m in sys.modules]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *instalar], check=True)
    if ja_importados:
        print(f"ATENÇÃO: {ja_importados} já estavam importados; reinicie a sessão e execute de novo a partir da seção 2.")

PLOTLY_DISPONIVEL = importlib.util.find_spec("plotly") is not None
if importlib.util.find_spec("torch") is None:
    print("PyTorch ausente: só os backends simulado, Ollama e API funcionam.")
else:
    import torch

    if torch.cuda.is_available():
        print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}: {torch.cuda.get_device_name(0)}")
    else:
        print(f"PyTorch {torch.__version__} SEM CUDA: no Colab, Ambiente de execução > Alterar o tipo > GPU; "
              "modelos Hugging Face exigem GPU (bitsandbytes).")
print(f"plotly disponível: {PLOTLY_DISPONIVEL}")
AMBIENTE = relatorio_ambiente()
""")

# ============================================================================ 4
md(r"""
## 4. Configuração única

**Finalidade:** declarar, em um só lugar, todos os fatores da campanha. **Método:**
`ConfiguracaoExperimento` valida os valores na construção; a quantização de cada modelo
Hugging Face vem de `recomendar_quantizacao` (memória da GPU); `EXECUCAO_DE_VERIFICACAO = True`
troca tudo pela configuração mínima (N3, 1 semente, DEAP mínimo, M2 com 2 candidatos) e um
backend de verificação. **Resultado esperado:** configuração efetiva, plano de ensaios e o
diretório `results/<id>/` (fixo pelo nome: é ele que permite a retomada).
""")

code(r"""
# ---- CONFIGURAÇÃO ÚNICA DO EXPERIMENTO (edite só este bloco) --------------------------
EXECUCAO_DE_VERIFICACAO = True     # True: configuração mínima + backend de verificação (não constitui resultado)
BACKEND_VERIFICACAO = "simulado"   # "simulado" | "ollama" (mistral:7b-instruct local; fora do Colab)
ARQUIVOS_LINHA_DE_BASE: list[str] = []  # padrões glob de resultados M1/A existentes (seção 7)

from otft_sr.campanha import MISTRAL_7B, N_PARAMETROS_B, PHI3_MINI, QWEN25_3B, ConfiguracaoExperimento
from otft_sr.provedores_llm import ConfiguracaoLLM  # noqa: F401 (modelos de API, abaixo)

CONFIGURACAO_EXPERIMENTO = ConfiguracaoExperimento(
    nome_experimento="otft_multi_llm_v1",
    modelos=(
        MISTRAL_7B,   # mistralai/Mistral-7B-Instruct-v0.1 @ ec5deb64 (revisão do physics-llm-sr)
        QWEN25_3B,    # Qwen/Qwen2.5-3B-Instruct @ aa8e7253
        PHI3_MINI,    # microsoft/Phi-3-mini-4k-instruct @ f39ac1d2
        # APIs (opcional; chave em Colab Secrets, nunca no código; determinismo não garantido):
        # ConfiguracaoLLM("openai", "NOME_DO_MODELO"),
        # ConfiguracaoLLM("anthropic", "NOME_DO_MODELO"),
        # ConfiguracaoLLM("google", "NOME_DO_MODELO"),
    ),
    n_parametros_b=N_PARAMETROS_B,      # bilhões de parâmetros (escolha da quantização)
    modos=("M1", "M2"),
    condicoes_m1=("A", "B", "C"),
    condicoes_m2=("G0", "G1", "G2"),
    niveis=("N1", "N2", "N3", "N4"),
    niveis_llm=None,                    # None = os mesmos níveis nos ensaios com LLM
    algoritmos_rs=("deap", "gplearn"),  # "pysr" exige USAR_PYSR = True (Julia)
    algoritmos_rs_llm=("deap",),        # algoritmos de RS das condições B/C
    configuracao_busca="calibrada",     # "calibrada" (best_config) | "verificacao" | dict
    configuracao_busca_llm=None,        # None = a mesma de A (esforço de busca equivalente)
    sementes=(0, 1, 2),
    ruidos=(0.0,),                      # proposta §4.5: (0.0, 0.01, 0.05, 0.10)
    n_curvas=20, n_pontos=50,
    n_candidatos=5, n_iteracoes_refinamento=3,
    formulacao_perda="linear",          # "linear" (MSE normalizado do artigo) | "log10" (H-OTFT-19)
)
# ---------------------------------------------------------------------------------------
import dataclasses

from otft_sr import campanha
from otft_sr.contexto import ollama_disponivel
from otft_sr.provedores_llm import RegistroUso

if EXECUCAO_DE_VERIFICACAO:
    CFG = campanha.configuracao_de_verificacao(CONFIGURACAO_EXPERIMENTO, BACKEND_VERIFICACAO)
    if BACKEND_VERIFICACAO == "ollama" and not ollama_disponivel():
        raise RuntimeError("Ollama indisponível em localhost:11434 ou sem mistral:7b-instruct")
    display(Markdown("**EXECUÇÃO DE VERIFICAÇÃO** — valida a integridade do pipeline; não constitui resultado."))
else:
    CFG = CONFIGURACAO_EXPERIMENTO
    if "pysr" in CFG.algoritmos_rs and not USAR_PYSR:
        raise ValueError('"pysr" em algoritmos_rs exige USAR_PYSR = True (seção 2) e a seção 3 executada')
    if any(m.provedor == "huggingface" for m in CFG.modelos):
        CFG, tabela_quantizacao = campanha.aplicar_quantizacao(CFG, AMBIENTE["memoria_gpu_gb"])
        display(tabela_quantizacao)

ROTULO_BUSCA = "verificacao" if CFG.execucao_de_verificacao else (
    CFG.configuracao_busca if isinstance(CFG.configuracao_busca, str) else "definida_pelo_usuario")
ROTULO_BUSCA_LLM = "verificacao" if CFG.execucao_de_verificacao else (
    CFG.busca_llm_efetiva if isinstance(CFG.busca_llm_efetiva, str) else "definida_pelo_usuario")
REG_USO = RegistroUso()  # chamadas, tokens e tempo por modelo (identidade completa)
DIR_RESULTADOS = Path(os.environ.get("OTFT_SR_DIR_RESULTADOS", RAIZ / "results"))
DIR_CACHE_LLM = Path(os.environ.get("OTFT_SR_DIR_CACHE_LLM", RAIZ / ".cache" / "llm"))
DRIVE_RESULTADOS = DIR_COPIA_DRIVE / "results" / CFG.id_experimento if DIR_COPIA_DRIVE else None
DRIVE_CACHE = DIR_COPIA_DRIVE / "cache_llm" if DIR_COPIA_DRIVE else None

# Retomada: o que já estiver no Drive volta para o disco local antes de abrir o registro.
if DRIVE_RESULTADOS is not None and DRIVE_RESULTADOS.is_dir():
    campanha.sincronizar(DRIVE_RESULTADOS, DIR_RESULTADOS / CFG.id_experimento)
if DRIVE_CACHE is not None and DRIVE_CACHE.is_dir():
    campanha.sincronizar(DRIVE_CACHE, DIR_CACHE_LLM)
REGISTRO = campanha.abrir_registro(DIR_RESULTADOS, CFG, extras={
    "physics_llm_sr_commit": PHYSICS_LLM_SR_COMMIT if EM_COLAB else "instalação local",
    "pacote_otft_sr": VERSAO_PACOTE})


def sincronizar_drive() -> None:
    # Copia resultados e cache do LLM para o Drive (no-op fora do Colab).
    campanha.sincronizar(REGISTRO.diretorio, DRIVE_RESULTADOS)
    campanha.sincronizar(DIR_CACHE_LLM, DRIVE_CACHE)


def mostrar_figura(nome: str, construtor, *args, **kwargs) -> None:
    # Constrói, rotula (verificação), grava em figures/ (HTML + JSON) e exibe.
    if not PLOTLY_DISPONIVEL:
        print(f"plotly indisponível: figura {nome} não gerada")
        return
    from otft_sr.visualizacao import salvar_figura

    fig = construtor(*args, **kwargs)
    if CFG.execucao_de_verificacao:
        fig.update_layout(title=f"{fig.layout.title.text or nome} — EXECUÇÃO DE VERIFICAÇÃO")
    salvar_figura(fig, REGISTRO.diretorio / "figures" / nome)
    fig.show()


PLANO = campanha.plano_de_ensaios(CFG)
print(f"experimento: {CFG.id_experimento} -> {REGISTRO.diretorio}")
print(f"modelos (provedor/modelo@revisão[quantização]): {[m.rotulo for m in CFG.modelos]}")
print(f"formulação da perda (M1): {CFG.formulacao_perda}; configuração de busca: {ROTULO_BUSCA} / LLM: {ROTULO_BUSCA_LLM}")
concluidos = PLANO.drop(columns="etapa").apply(lambda r: REGISTRO.concluido(r.to_dict()), axis=1)
display(PLANO.assign(concluido=concluidos).groupby("etapa")["concluido"].agg(ensaios="size", concluidos="sum"))
""")

# ============================================================================ 5
md(r"""
## 5. Dados

**Finalidade:** conferir os conjuntos de cada nível. **Método:** grade por curvas na região do
nível, ruído multiplicativo, partição treino/validação/teste por curva (60/20/20 %,
H-OTFT-06). **Resultado esperado:** partições disjuntas por curva; tabela por nível.
""")

code(r"""
import numpy as np

from otft_sr.dados import construir_datasets
from otft_sr.niveis import nivel

linhas = []
for nid in CFG.niveis:
    n = nivel(nid)
    ds = construir_datasets(n, max(CFG.ruidos), CFG.sementes[0], CFG.n_curvas, CFG.n_pontos)
    ids = {k: set(np.unique(d.simulation_ids)) for k, d in ds.items()}
    assert not (ids["treino"] & ids["teste"] or ids["treino"] & ids["validacao"] or ids["validacao"] & ids["teste"])
    linhas.append({"nível": f"{nid} — {n.nome}", "região": n.regiao.descricao, "I_D": n.unidades["I_D"],
                   **{f"curvas {k}": len(v) for k, v in ids.items()},
                   **{f"pontos {k}": d.y_observed.size for k, d in ds.items()},
                   "σ": max(CFG.ruidos), "referência": n.expressao})
display(pd.DataFrame(linhas).set_index("nível"))
""")

# ============================================================================ 6
md(r"""
## 6. Contexto físico

**Finalidade:** mostrar quanto cada nível se afasta do OVSED completo e da Org1 medida.
**Método:** curva do nível só na sua região, OVSED na mesma faixa e pontos medidos dentro
dela. **Resultado esperado:** níveis sobre o OVSED; desvios documentados em H-OTFT-12…18.
""")

code(r"""
from otft_sr.visualizacao import figura_contexto_fisico

mostrar_figura("contexto_fisico", figura_contexto_fisico, CFG.niveis)
""")

# ============================================================================ 7
md(r"""
## 7. Linha de base (condição A, M1 sem LLM)

**Finalidade:** referência pareada para B/C. **Método:** `executar_m1` com $w_3 = 0$ em todos os
níveis × algoritmos de RS × ruídos × sementes, retomável; resultados antigos (notebook 01)
podem ser importados por `ARQUIVOS_LINHA_DE_BASE`, mantendo só ensaios cuja referência é a
atual. **Resultado esperado:** uma linha por ensaio em `metrics.csv`; importados só exibidos.
""")

code(r"""
import itertools

from otft_sr.estatistica import taxa_recuperacao
from otft_sr.experimento import MODELO_SEM_LLM, executar_m1

if "M1" in CFG.modos and "A" in CFG.condicoes_m1:
    for nid, alg, r, s in itertools.product(CFG.niveis, CFG.algoritmos_rs, CFG.ruidos, CFG.sementes):
        campos = campanha.campos_m1(nid, "A", alg, MODELO_SEM_LLM, r, s, CFG.formulacao_perda)
        linha = REGISTRO.executar(campos, lambda: {
            **executar_m1(nivel(nid), alg, "A", r, s, CFG.configuracao_busca, n_curvas=CFG.n_curvas,
                          n_pontos=CFG.n_pontos, formulacao_perda=CFG.formulacao_perda),
            "configuracao_busca": ROTULO_BUSCA})
        print(f"A {nid} {alg:7s} σ={r} s={s} [{linha['status']}] {str(linha.get('expressao'))[:60]} "
              f"| RMSE log10 = {linha.get('rmse_log10', float('nan')):.3g} | {linha['classe']}")
    sincronizar_drive()

LINHA_DE_BASE_IMPORTADA, relatorio_importacao = campanha.importar_linha_de_base(ARQUIVOS_LINHA_DE_BASE)
if len(relatorio_importacao):
    display(relatorio_importacao)
if len(LINHA_DE_BASE_IMPORTADA):
    display(taxa_recuperacao(LINHA_DE_BASE_IMPORTADA, grupos=("nivel", "algoritmo", "configuracao_busca")))
""")

# ============================================================================ 8
md(r"""
## 8. Prompts

**Finalidade:** tornar auditável o que cada LLM recebe. **Método:** renderiza os prompts do M1
(B, C; variante E, com uma expressão neutra) e do M2 (G0, G1, G2) para o primeiro nível com
LLM, registra os hashes SHA-256 e verifica que nenhum contém a equação-alvo nem suas
constantes. **Resultado esperado:** tabela de componentes com hashes; verificação aprovada.
""")

code(r"""
import hashlib

from physics_llm_sr.symbolic.expression import parse_infix

from otft_sr.contexto import VARIANTE_PROMPT, carregar_biblioteca, contexto_fisico
from otft_sr.gerador import carregar_templates, hash_templates, montar_prompt
from otft_sr.niveis import FEATURES


def sha(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]


nid = CFG.niveis_llm_efetivos[0]
n_lin = nivel(nid)
n_form = campanha.nivel_da_formulacao(nid, CFG.formulacao_perda)
EXPRESSAO_NEUTRA = "V_GS * V_DS"  # nunca a equação-alvo
PROMPTS = {}
for cond in ("B", "C"):
    r = carregar_biblioteca(cond).render(parse_infix(EXPRESSAO_NEUTRA, FEATURES), contexto_fisico(n_form), VARIANTE_PROMPT)
    PROMPTS[("M1", cond, "sistema")], PROMPTS[("M1", cond, "usuario")] = r.system, r.user
ds_p = construir_datasets(n_lin, 0.0, CFG.sementes[0], CFG.n_curvas, CFG.n_pontos)
for g in ("G0", "G1", "G2"):
    s_, u_ = montar_prompt(n_lin, g, ds_p["treino"], 1, CFG.n_candidatos)
    PROMPTS[("M2", g, "sistema")], PROMPTS[("M2", g, "usuario")] = s_, u_
proibidos = [n_lin.expressao, n_form.expressao, repr(n_lin.constantes.get("K", "@@")),
             repr(n_lin.constantes.get("I_0", "@@"))]
vazamentos = [k for k, t in PROMPTS.items() if any(p in t for p in proibidos)]
assert not vazamentos, f"equação-alvo ou constante no prompt: {vazamentos}"
display(pd.DataFrame([{"modo": k[0], "condição": k[1], "componente": k[2], "sha256[:16]": sha(t),
                       "caracteres": len(t)} for k, t in PROMPTS.items()]))
print("hash do conjunto de templates do M2:", hash_templates(carregar_templates())["conjunto"][:16])
print("\n--- M1, condição C, usuário ---\n" + PROMPTS[("M1", "C", "usuario")])
print("\n--- M2, G1, usuário ---\n" + PROMPTS[("M2", "G1", "usuario")])
""")

# ============================================================================ 9
md(r"""
## 9. Auditoria do avaliador (M1)

**Finalidade:** pré-requisito para interpretar B/C. **Método:** para cada modelo, condição
(B, C) e nível, uma chamada sem cache por expressão: referência, forma equivalente, formas
erradas plausíveis, constante e absurda; $c = 1 - (c_1 + c_2 + c_3)/3$. Discrimina se
$c$(referência) < $c$ de todas as formas erradas (empate é falha; o Mistral já atribuiu o mesmo
$c$ à referência e a uma forma errada). **Resultado esperado:** tabela de discriminação por
modelo; modelos que não discriminam têm B/C sem interpretação física.
""")

code(r"""
from otft_sr.auditoria import auditar_avaliador, resumo_discriminacao
from otft_sr.provedores_llm import criar_avaliador_m1, usar_provedor


def resolver_identidade(provedor) -> None:
    # Fixa digest (Ollama) ou formato do prompt (Hugging Face): entram na identidade do cache.
    resolver = getattr(provedor, "resolve_identity", None)
    if callable(resolver):
        resolver()


ARQ_AUDITORIA = REGISTRO.diretorio / "auditoria_avaliador.csv"
AUDITORIA = pd.read_csv(ARQ_AUDITORIA) if ARQ_AUDITORIA.is_file() else pd.DataFrame()
feitos = set(zip(AUDITORIA["modelo"], AUDITORIA["condicao"], AUDITORIA["nivel"])) if len(AUDITORIA) else set()
for cfg_llm in CFG.modelos if CFG.condicoes_m1_llm else ():
    pendentes = [(c, nid) for c in CFG.condicoes_m1_llm for nid in CFG.niveis_llm_efetivos
                 if (cfg_llm.rotulo, c, nid) not in feitos]
    if not pendentes:
        continue
    with usar_provedor(cfg_llm, registro=REG_USO) as prov:
        resolver_identidade(prov)
        for c, nid in pendentes:
            av = criar_avaliador_m1(cfg_llm, c, provedor=prov, dir_cache=None, registro=REG_USO)
            df_a = auditar_avaliador(av, nivel(nid), CFG.formulacao_perda).assign(modelo=cfg_llm.rotulo, condicao=c)
            AUDITORIA = pd.concat([AUDITORIA, df_a], ignore_index=True)
            AUDITORIA.to_csv(ARQ_AUDITORIA, index=False)
    sincronizar_drive()

RESUMO_AUDITORIA = pd.DataFrame([
    {"modelo": m, "condicao": c, "nivel": nid, **resumo_discriminacao(g)}
    for (m, c, nid), g in AUDITORIA.groupby(["modelo", "condicao", "nivel"], sort=True)
]) if len(AUDITORIA) else pd.DataFrame()
if len(AUDITORIA):
    display(RESUMO_AUDITORIA)
    display(AUDITORIA.pivot_table(index=["modelo", "condicao", "nivel"], columns="rotulo", values="c", sort=False))
""")

# ============================================================================ 10
md(r"""
## 10. Geração (M2: G0, G1, G2)

**Finalidade:** o LLM como gerador de equações. **Método:** por modelo (um de cada vez, GPU
liberada ao final), `executar_geracao`: interpretação segura do JSON, limites de
complexidade, ajuste dos parâmetros no treino (resíduo em log10 I_D), checagens físicas,
refinamento em G2 e seleção só pela validação. **Resultado esperado:** uma linha por ensaio
e o histórico de candidatos em `candidates.csv`.
""")

code(r"""
from otft_sr.gerador import executar_geracao

if "M2" in CFG.modos:
    for cfg_llm in CFG.modelos:
        cfg_m2 = dataclasses.replace(cfg_llm, max_novos_tokens=max(cfg_llm.max_novos_tokens, CFG.max_novos_tokens_m2))
        ensaios = [(nid, g, r, s) for nid, g, r, s in itertools.product(
            CFG.niveis_llm_efetivos, CFG.condicoes_m2, CFG.ruidos, CFG.sementes)
            if not REGISTRO.concluido(campanha.campos_m2(nid, g, cfg_llm.rotulo, r, s))]
        if not ensaios:
            print(f"M2 {cfg_llm.rotulo}: todos os ensaios já registrados")
            continue
        with usar_provedor(cfg_m2, registro=REG_USO) as prov:
            resolver_identidade(prov)
            for nid, g, r, s in ensaios:
                linha = REGISTRO.executar(campanha.campos_m2(nid, g, cfg_llm.rotulo, r, s), lambda: executar_geracao(
                    nivel(nid), g, prov, s, CFG.n_candidatos, CFG.n_iteracoes_refinamento,
                    ruido=r, n_curvas=CFG.n_curvas, n_pontos=CFG.n_pontos))
                print(f"M2 {cfg_llm.rotulo} {nid} {g} s={s} [{linha['status']}] {str(linha.get('expressao'))[:60]} "
                      f"| válidos {linha.get('n_candidatos_validos')}/{linha.get('n_candidatos_avaliados')} "
                      f"| RMSE log10 = {linha.get('rmse_log10', float('nan')):.3g} | {linha['tempo_s']:.0f} s")
        sincronizar_drive()
""")

# ============================================================================ 11
md(r"""
## 11. M1 com LLM (condições B e C)

**Finalidade:** o LLM como avaliador na perda (método do artigo). **Método:** por modelo, em
sequência e com a GPU liberada entre modelos (`usar_provedor`), `executar_m1` com o mesmo
provedor para B e C e cache JSONL por condição e modelo. **Resultado esperado:** ensaios
pareados com a condição A (mesmos dados, mesma semente, mesma configuração de busca).
""")

code(r"""
for cfg_llm in CFG.modelos if CFG.condicoes_m1_llm else ():
    ensaios = [(c, nid, alg, r, s) for c, nid, alg, r, s in itertools.product(
        CFG.condicoes_m1_llm, CFG.niveis_llm_efetivos, CFG.algoritmos_rs_llm_efetivos, CFG.ruidos, CFG.sementes)
        if not REGISTRO.concluido(campanha.campos_m1(nid, c, alg, cfg_llm.rotulo, r, s, CFG.formulacao_perda))]
    if not ensaios:
        print(f"M1 {cfg_llm.rotulo}: todos os ensaios já registrados")
        continue
    with usar_provedor(cfg_llm, registro=REG_USO) as prov:
        resolver_identidade(prov)
        for c, nid, alg, r, s in ensaios:
            campos = campanha.campos_m1(nid, c, alg, cfg_llm.rotulo, r, s, CFG.formulacao_perda)
            linha = REGISTRO.executar(campos, lambda: {
                **executar_m1(nivel(nid), alg, c, r, s, CFG.busca_llm_efetiva, cfg_llm=cfg_llm, provedor=prov,
                              registro_uso=REG_USO, dir_cache=DIR_CACHE_LLM, n_curvas=CFG.n_curvas,
                              n_pontos=CFG.n_pontos, formulacao_perda=CFG.formulacao_perda),
                "configuracao_busca": ROTULO_BUSCA_LLM})
            print(f"{c} {cfg_llm.rotulo} {nid} {alg} s={s} [{linha['status']}] {str(linha.get('expressao'))[:50]} "
                  f"| chamadas LLM {linha['chamadas_llm']} ({linha['tempo_llm_s']:.0f} s) | {linha['tempo_s']:.0f} s")
    sincronizar_drive()
""")

# ============================================================================ 12
md(r"""
## 12. Avaliação

**Finalidade:** mesmas métricas para M1 e M2. **Método:** relê `results/<id>/metrics.csv`
(independe das seções 7–11 terem rodado nesta sessão) e confere o controle positivo na
formulação da perda. Métricas de teste (I_D limpo): RMSE de log10 I_D, $1 - R^2$ (no alvo
da formulação) e $1 - R^2$ em I_D, nós, distâncias de árvore e classe de recuperação.
**Resultado esperado:** tabela de ensaios; controle positivo com $1 - R^2 \approx 0$.
""")

code(r"""
from otft_sr.experimento import controle_positivo
from otft_sr.persistencia import carregar_resultados

DF = carregar_resultados(REGISTRO.diretorio)
if CFG.execucao_de_verificacao:
    display(Markdown("**EXECUÇÃO DE VERIFICAÇÃO** — a tabela demonstra o formato; não constitui resultado."))
COLUNAS = ["nivel", "modo", "condicao", "algoritmo", "modelo", "ruido", "semente", "formulacao_perda",
           "status", "expressao", "n_nos", "rmse_log10", "one_minus_r2", "one_minus_r2_I_D",
           "tree_distance", "tree_distance_algebraic", "classe", "tempo_s", "tempo_llm_s", "chamadas_llm"]
display(DF.reindex(columns=COLUNAS))
CONTROLE = pd.DataFrame([controle_positivo(nivel(nid), formulacao_perda=CFG.formulacao_perda) for nid in CFG.niveis])
display(CONTROLE[["nivel", "formulacao_perda", "expressao", "n_nos", "one_minus_r2", "rmse_log10", "tree_distance"]])
assert (CONTROLE["one_minus_r2"] < 1e-12).all() and (CONTROLE["tree_distance"] == 0).all()
""")

# ============================================================================ 13
md(r"""
## 13. Benchmark

**Finalidade:** quantificar o efeito do LLM. **Método:** recuperação por
nível × modo × condição × modelo com IC 95 % de Wilson; diferenças pareadas por semente
(Wilcoxon no RMSE de log10 I_D e McNemar exato na recuperação, sobre o mesmo conjunto de
pares); qualidade (média e IC t). Ensaios com `erro_execucao` ficam fora de todos os
denominadores e cada tabela informa `n_excluido_erro`; não convergência conta como não
recuperação;
custo (chamadas, tokens, tempo, razão tempo de LLM / tempo de RS contra 0,4 do artigo).
Comparações rotuladas **equivalente** (mesmo esforço: mesma configuração de busca ou mesmo
número de candidatos) ou **exploratória** (esforço diferente: G2, M1 × M2).
**Resultado esperado:** tabelas `TAXA`, `COMPARACOES`, `QUALIDADE`, `CUSTO`.
""")

code(r"""
from otft_sr import estatistica as est

GRUPOS = ("nivel", "modo", "condicao", "modelo", "algoritmo")
TAXA = est.taxa_recuperacao(DF, grupos=GRUPOS)
display(Markdown("**Recuperação simbólica (IC 95 % de Wilson)**"))
display(TAXA)


def comparar(rotulo: str, nid: str, filtro_a: dict, filtro_b: dict, tipo: str) -> dict | None:
    fa, fb = {**filtro_a, "nivel": nid}, {**filtro_b, "nivel": nid}
    if est._filtrar(DF, fa).empty or est._filtrar(DF, fb).empty:
        return None
    r = est.comparacao_pareada(DF, "rmse_log10", fa, fb)
    mc = r.get("mcnemar_recuperacao", {})
    nan = float("nan")
    return {"comparacao": rotulo, "nivel": nid, "tipo": tipo, "n_pares": r["n_pares"],
            "n_excluido_erro": r["n_excluido_erro"], "n_excluido_metrica": r["n_excluido_metrica"],
            "diferenca_media": r["diferenca_media"], "p_valor": nan if r["p_valor"] is None else r["p_valor"],
            "p_mcnemar": mc.get("p_valor", nan), "nota": r["nota"]}


comps = []
for nid in sorted(DF["nivel"].dropna().unique()):
    for m in CFG.modelos:
        for alg in CFG.algoritmos_rs_llm_efetivos:
            base = {"modo": "M1", "algoritmo": alg}
            for c in CFG.condicoes_m1_llm:
                comps.append(comparar(f"M1 A × {c} ({m.rotulo}, {alg})", nid, {**base, "condicao": "A"},
                                      {**base, "condicao": c, "modelo": m.rotulo}, "equivalente"))
            comps.append(comparar(f"M1 B × C ({m.rotulo}, {alg})", nid, {**base, "condicao": "B", "modelo": m.rotulo},
                                  {**base, "condicao": "C", "modelo": m.rotulo}, "equivalente"))
        g = {"modo": "M2", "modelo": m.rotulo}
        comps.append(comparar(f"M2 G0 × G1 ({m.rotulo})", nid, {**g, "condicao": "G0"}, {**g, "condicao": "G1"}, "equivalente"))
        comps.append(comparar(f"M2 G1 × G2 ({m.rotulo})", nid, {**g, "condicao": "G1"}, {**g, "condicao": "G2"}, "exploratória"))
COMPARACOES = pd.DataFrame([c for c in comps if c is not None])
display(Markdown("**Comparações pareadas por semente** (diferença = a − b no RMSE de log10 I_D; positiva = b melhor)"))
display(COMPARACOES)

QUALIDADE = est.agregar(DF, grupos=GRUPOS, metricas=("rmse_log10", "one_minus_r2", "one_minus_r2_I_D", "n_nos"))
display(Markdown("**Qualidade (média e IC 95 % t)**"))
display(QUALIDADE.pivot_table(index=list(GRUPOS), columns="metrica", values="media", sort=False))

CUSTO = est.tabela_custo_qualidade(DF, grupos=GRUPOS).assign(tipo_comparacao="exploratória (custo efetivo)")
ESFORCO = est.tabela_esforco_equivalente(DF, grupos=("nivel", "modo", "condicao", "modelo")).assign(
    tipo_comparacao="equivalente (mesmo número de candidatos)")
RAZAO_TEMPO = est.tabela_razao_tempo_llm(DF)
USO_LLM = pd.DataFrame.from_dict(REG_USO.como_dict(), orient="index")
display(Markdown("**Custo efetivo × qualidade**"))
display(CUSTO)
display(Markdown("**Esforço equivalente (M2)**"))
display(ESFORCO)
display(Markdown("**Tempo de LLM / tempo de RS (referência do artigo: 0,4)**"))
display(RAZAO_TEMPO)
display(Markdown("**Uso por modelo nesta sessão (chamadas, tokens, segundos)**"))
display(USO_LLM)
""")

# ============================================================================ 14
md(r"""
## 14. Visualizações

**Finalidade:** inspeção visual das equações selecionadas. **Método:** Plotly
(`graph_objects`); previsões avaliadas só nos pontos de teste (sem interpolar nem extrapolar);
figuras gravadas em `figures/` (HTML e JSON). Requer as seções 12 e 13. **Resultado
esperado:** observado × previsto, curvas reconstruídas, resíduos, Pareto erro × nós,
tempo × qualidade, evolução do erro em G2, comparação entre LLMs e barras com IC.
""")

code(r"""
from otft_sr import visualizacao as vz

ok = DF[DF["status"] == "ok"]
if len(ok):
    nid_fig = CFG.niveis_llm_efetivos[0] if (ok["nivel"] == CFG.niveis_llm_efetivos[0]).any() else ok["nivel"].iloc[0]
    sel = ok[(ok["nivel"] == nid_fig) & (ok["semente"] == CFG.sementes[0]) & (ok["ruido"] == CFG.ruidos[0])]
    PREVISOES = {vz._rotulo_linha(lin): vz.previsoes_teste(lin, CFG.n_curvas, CFG.n_pontos)
                 for lin in sel.to_dict("records")}
    melhor = sel.sort_values("rmse_log10").iloc[0].to_dict()
    mostrar_figura(f"observado_previsto_{nid_fig}", vz.figura_observado_previsto, PREVISOES)
    mostrar_figura(f"curvas_reconstruidas_{nid_fig}", vz.figura_curvas_reconstruidas,
                   vz.previsoes_teste(melhor, CFG.n_curvas, CFG.n_pontos))
    mostrar_figura(f"residuos_{nid_fig}", vz.figura_residuos, PREVISOES)
mostrar_figura("pareto_erro_nos", vz.figura_pareto, DF)
mostrar_figura("tempo_qualidade", vz.figura_tempo_qualidade, DF)
arq_cand = REGISTRO.diretorio / "candidates.csv"
if arq_cand.is_file():
    cand = pd.read_csv(arq_cand)
    mostrar_figura("evolucao_refinamento_G2", vz.figura_evolucao_refinamento,
                   cand[cand["chave_ensaio"].str.contains("condicao=G2", regex=False)])
mostrar_figura("comparacao_llms", vz.figura_comparacao_llms, DF[DF["modelo"] != MODELO_SEM_LLM])
mostrar_figura("recuperacao_ic", vz.figura_barras_ic, TAXA)
""")

# ============================================================================ 15
md(r"""
## 15. Exportação

**Finalidade:** persistir tudo o que sustenta as tabelas. **Método:** `finalizar()` grava
`metrics_summary.csv`; as tabelas do benchmark e o uso por modelo vão para `results/<id>/`;
no Colab o diretório e o cache do LLM são copiados para `MyDrive/otft-sr/`. A reimportação
confere a integridade. **Resultado esperado:** lista de arquivos e reimportação idêntica.
""")

code(r"""
RESUMO = REGISTRO.finalizar()
tabelas = {"benchmark_recuperacao.csv": TAXA, "benchmark_comparacoes.csv": COMPARACOES,
           "benchmark_qualidade.csv": QUALIDADE, "benchmark_custo.csv": CUSTO,
           "benchmark_esforco_equivalente.csv": ESFORCO, "benchmark_razao_tempo_llm.csv": RAZAO_TEMPO,
           "auditoria_resumo.csv": RESUMO_AUDITORIA}
for nome, tab in tabelas.items():
    tab.to_csv(REGISTRO.diretorio / nome, index=False)
(REGISTRO.diretorio / "uso_llm.json").write_text(json.dumps(REG_USO.como_dict(), indent=1))
REGISTRO.log("exportacao", arquivos=sorted(tabelas))
sincronizar_drive()

reimportado = carregar_resultados(REGISTRO.diretorio)
assert len(reimportado) == len(DF) and set(reimportado["chave_ensaio"]) == set(DF["chave_ensaio"])
print("diretório:", REGISTRO.diretorio, "| cópia no Drive:", DRIVE_RESULTADOS or "não (execução local)")
for p in sorted(REGISTRO.diretorio.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(REGISTRO.diretorio)} ({p.stat().st_size} B)")
""")

# ============================================================================ 16
md(r"""
## 16. Conclusões e limitações

**Finalidade:** síntese a partir das tabelas desta execução (seções 9 e 13). Em execução de
verificação o texto declara que não há resultado.
""")

code(r"""
display(Markdown(campanha.texto_conclusoes(TAXA, RESUMO_AUDITORIA, COMPARACOES, CFG)))
""")

md(r"""
**Limitações (independentes da execução).**

- Dados sintéticos gerados pelas equações dos níveis (H-OTFT-04); a Org1 medida entra só como
  validação de contexto (seção 6), com um único dispositivo.
- N3 usa $F_{sat}$ constante (erro de 0,14 década contra o OVSED, H-OTFT-13) e N4 tem faixa
  dinâmica curta acima da fuga (0,59 década, H-OTFT-17); o $S$ do N4 é um swing efetivo.
- M1: o termo $c$ só pesa $w_3/w_1 = 0{,}01$ (H-OTFT-07); sem discriminação na auditoria, B/C
  não informam a busca.
- M2 é exploratório: a seleção usa a validação, e G2 consome mais candidatos que G0/G1.
- Modelos de 3–7 B quantizados são ruidosos; APIs não garantem determinismo (respostas ficam no
  cache, não são presumidas reproduzíveis).
- Poucas sementes produzem IC largos; p-valores sem correção para comparações múltiplas.
""")


def construir() -> nbf.NotebookNode:
    """Monta o notebook a partir da lista ``celulas`` declarada acima."""
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["colab"] = {"provenance": [], "gpuType": "A100"}
    nb.metadata["accelerator"] = "GPU"
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
