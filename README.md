# otft-sr — regressão simbólica guiada por LLM para OTFTs

Adapta Taskin, Xie & Lazebnik (Sci Rep 2026), perda L = w1·e + w2·s + w3·c, a equações de
corrente de dreno de OTFTs (proposta `proposta_de_artigo___DL__Stefan_.pdf`). Equações-alvo:
níveis N1–N4 da proposta (Seção 5.1) ancorados no modelo OVSED com parâmetros da Org1 (TCC, `Model_OTFT`).
Condições experimentais: A = RS sem LLM, B = LLM com prompt genérico, C = LLM com prompt de
física de dispositivo.

| Id (= nível da proposta) | Nome científico | Equação | Região | Id antes do alinhamento |
|---|---|---|---|---|
| N1 | Saturação quadrática ideal | I_D = K (V_GS − V_Th)² | saturação, V_GS 25–50 V (controle canônico) | N1 |
| N2 | Regime linear (triodo) com termo quadrático em V_DS | I_D = K (2 (V_GS − V_Th) V_DS − V_DS²) | triodo (V_DS ≤ V_GS − V_Th) | N3 |
| N3 | Saturação com mobilidade em lei de potência | I_D = K (V_GS − V_th0)^l | saturação perto do limiar, V_GS 10,5–28 V, F_sat constante (H-OTFT-18) | N2 |
| N4 | Corrente sublimiar exponencial | I_D = I_leak + I_0 exp(ln10 V_GS / S) | sublimiar observável, V_GS 0–10 V, I_leak = 65 pA aditivo (pA; H-OTFT-17) | N4 |
| N5 | Saturação com resistência de contato | implícita com R_S | só documentado (H-OTFT-08) | N5 |

## Ambiente

Usa o venv do repositório validado `physics-llm-sr` (deap, gplearn, PySR 2.7.2/juliacall,
nbclient, ipykernel); nada é instalado aqui:

```bash
PY="/home/rsb6/Desktop/projeto cleber /physics-llm-sr/.venv/bin/python"
```

LLM opcional: Ollama local com `mistral:7b-instruct` (`curl -s localhost:11434/api/tags`).

## Execução

```bash
cd "/home/rsb6/Desktop/projeto cleber /otft-sr"
"$PY" -m pytest                              # testes (~1-2 min, sem Julia/LLM)
"$PY" notebooks/build_01_rs_otft.py          # gera notebooks/01_rs_otft.ipynb
# abrir o notebook com o kernel do venv; editar só o bloco de parâmetros da seção 0
```

`CONFIGURACAO_BUSCA = "verificacao"` verifica a integridade do pipeline e não constitui
resultado; `"calibrada"` usa o `best_config.yaml` da queda livre de cada algoritmo de RS.
Os testes de execução completa do notebook têm o marcador `lento`
(`"$PY" -m pytest -m lento`) e ficam fora da suíte padrão. O `.ipynb` versionado não
guarda saídas: é sempre regenerado pelo builder; resultados executados vão para `results/`.
As colunas exportadas são `algoritmo` e `configuracao_busca` (CSV anteriores à renomeação
usam outros nomes e não são lidos pelo notebook atual).

## Modo M2 (exploratório): LLM como gerador de equações

`otft_sr.gerador.executar_geracao(nivel, "G0"|"G1"|"G2", provedor, semente, n_candidatos,
n_iteracoes_refinamento)` devolve `(linha, candidatos)`; a linha segue `COLUNAS_M1` (as mesmas
colunas de `executar_condicao`) seguidas de `COLUNAS_M2`. G0 = variáveis, unidades e resumo
numérico do treino; G1 = G0 + contexto físico de OTFT (sem a equação-alvo nem constantes);
G2 = G1 + `k` rodadas de refinamento por linhagem com retorno de treino/validação e das
checagens físicas (`n_candidatos x (1 + k)` candidatos avaliados).

Decisões (pré-definidas, antes de qualquer execução real; registradas em H-OTFT-20):

- Interpretação: primeiro objeto JSON da resposta, schema pydantic (`equation`, `variables`,
  `parameters` obrigatórios), `parse_infix` (whitelist de AST, sem eval/exec/sympify),
  operações = as do PySR no M1, no máximo 30 nós, profundidade 8 e 6 parâmetros; até
  `max_retentativas` novas tentativas com o motivo da recusa.
- Ajuste: `least_squares` (TRF, `x_scale="jac"`) no TREINO com resíduo em log10|I_D|
  (proposta §4.4); 8 inícios determinísticos (chutes do LLM, todos 1, sorteios log-uniformes
  semeados por semente e hash da equação).
- Checagens (treino + grade da região): I_D finito e > 0, monotonia em V_GS, limiar
  (N1/N3: I_D ~ 0 no V_Th extrapolado de sqrt(I_D); N2: I_D = 0 em V_DS = 0; N4: I_D -> I_leak
  estimado no menor V_GS), saturação (N1/N3/N4: I_D independente de V_DS; N2: dI/dV_DS -> 0
  na constrição) e dimensional só qualitativa (informativa).
- Seleção só por VALIDAÇÃO: fisicamente válido primeiro, menor RMSE de log10 I_D (empate a
  1e-6 década), menos nós. Métricas de teste só do selecionado, por `metricas_expressao`.

Persistência e estatística comuns a M1/M2: `otft_sr.persistencia.RegistroExperimento` grava
`results/<id>/` (config, ambiente com commit git, métricas, resumo, candidatos, equações,
metadados de geração com hashes, log JSONL, figuras), pula ensaios já registrados com
`status = "ok"`, repete os de `status = "erro_execucao"` (falha de infraestrutura; não
convergência é `ok` com classe "não converge") e nunca grava segredos; `carregar_resultados`
lê também os CSV/JSON antigos do notebook 01 (`data/legado/colunas_notebook01.json`).
`otft_sr.estatistica` agrega (IC 95 % t), calcula recuperação com IC de Wilson, comparação
pareada por semente (Wilcoxon) e McNemar exato sobre o mesmo conjunto de pares, sempre sem os
erros de execução (cada tabela informa `n_excluido_erro`).

## Notebook 02 no Colab (multi-LLM, M1 e M2)

`notebooks/02_multi_llm_descoberta_otft_colab.ipynb` (gerado por
`notebooks/build_02_multi_llm_colab.py`) compara Mistral-7B-Instruct-v0.1, Qwen2.5-3B-Instruct e
Phi-3-mini-4k-instruct (revisões fixadas por SHA) nos modos M1 (avaliador, condições A/B/C) e
M2 (gerador, G0/G1/G2), com auditoria do avaliador, benchmark pareado por semente e figuras
Plotly. A configuração única (`otft_sr.campanha.ConfiguracaoExperimento`) fica na seção 4;
`formulacao_perda = "log10"` é opcional (H-OTFT-19).

1. Abra direto do GitHub (sempre a versão publicada mais recente):
   <https://colab.research.google.com/github/RodrigoSantosB/otft-sr/blob/main/notebooks/02_multi_llm_descoberta_otft_colab.ipynb>
2. Escolha GPU (A100 para os três modelos sem quantização; T4 usa 8 bits no Mistral).
3. A seção 2 clona/atualiza este repositório em `OTFT_SR_REF` (`"main"` = última versão; um hash
   de commit fixa a versão) e imprime o commit usado, gravado em `environment.json`. Cada nova
   sessão do Colab puxa as correções publicadas; numa sessão já aberta, reinicie e rode a seção 2.
4. Resultados: `results/<id>/` no disco do Colab, copiados para `MyDrive/otft-sr/results/<id>/`
   ao fim de cada etapa longa; a reexecução retoma de onde parou.

Alternativa sem GitHub: `"$PY" scripts/empacotar_para_colab.py` gera `dist/otft-sr-<commit>.zip`;
copie para `MyDrive/otft-sr/` e use `OTFT_SR_FONTE = "zip"` na seção 2.

Localmente o notebook roda com `EXECUCAO_DE_VERIFICACAO = True` (código do Colab pulado) e
`BACKEND_VERIFICACAO = "simulado"` ou `"ollama"`; `"$PY" -m pytest -m lento` o executa.

## Estrutura

```
src/otft_sr/ovsed.py         OVSED completo (R_S por ponto fixo), forma fechada, dados medidos Org1
src/otft_sr/niveis.py        níveis N1-N4 (dataclass NivelEquacao) e erro vs OVSED
src/otft_sr/dados.py         grade por curvas, ruído multiplicativo, partição por curva -> Dataset
src/otft_sr/contexto.py      PhysicalContext por nível, bibliotecas de prompts, avaliador Ollama
src/otft_sr/provedores_llm.py ConfiguracaoLLM e provedores (Hugging Face 4/8 bits, Ollama, OpenAI, Anthropic, Google, simulado), registro de uso, ambiente, avaliador M1
src/otft_sr/experimento.py   configurações de busca (nb01_baseline.yaml + overrides), executar_condicao, controle positivo
src/otft_sr/gerador.py       modo M2: LLM gerador (G0, G1, G2), interpretação segura, ajuste, checagens, seleção
src/otft_sr/persistencia.py  results/<id>/ comum a M1/M2, retomada, saneamento de segredos, leitura legada
src/otft_sr/estatistica.py   agregação com IC t, Wilson, pareada por semente, McNemar exato, custo-qualidade
src/otft_sr/formulacao.py    formulação da perda do M1: linear (artigo) ou log10 (H-OTFT-19)
src/otft_sr/auditoria.py     auditoria do avaliador M1 (referência × formas erradas × absurda)
src/otft_sr/campanha.py      configuração única, plano de ensaios e retomada do notebook 02
src/otft_sr/visualizacao.py  figuras Plotly do notebook 02 (sem extrapolação)
prompts/generico/            prompt do artigo (cópia de physics-llm-sr/prompts)
prompts/dispositivo/         prompt de física de OTFT (critérios da proposta §5.2)
prompts/gerador/             templates modulares do modo M2 (hash registrado)
data/org1/                   curvas medidas e parâmetros da Org1 (cópia de Model_OTFT)
data/legado/                 mapeamento de colunas antigas do notebook 01 (só leitura)
references/premissas_cientificas.md   premissas H-OTFT-01...
notebooks/build_01_rs_otft.py        gerador do notebook 01_rs_otft.ipynb
notebooks/build_02_multi_llm_colab.py gerador do notebook 02 (Colab, multi-LLM)
tests/                       pytest (inclui varredura de terminologia)
results/                     CSV/JSON/figuras exportados pelo notebook
```
