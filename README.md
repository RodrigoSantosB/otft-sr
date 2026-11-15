# otft-sr — regressão simbólica guiada por LLM para OTFTs

Adapta Taskin, Xie & Lazebnik (Sci Rep 2026), perda L = w1·e + w2·s + w3·c, a equações de
corrente de dreno de OTFTs (proposta `proposta_de_artigo___DL__Stefan_.pdf`). Equações-alvo:
escada N1–N4 ancorada no modelo OVSED com parâmetros da Org1 (TCC, `Model_OTFT`).
Condições experimentais: A = RS sem LLM, B = LLM com prompt genérico, C = LLM com prompt de
física de dispositivo.

| Nível | Equação | Região |
|---|---|---|
| N1 | I_D = K (V_GS − V_Th)² | saturação (controle canônico) |
| N2 | I_D = K (V_GS − V_th0)^l | acumulação forte, F_sat constante |
| N3 | I_D = K (2 (V_GS − V_Th) V_DS − V_DS²) | triodo (V_DS ≤ V_GS − V_Th) |
| N4 | I_D = I_0 exp(ln10 V_GS / S) | sublimiar, I_leak removido (pA) |
| N5 | implícita com R_S | só documentado (H-OTFT-08) |

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

## Estrutura

```
src/otft_sr/ovsed.py         OVSED completo (R_S por ponto fixo), forma fechada, dados medidos Org1
src/otft_sr/niveis.py        escada N1-N4 (dataclass NivelEquacao) e erro vs OVSED
src/otft_sr/dados.py         grade por curvas, ruído multiplicativo, partição por curva -> Dataset
src/otft_sr/contexto.py      PhysicalContext por nível, bibliotecas de prompts, avaliador Ollama
src/otft_sr/experimento.py   configurações de busca (nb01_baseline.yaml + overrides), executar_condicao, controle positivo
prompts/generico/            prompt do artigo (cópia de physics-llm-sr/prompts)
prompts/dispositivo/         prompt de física de OTFT (critérios da proposta §5.2)
data/org1/                   curvas medidas e parâmetros da Org1 (cópia de Model_OTFT)
references/premissas_cientificas.md   premissas H-OTFT-01...
notebooks/build_01_rs_otft.py        gerador do notebook 01_rs_otft.ipynb
tests/                       pytest (inclui varredura de terminologia)
results/                     CSV/JSON/figuras exportados pelo notebook
```
