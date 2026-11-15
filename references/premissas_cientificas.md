# Premissas científicas (H-OTFT)

Registro curto de toda simplificação ou escolha não imposta pela proposta. Valores numéricos
gerados por `src/otft_sr/niveis.py` (constantes arredondadas a 4 algarismos significativos).

- **H-OTFT-01 — Convenção de sinal.** Canal p tratado em módulo: V_GS, V_DS e I_D ≥ 0
  (proposta §4.4). O modelo é a reimplementação de `Model_OTFT/modules_otft/_model.py`
  (sem importar `modules_otft`).
- **H-OTFT-02 — V_th0 = 8,57 V (e não 18,57 V).** O JSON da Org1 traz 18,57, valor fora dos
  próprios limites de otimização do arquivo ([8, 12] V). Contra as 5 curvas medidas de
  `datas/Org1`, o OVSED completo dá RMSE em log10|I_D| de 0,125 década com 8,57 e de 0,885
  década com 18,57. Adotado 8,57 (o mesmo dos índices sintéticos e do RELATORIO_MLP).
- **H-OTFT-03 — Unidades dos dados medidos.** Transferência (`*VDS.csv`): I_D em A;
  saída (`*VGS.csv`): rotulada A, lida como µA (`experimental_data_scale_output = uA`).
- **H-OTFT-04 — Ancoragem das constantes.** Em cada nível, as constantes livres são
  ajustadas ao OVSED completo da Org1 (R_S e I_leak incluídos em todos os níveis) na
  região do nível, por mínimo erro relativo quadrático (N1, N2, N3) ou mínimo erro em
  log10 I_D (N4, com I_leak fixo no valor do OVSED; H-OTFT-17). Os dados de cada nível são gerados pela própria equação do nível
  (gerador = expressão de referência); o desvio para o OVSED é medido à parte.
- **H-OTFT-05 — N2 (regime linear) na forma quadrática clássica, restrita ao triodo.** A alternativa
  F_sat ≈ tanh(V_DS/(α V_ov)) exige tanh, fora do conjunto de operadores; escrita com exp
  ela passa de 25 nós com subárvores repetidas (recuperação improvável com max_nodes = 30).
  A forma I_D = K (2 (V_GS − V_Th) V_DS − V_DS²) tem 13 nós, só usa sub/mul/pow e é a
  expansão de τ(2 − τ) do F_sat OVSED para V_DS ≪ V_gn. Sem min/max, ela só vale em
  V_DS ≤ V_GS − V_Th; por isso a região do N2 é o triodo.
- **H-OTFT-06 — Partição por curva.** Treino/validação/teste = 60/20/20 % das curvas
  (20 curvas × 50 pontos por padrão); `simulation_ids` = id da curva. O ruído multiplicativo
  gaussiano (σ ∈ {0; 1; 5; 10} % de |I_D|) entra em todas as partições; métricas de teste
  usam o I_D limpo.
- **H-OTFT-07 — Perda normalizada e pesos.** e = MSE/Var(y), s = nós/30 (trilha
  `extended_analysis`, `loss.max_nodes = 30`, `sr.pysr.maxsize = 30`). Pesos fixados por
  razões: w2/w1 = 0,003 (um nó custa o mesmo que 1e-4 de 1 − R²) e w3/w1 = 0,01 (a escala
  inteira de c vale 1e-2 de 1 − R²); A = (0,997; 0,003; 0), B/C = (0,987; 0,003; 0,010).
  Uma primeira tentativa com (0,8; 0,1; 0,1) foi descartada: com e normalizado, a
  parcimônia dominava e o mínimo de L não era a equação-alvo (PySR, N1: 1,048^V_GS − 3,33,
  5 nós, 1 − R² = 1,3e-3, tinha L menor que a referência de 7 nós com 1 − R² = 0). O modo
  literal do artigo (MSE em unidades físicas + nós) depende da unidade de I_D. PySR com MSE
  (não Huber), como no `best_config` da queda livre.
- **H-OTFT-07b — Equivalência simbólica.** Além das distâncias de árvore do
  `physics_llm_sr`, a forma funcional é comparada com sympy (`src/otft_sr/equivalencia.py`:
  c^f(V) -> exp(f log c), expand_log, expansão em termos, constantes mascaradas, expoentes
  reais a 2 decimais). Classe "recupera estrutura (equivalência sympy)" exige também
  1 − R² ≤ 1e-3. Motivo: o PySR devolveu log(1,00064^(4,15^V_GS)), idêntico a
  I_0 exp(1,423 V_GS) (N4 anterior a H-OTFT-17), com distância algébrica 1.
- **H-OTFT-08 — N5 (saturação com resistência de contato) fora do escopo de execução.** I_D = K (V_GS − V_Th − I_D R_C)^(γ+2) é
  implícita; os algoritmos de RS só buscam y = f(X) explícita. Para a Org1, I_D R_S ≤ ~0,1 V
  (I_D < 10 µA, R_S = 10 kΩ), efeito < 1 % na faixa medida. Trabalho futuro: RS sobre a
  inversa V_GS(I_D, V_DS) ou com I_D como entrada.
- **H-OTFT-09 — Contexto igual em B e C.** As condições B e C usam o mesmo PhysicalContext
  e a variante E (variáveis + experimento, sem a equação-alvo); só o prompt de sistema muda
  (`prompts/generico` = artigo; `prompts/dispositivo` = física de OTFT + critérios da
  proposta §5.2). Variantes D/F/G/H (revelam a GT) existem só por exigência da biblioteca.
- **H-OTFT-10 — Prompts em inglês.** Textos enviados ao LLM ficam em inglês, como o prompt
  reconstruído do artigo; o restante do projeto está em PT-BR.
- **H-OTFT-11 — N4 em pA.** A saída do N4 é em pA (fator 1e6), unidade usual para correntes
  sublimiares. Com a fuga aditiva (H-OTFT-17), Var(I_D) ≈ 1e3 pA² nos conjuntos gerados,
  muito acima do mínimo do modo normalizado da biblioteca (1e-12); o N4 anterior, sem fuga,
  tinha Var(I_D) ~ 1e-15 µA² e só passava esse mínimo em pA.
- **H-OTFT-12 — S do N4 é um swing efetivo.** O OVSED tem F_sat ∝ q_tot no sublimiar, logo
  I_D ∝ q_tot^(l+1) e a assíntota é S = ln10·n·V_T/(l+1) ≈ 1,50 V/déc (e não
  ln10·n·V_T/l ≈ 1,97 V/déc); em V_GS ∈ [0, 4] V a inclinação local do canal é
  1,60–1,71 V/déc. Esse sublimiar profundo (0,001–0,2 pA) fica 2 a 5 décadas abaixo de
  I_leak = 65 pA e não é observável na Org1. O canal só supera I_leak a partir de
  V_GS ≈ 8,9 V (V_DS = 2 V), onde n = 108 já põe o softplus em transição: a inclinação local
  sobe de 2,2 V/déc (8 V) para 2,8 V/déc (10 V). O S do N4 (2,33 V/déc) é o swing efetivo
  dessa transição e depende do limite superior da região (2,01; 2,16; 2,33; 2,51 V/déc para
  8, 9, 10 e 11 V); não deve ser lido como o S do sublimiar profundo.
- **H-OTFT-13 — N3 (lei de potência) com F_sat constante.** No OVSED, F_sat em saturação
  varia de 0,024 a 0,108 na região do N3 (V_GS de 10,5 a 28 V; F_sat ≈ V_gn/(V_gn + 4ΛV_T));
  fixá-lo em F_sat,ref ≈ 0,055 e usar q_tot ≈ (V_GS − V_th0)/(n V_T) (o softplus ainda não é
  identidade perto do limiar) dá erro relativo médio de 25 % contra o OVSED (RMSE de 0,140
  década, máximo de 0,256 década; contra a Org1 medida a V_DS = 50 V, 0,117 década em 19
  pontos). O expoente efetivo do OVSED em saturação é maior que l.
- **H-OTFT-14 — V_DS como variável distratora.** Todos os níveis recebem (V_GS, V_DS); em N1,
  N3 e N4 a corrente não depende de V_DS.
- **H-OTFT-15 — Configuração de busca.** "calibrada" = `best_config.yaml` da queda livre por
  algoritmo de RS (PySR 8×50, 200 ciclos, 20 iterações; DEAP 500×200; gplearn 1000×100);
  "verificacao" = verificação de integridade do pipeline, não constitui resultado.
- **H-OTFT-16 — Nomenclatura dos níveis.** O id `N<k>` é o nível k da proposta (Seção 5.1) e
  `nivel_proposta` = k acompanha o id em `NivelEquacao` e nas linhas de resultado.
  Correspondência (id ↔ nome científico ↔ id antes do alinhamento):
  N1 saturação quadrática ideal (antes N1); N2 regime linear (triodo) com termo quadrático
  em V_DS (antes N3); N3 saturação com mobilidade em lei de potência (antes N2); N4 corrente
  sublimiar exponencial (antes N4); N5 saturação com resistência de contato (só documentado,
  H-OTFT-08). Na troca de posição as expressões não mudaram; depois, N3 e N4 mudaram de
  região e de expressão (H-OTFT-17, H-OTFT-18), registradas com as anteriores em
  `tests/test_paridade_niveis.py`. Resultados exportados antes da troca usam a numeração
  antiga na coluna `nivel`; os exportados antes da revisão das regiões usam as expressões
  antigas de N3 e N4 (coluna `expressao` do JSON `niveis_referencia`).
- **H-OTFT-17 — N4 com fuga aditiva, na região observável.** I_D = I_leak + I_0 exp(V_GS/V_S),
  V_S = S/ln10, com I_leak = 65 pA fixo (parâmetro do OVSED; ajustado livre daria 64,6 pA) e
  I_0, S por mínimo erro em log10 contra o OVSED completo. Região: transferência com
  V_DS ∈ [1, 5] V e V_GS ∈ [0, 10] V, do patamar de fuga até pouco acima do joelho (canal =
  I_leak em ~8,9 V). Acima de 10 V o erro no limite superior passa de 0,17 década (0,24 em
  11 V) e o canal deixa de ser exponencial. Resultado: 65,0 + 0,009502 exp(0,9882 V_GS) pA;
  RMSE de 0,024 década contra o OVSED (máximo de 0,17 década em V_GS = 10 V) e de 0,033
  década contra a Org1 medida a V_DS = 2 V (6 pontos, V_GS de 4,4 a 10 V; o OVSED completo
  dá 0,032 década nos mesmos pontos). Sem a fuga, o exponencial fica > 1 década abaixo da
  medida. Limite: a faixa acima da fuga é curta (0,59 década de faixa dinâmica, ~11 % dos
  pontos com canal > I_leak); com ruído de 5 % o S ainda é identificável (erro-padrão
  relativo ~0,9 % por informação de Fisher, 1000 pontos), mas o valor é o swing efetivo de
  H-OTFT-12. Se o objetivo for o S do sublimiar profundo, a Org1 não o mede: seria preciso um
  dispositivo ou medida com fuga ≲ 0,01 pA, ou manter um nível sintético auxiliar só com o
  canal (o N4 anterior, 0,0006386 exp(1,423 V_GS) pA em V_GS ∈ [0, 4] V), declarado não
  observável.
- **H-OTFT-18 — Região do N3 perto do limiar.** Em V_GS ∈ [20, 50] V a lei de potência e a
  lei quadrática quase coincidem (o PySR devolveu (V_GS − 8,99)^3,137): o melhor
  K (V_GS − V_T)² com K e V_T livres ajusta os dados do N3 com RMSE de 0,052 década. A região
  foi escolhida por varredura de [a, b] (V_DS ∈ [40, 50] V), maximizando esse RMSE do melhor
  ajuste quadrático (discriminabilidade) com o erro do N3 contra o OVSED limitado a 0,15
  década (RMSE) e 0,30 década (máximo). Trechos da varredura (discriminabilidade / RMSE vs
  OVSED / máximo vs OVSED, em década): [20, 50] 0,052 / 0,125 / 0,216; [12, 50] 0,177 /
  0,237 / 0,370; [10, 25] 0,171 / 0,142 / 0,506; [11, 28] 0,129 / 0,139 / 0,260; [10,5; 30]
  0,167 / 0,153 / 0,275; escolhida [10,5; 28]: 0,155 / 0,140 / 0,256 (3,0× a região
  anterior). Descer abaixo de ~10,5 V faz o N3 subestimar o OVSED por fator > 2 na borda
  (q_tot ≈ θ falha); subir o limite acima de ~28 V aumenta o erro de F_sat constante. Em
  1 − R² linear, que é o que a perda normalizada mede, a discriminabilidade quase não depende
  da região (3,8e-3 contra 3,2e-3 em [20, 50] V): o MSE é dominado pelas correntes altas, e
  um V_T livre acomoda a curvatura. 3,8e-3 em 1 − R² equivale a ~38 nós na perda
  (H-OTFT-07), mas é da ordem do 1 − R² que o próprio ruído de 5 % produz (~4e-3). Já a
  compensação expoente × limiar ((V_GS − V_x)^3,137 contra ^3,18) dá 1 − R² ~ 2e-6 em
  qualquer região e não é resolvida pela escolha da região.
- **H-OTFT-19 — Formulação da perda do M1 em log10 (opção; padrão "linear").** Com
  `formulacao_perda = "log10"` (`src/otft_sr/formulacao.py`) o alvo da RS passa a ser
  log10(I_D) e e = MSE/Var(log10 I_D); nada muda no `physics_llm_sr`. A referência é
  log10 da expressão do nível, escrita com `log` natural e expandida por sympy (`expand_log`
  com `force=True`, válido na região, onde as bases são positivas): N1
  −2,081 + 0,8686 log(V_GS − 20,24); N3 −4,658 + 1,381 log(V_GS − 8,57); N2 e N4 ficam
  0,4343 log(...) do polinômio/da soma. A expansão é obrigatória: o `log` protegido vale 0
  para |x| < 1e-3 (GAP-06) e o I_D do N3 desce a ~2e-4 µA. O ruído multiplicativo é aplicado
  em I_D antes do log (mesmos dados da formulação linear; valores ≤ 0 recebem piso de 1e-6 do
  menor positivo). Métricas: `one_minus_r2` no alvo da formulação (é o que o critério de
  recuperação usa), `rmse_log10` direto em década (comparável entre formulações) e
  `one_minus_r2_I_D` com 10^f contra I_D. O contexto enviado ao LLM declara que a equação
  prevê log10(I_D). Ensaios log10 têm chave de retomada própria; os lineares mantêm a chave
  anterior. Limite: em log10 o limiar 1 − R² ≤ 1e-3 tem outra severidade e as distâncias de
  árvore são medidas contra a forma expandida; resultados das duas formulações não entram na
  mesma comparação pareada.
- **H-OTFT-20 — Decisões do modo M2 (gerador; exploratório), fixadas antes de qualquer
  execução real.** Interpretação: primeiro objeto JSON da resposta, schema pydantic
  (`equation`, `variables`, `parameters`), `parse_infix` com whitelist de AST (sem
  eval/exec/sympify), operações do PySR no M1, no máximo 30 nós, profundidade 8 e 6
  parâmetros, até `max_retentativas` novas tentativas com o motivo da recusa. Ajuste:
  `least_squares` (TRF) no treino com resíduo em log10|I_D| (proposta §4.4), 8 inícios
  determinísticos. Checagens físicas no treino e na grade da região (positividade,
  monotonia em V_GS, limiar, saturação; dimensional só informativa). Seleção só pela
  validação (fisicamente válido, menor RMSE de log10 I_D com empate a 1e-6 década, menos
  nós); teste só do selecionado, com as métricas do M1 (`formulacao_perda = "ajuste_log10"`
  nas linhas do M2). G0 e G1 avaliam `n_candidatos`; G2 avalia `n_candidatos × (1 + k)`, por
  isso G1 × G2 é comparação exploratória (esforço diferente) e G0 × G1 é equivalente.
