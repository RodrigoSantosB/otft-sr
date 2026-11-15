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
  ajustadas ao OVSED completo da Org1 (R_S e I_leak incluídos; I_leak removido no N4) na
  região do nível, por mínimo erro relativo quadrático (N1, N2, N3) ou regressão de
  log10 I_D (N4). Os dados de cada nível são gerados pela própria equação do nível
  (gerador = expressão de referência); o desvio para o OVSED é medido à parte.
- **H-OTFT-05 — N3 na forma quadrática clássica, restrita ao triodo.** A alternativa
  F_sat ≈ tanh(V_DS/(α V_ov)) exige tanh, fora do conjunto de operadores; escrita com exp
  ela passa de 25 nós com subárvores repetidas (recuperação improvável com max_nodes = 30).
  A forma I_D = K (2 (V_GS − V_Th) V_DS − V_DS²) tem 13 nós, só usa sub/mul/pow e é a
  expansão de τ(2 − τ) do F_sat OVSED para V_DS ≪ V_gn. Sem min/max, ela só vale em
  V_DS ≤ V_GS − V_Th; por isso a região do N3 é o triodo.
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
  I_0 exp(1,423 V_GS), com distância algébrica 1.
- **H-OTFT-08 — N5 fora do escopo de execução.** I_D = K (V_GS − V_Th − I_D R_C)^(γ+2) é
  implícita; os motores só buscam y = f(X) explícita. Para a Org1, I_D R_S ≤ ~0,1 V
  (I_D < 10 µA, R_S = 10 kΩ), efeito < 1 % na faixa medida. Trabalho futuro: RS sobre a
  inversa V_GS(I_D, V_DS) ou com I_D como entrada.
- **H-OTFT-09 — Contexto igual em B e C.** As condições B e C usam o mesmo PhysicalContext
  e a variante E (variáveis + experimento, sem a equação-alvo); só o prompt de sistema muda
  (`prompts/generico` = artigo; `prompts/dispositivo` = física de OTFT + critérios da
  proposta §5.2). Variantes D/F/G/H (revelam a GT) existem só por exigência da biblioteca.
- **H-OTFT-10 — Prompts em inglês.** Textos enviados ao LLM ficam em inglês, como o prompt
  reconstruído do artigo; o restante do projeto está em PT-BR.
- **H-OTFT-11 — N4 em pA.** Em µA o N4 tem Var(I_D) ~ 1e-15 µA², abaixo do mínimo do modo
  normalizado da biblioteca (1e-12); a saída do N4 é em pA (fator 1e6), unidade usual para
  correntes sublimiares.
- **H-OTFT-12 — S do N4.** O OVSED tem F_sat ∝ q_tot no sublimiar, logo I_D ∝ q_tot^(l+1) e a
  assíntota é S = ln10·n·V_T/(l+1) ≈ 1,50 V/déc (e não ln10·n·V_T/l ≈ 1,97 V/déc). Na região
  V_GS ∈ [0, 4] V a inclinação local é 1,60–1,71 V/déc; o N4 usa o S ajustado (≈ 1,62 V/déc).
  Essa região fica abaixo de I_leak = 65 pA: N4 descreve o canal com I_leak removido, não
  as medidas.
- **H-OTFT-13 — N2 com F_sat constante.** No OVSED, F_sat em saturação varia de 0,074 a 0,17
  entre V_GS = 20 e 50 V (F_sat ≈ V_gn/(V_gn + 4ΛV_T)); fixá-lo em F_sat,ref ≈ 0,103 dá erro
  relativo médio de ~24 % contra o OVSED (0,13 década). O expoente efetivo do OVSED em
  saturação é maior que l.
- **H-OTFT-14 — V_DS como variável distratora.** Todos os níveis recebem (V_GS, V_DS); em N1,
  N2 e N4 a corrente não depende de V_DS.
- **H-OTFT-15 — Orçamento.** "completo" = `best_config.yaml` da queda livre por motor
  (PySR 8×50, 200 ciclos, 20 iterações; DEAP 500×200; gplearn 1000×100); "fumaca" só testa o
  encadeamento e nunca é resultado.
