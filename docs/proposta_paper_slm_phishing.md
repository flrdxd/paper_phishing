# Proposta de Paper - Detecção SLM-first de Phishing por E-mail

**Titulo provisório em inglês:**  
**SLM-First and URL-Aware Phishing Email Detection Under Realistic Base Rates and AI-Rewritten Attacks**

**Titulo provisório em português:**  
**Detecção de Phishing por E-mail com SLM-first, URLs e Avaliação Realista contra Ataques Reescritos por IA**

---

## 1. Tese Central

O paper baseline compara Naive Bayes, Dandelion-Naive Bayes, BERT e DistilBERT para detecção de phishing por e-mail. A nossa reprodução mostrou um ponto importante:

- **BERT e DistilBERT foram reproduzidos com alta fidelidade**, com F1 acima de 0.99 e diferença pequena em relação ao paper.
- **Naive Bayes ficou próximo**, mas abaixo do baseline.
- **Dandelion-Naive Bayes não reproduziu o ganho reportado pelo paper**, ficando praticamente igual ao Naive Bayes simples.

Portanto, o novo paper não deve tentar vencer o baseline apenas em accuracy. Esse caminho é fraco porque o dataset balanceado já está quase saturado pelos transformers.

A tese recomendada é:

> Um pipeline SLM-first, combinando compact text encoders, features de URL/metadados e fusão calibrada, pode manter alto recall de phishing com menor custo, menor latência e menos falsos positivos em cenários mais realistas que datasets balanceados tradicionais, especialmente sob phishing reescrito por IA e base rates desbalanceados.

Neste documento, **SLM** significa principalmente **compact transformer encoders / small language encoders** para classificação local, como DistilBERT, MiniLM, TinyBERT, MobileBERT e DeBERTa-v3-small. Isso evita a ambiguidade com pequenos LLMs generativos.

---

## 2. Recorte do Paper v1

### 2.1 O que entra no v1

O paper v1 deve focar em uma contribuição executável:

```text
Texto do e-mail
+ URL/metadados disponíveis
+ modelo tabular
+ compact text encoder
+ fusão calibrada
+ avaliação desbalanceada
+ robustez contra AI-rewrite
```

Componentes centrais:

1. Reprodução do baseline.
2. Avaliação em dataset multi-source.
3. Compact encoders para texto.
4. Features de URL/metadados.
5. Fusão de scores e calibração.
6. Simulação de base rates realistas.
7. Teste de robustez contra phishing reescrito por IA.

### 2.2 O que fica fora do v1

Estas ideias são boas, mas aumentam demais o escopo:

- RAG/contexto local completo;
- LLM fallback em produção;
- QR code e anexos;
- PETA/estudo com usuários;
- PT-BR como contribuição principal;
- distilação LLM -> SLM como eixo principal.

Esses itens devem aparecer como **extensões futuras**, não como requisitos do primeiro paper.

---

## 3. Pergunta de Pesquisa

### Pergunta principal

> Um detector SLM-first híbrido, com texto + URL/metadados e fusão calibrada, consegue reduzir falsos positivos e custo operacional mantendo recall alto em cenários realistas de phishing por e-mail?

### Perguntas secundárias

1. Compact encoders conseguem desempenho competitivo com BERT/DistilBERT em phishing por e-mail?
2. URL e metadados melhoram a detecção além do texto?
3. Fusão calibrada reduz falsos positivos mantendo recall alto?
4. Benchmarks multi-source e splits por fonte reduzem o desempenho aparente de modelos treinados em splits aleatórios?
5. Phishing reescrito por IA reduz o desempenho de modelos text-only?
6. Base rates realistas mudam a decisão sobre qual arquitetura é melhor?

---

## 4. Método Científico e Estratégia de Descoberta

Este projeto não deve começar com a conclusão pronta. A abordagem correta é usar os experimentos para descobrir qual tese se sustenta.

Fluxo geral:

```text
pergunta ampla
-> hipóteses testáveis
-> experimentos pequenos
-> resultados
-> descarte ou ajuste de hipóteses
-> escolha da arquitetura candidata
-> congelamento do protocolo
-> teste final confirmatório
-> escrita do paper com evidência
```

### 4.1 Fase exploratória

Nesta fase, testamos hipóteses para decidir o que merece entrar no paper final.

Exemplos de decisões:

- se MiniLM ficar próximo do DistilBERT com menor custo, ele vira candidato principal;
- se URL/metadados reduzirem falso positivo, entram no sistema final;
- se URL/metadados não ajudarem, saem do núcleo do paper;
- se AI-rewrite derrubar modelos text-only, robustez vira contribuição central;
- se split por fonte reduzir muito as métricas, generalização vira argumento principal;
- se fusão calibrada não melhorar FPR@Recall>=98%, usamos uma arquitetura mais simples.

O objetivo desta fase é aprender, não confirmar uma narrativa pré-definida.

### 4.2 Critérios de decisão

Cada hipótese deve produzir uma decisão objetiva:

| Resultado experimental | Decisão |
|---|---|
| Modelo compacto mantém desempenho próximo com menor latência | Manter como candidato principal |
| URL/metadados reduzem FPR mantendo recall | Incluir ramo tabular |
| Fusão calibrada melhora FPR@Recall>=98% | Manter fusão/calibração |
| AI-rewrite reduz muito o recall | Colocar robustez como contribuição central |
| Split por fonte derruba modelos text-only | Priorizar generalização cross-source |
| Ganho de um componente é pequeno ou instável | Remover do v1 |

### 4.3 Fase confirmatória

Após a fase exploratória, congelamos:

- arquitetura;
- modelos;
- features;
- thresholds;
- métricas primárias;
- splits;
- datasets;
- protocolo estatístico.

Depois disso, o teste final deve ser executado uma única vez. Essa separação evita escolher o melhor resultado depois de muitas tentativas e fortalece a validade científica do paper.

### 4.4 Resultado esperado do processo

O paper final deve nascer dos resultados, não da intuição inicial.

Exemplos de teses finais possíveis:

- **Tese A:** MiniLM + URL/metadados + calibração reduz falsos positivos mantendo recall alto e custo menor que BERT.
- **Tese B:** DistilBERT continua sendo o melhor texto-only, mas URL/metadados são essenciais para cenário operacional.
- **Tese C:** compact encoders são suficientes em split aleatório, mas falham em cross-source; a contribuição principal passa a ser generalização.
- **Tese D:** AI-rewrite derruba modelos text-only, e a robustez passa a ser o eixo principal do paper.

Isso significa que a proposta atual é um mapa experimental. A hipótese final do paper será definida depois que os testes mostrarem qual argumento é realmente defensável.

---

## 5. Pesquisa de Estado da Arte

### 5.1 Diagnóstico geral

A literatura recente indica uma lacuna clara:

- Muitos trabalhos atingem desempenho quase perfeito em datasets antigos, balanceados ou com split aleatório.
- LLMs conseguem alta acurácia, mas têm custo, latência, privacidade e reprodutibilidade piores.
- Trabalhos recentes começam a avaliar phishing como problema multimodal, contextual, adversarial e operacional.
- Há datasets novos que tentam corrigir a limitação de benchmarks tradicionais.

### 5.2 Trabalhos relevantes

| Trabalho | Relevância para o nosso paper | Como usar |
|---|---|---|
| **Paper baseline WF-PST 2025** | Base de comparação: NB, Dandelion-NB, BERT e DistilBERT | Reproduzir e usar como ponto de partida |
| **E-PhishGen / E-PhishLLM** | Argumenta que datasets antigos estão saturados e propõe benchmarks mais difíceis, multilíngues e gerados por LLM | Usar para motivar robustez e teste externo |
| **MultiPhishGuard** | Sistema multiagente com agentes para texto, URL, metadados, explicação e adversarial training | Referência de estado da arte complexo; nosso contraste é custo/implantação |
| **ChatSpamDetector** | LLMs conseguem alta acurácia e explicações em phishing | Usar como baseline conceitual LLM-only, não como arquitetura principal |
| **LLM-PEA** | Avalia LLMs em phishing e discute prompt injection, ataques adversariais e multilinguismo | Usar para justificar cautela com LLM-only |
| **PiMRef** | Detecta spear-phishing usando invariantes e checagem de identidade/contexto | Valida a importância de metadados, remetente e referência externa |
| **Small Language Models for Phishing Website Detection** | Mostra viabilidade de modelos menores locais para phishing | Apoia a tese SLM-first |
| **PhiUSIIL URL Dataset** | Dataset tabular de URLs com muitas features | Usar para ramo URL/tabular |
| **MeAJOR Corpus** | Corpus multi-source de e-mails para phishing | Usar como dataset principal novo |

### 5.3 Fontes principais

- Baseline: *Optimizing Phishing Detection: Comparative Analysis of Lightweight Machine Learning and Transformer Models*, IEEE WF-PST 2025.
- MeAJOR Corpus: <https://arxiv.org/abs/2507.17978> e <https://zenodo.org/records/18471483>
- PhiUSIIL URL Dataset: <https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bdataset>
- E-PhishGen: <https://arxiv.org/abs/2509.01791>
- PhishFuzzer: <https://arxiv.org/abs/2511.21448>
- MultiPhishGuard: <https://arxiv.org/abs/2505.23803>
- ChatSpamDetector: <https://arxiv.org/abs/2402.18093>
- LLM-PEA: <https://arxiv.org/abs/2512.10104>
- PiMRef: <https://arxiv.org/abs/2507.15393>
- SLMs for phishing website detection: <https://arxiv.org/abs/2511.15434>
- CIC-Trap4Phish: <https://www.unb.ca/cic/datasets/trap4phish2025.html>

---

## 6. Datasets

### 6.1 Prioridade alta

| Dataset | Uso | Decisão |
|---|---|---|
| **Baseline atual** | Comparação direta com o paper base | Obrigatório |
| **MeAJOR Corpus** | Dataset multi-source para e-mail, generalização e split por fonte | Principal dataset novo |
| **PhiUSIIL URL Dataset** | Benchmark tabular de URL/webpage features | Validar ramo URL/metadados |

### 6.2 Prioridade média

| Dataset | Uso | Decisão |
|---|---|---|
| **PhishFuzzer** | Robustez, metadados, URL, intent e classes mais ricas | Usar após auditoria |
| **E-PhishLLM / E-PhishGen** | Teste externo contra phishing moderno e LLM-generated | Usar se dados estiverem acessíveis |

### 6.3 Extensão futura

| Dataset | Uso | Decisão |
|---|---|---|
| **CIC-Trap4Phish 2025** | Anexos, QR code, HTML e multi-formato | Fora do v1 |
| **Phishing-PT** | Português brasileiro | Stress test separado |
| **Dados PT-BR sanitizados** | Avaliação localizada defensiva | Extensão, não contribuição principal |

### 6.4 Observações de reprodutibilidade

- O baseline reporta aproximadamente 82.486 amostras. A nossa execução removeu duplicatas/textos vazios e ficou com 82.076 amostras finais.
- O MeAJOR pode ter diferença entre versão do paper e versão prática em Zenodo. O documento final deve registrar a versão exata usada.
- Datasets sintéticos ou gerados por LLM devem ser usados com cuidado, preferencialmente como teste de robustez, não como único benchmark.

---

## 7. Modelos a Testar

### 7.1 Baselines leves

- Naive Bayes + TF-IDF.
- Logistic Regression + TF-IDF.
- Linear SVM + TF-IDF.
- LightGBM/XGBoost em features estruturais.
- Random Forest apenas como baseline interpretável.

### 7.2 Compact text encoders

- DistilBERT.
- MiniLM.
- TinyBERT.
- MobileBERT.
- DeBERTa-v3-small.
- ModernBERT-base como candidato forte, mas mais caro.
- RoBERTa-base como upper baseline, não como modelo pequeno.

### 7.3 URL e metadados

- Logistic Regression calibrada.
- LightGBM.
- XGBoost.
- URLBERT-tiny como baseline neural específico de URL.
- Features manuais de URL/domínio/remetente quando disponíveis.

### 7.4 LLMs e pequenos LLMs generativos

Não entram no núcleo do v1. Podem aparecer em duas funções opcionais:

1. Gerar rewrites defensivos e sanitizados para teste de robustez.
2. Servir como teacher em experimento futuro de distilação.

Exemplos para trabalhos futuros:

- Qwen 2.5 1.5B/3B;
- Phi mini;
- Llama 3.x 3B;
- Gemma pequeno.

---

## 8. Arquitetura v1

```text
E-mail
  |
  |-- texto: assunto + corpo
  |       -> compact text encoder
  |       -> text_score
  |
  |-- URLs/metadados disponíveis
  |       -> extrator de features
  |       -> modelo tabular
  |       -> url_meta_score
  |
  |-- fusão calibrada
          -> score_final
          -> decisão: legítimo / phishing / suspeito
```

### 8.1 Features textuais

- assunto;
- corpo;
- tokens de URL preservados quando útil;
- termos de urgência;
- pedidos de credenciais;
- valores financeiros;
- sinais de ameaça/recompensa.

### 8.2 Features de URL

- quantidade de URLs;
- tamanho da URL;
- quantidade de subdomínios;
- presença de IP;
- presença de `@`;
- encurtadores;
- TLD incomum;
- entropia do domínio;
- tamanho do caminho;
- quantidade de parâmetros;
- tokens como `login`, `verify`, `update`, `secure`, `account`;
- punycode;
- mismatch entre texto âncora e URL real, quando disponível.

### 8.3 Features de metadados

Usar apenas quando o dataset disponibilizar:

- domínio do remetente;
- domínio do reply-to;
- mismatch From/Reply-To;
- presença de anexos;
- tipo de anexo;
- presença de HTML;
- idioma;
- horário de envio;
- número de destinatários.

### 8.4 Fusão e calibração

Estratégias iniciais:

- média ponderada calibrada;
- Logistic Regression como meta-classificador;
- LightGBM como meta-classificador, se houver features suficientes.

Métrica principal para escolher threshold:

```text
FPR@Recall>=98%
```

---

## 9. Hipóteses e Experimentos

### H0 - Baseline reproduzido

**Objetivo:** congelar os resultados atuais.

Modelos:

- Naive Bayes;
- Dandelion-NB;
- BERT;
- DistilBERT.

Conclusão atual:

- BERT/DistilBERT: alta fidelidade ao paper.
- Naive Bayes: próximo, mas abaixo.
- Dandelion-NB: não reproduziu o ganho reportado.

### H1 - Dataset mais realista reduz desempenho aparente

**Objetivo:** testar se desempenho alto depende de split aleatório/dataset saturado.

Executar no MeAJOR:

- Naive Bayes;
- Logistic Regression;
- Linear SVM;
- DistilBERT;
- MiniLM.

Comparar:

- split aleatório;
- split por fonte, quando possível;
- teste externo cross-source.

Métricas:

- F1;
- MCC;
- AUC-PR;
- FPR;
- FPR@Recall>=98%.

### H2 - URL/metadados melhoram além do texto

**Objetivo:** medir ganho de features estruturais.

Comparações:

```text
texto-only
URL-only
metadados-only
texto + URL
texto + URL + metadados
```

Modelos:

- compact encoder para texto;
- LightGBM/XGBoost para URL/metadados;
- Logistic Regression calibrada para fusão.

### H3 - Fusão calibrada reduz falso positivo

**Objetivo:** reduzir bloqueio indevido de e-mails legítimos.

Comparações:

- melhor text-only;
- melhor tabular-only;
- weighted average;
- Logistic Regression calibrada;
- LightGBM stacking.

Métrica principal:

```text
FPR@Recall>=98%
```

### H4 - Compact encoder é suficiente

**Objetivo:** descobrir se um encoder menor entrega desempenho próximo ao DistilBERT/BERT.

Comparar:

- DistilBERT;
- MiniLM;
- TinyBERT;
- MobileBERT;
- DeBERTa-v3-small;
- BERT como upper baseline.

Métricas:

- F1;
- MCC;
- latência média;
- latência p95;
- memória;
- custo por 10 mil e-mails.

### H5 - Robustez contra phishing reescrito por IA

**Objetivo:** medir queda de desempenho quando o texto fica mais natural.

Fontes:

- PhishFuzzer;
- E-PhishLLM/E-PhishGen;
- conjunto sanitizado próprio, se necessário.

Categorias:

- phishing original;
- phishing reescrito por IA;
- phishing traduzido;
- phishing com tom corporativo;
- phishing sem palavras óbvias;
- legítimos difíceis.

Métrica:

```text
Robustness Gap = F1_original - F1_adversarial
```

### H6 - Base rate realista muda a decisão

**Objetivo:** avaliar impacto operacional.

Simulações:

```text
50% phishing / 50% legítimo
10% phishing / 90% legítimo
5% phishing / 95% legítimo
1% phishing / 99% legítimo
0.5% phishing / 99.5% legítimo
```

Métricas:

- falsos positivos por 10 mil e-mails;
- falsos negativos por 10 mil e-mails;
- custo por 10 mil e-mails;
- latência p95;
- percentual de mensagens classificadas como suspeitas.

---

## 10. Métricas

### 10.1 Métricas clássicas

- Accuracy;
- Precision;
- Recall;
- F1-score.

### 10.2 Métricas para cenário desbalanceado

- MCC;
- AUC-PR;
- Balanced Accuracy;
- FPR;
- FNR;
- TNR/especificidade.

### 10.3 Métricas operacionais

- FP por 10 mil e-mails;
- FN por 10 mil e-mails;
- custo por 10 mil e-mails;
- latência média;
- latência p95;
- uso de memória;
- throughput;
- percentual de casos suspeitos.

### 10.4 Métricas de robustez

- F1 por categoria;
- recall por categoria;
- robustness gap;
- queda de desempenho em AI-rewrite;
- queda de desempenho em cross-source.

### 10.5 Métricas de calibração

- ECE;
- Brier Score;
- reliability diagram;
- confidence vs accuracy.

### 10.6 Métrica central recomendada

A métrica principal do paper deve ser:

```text
FPR@Recall>=98%
```

Motivo:

> Em segurança de e-mail, detectar phishing é obrigatório, mas falso positivo também tem custo operacional alto. Essa métrica força a comparação sob uma restrição realista.

---

## 11. Testes Estatísticos

Usar estatística para evitar conclusão baseada em variação aleatória.

Recomendado:

- bootstrap para F1, MCC, AUC-PR e FPR;
- intervalo de confiança de 95%;
- McNemar para comparar classificadores no mesmo teste;
- paired bootstrap para diferença de métricas;
- Holm-Bonferroni quando houver múltiplas comparações;
- tamanho de efeito: diferença absoluta, diferença relativa, redução de FPR e redução de custo.

Exemplo de conclusão desejada:

> O modelo híbrido reduziu o FPR em 38% em relação ao compact encoder text-only, mantendo recall acima de 98%, com latência p95 2.4x menor que BERT.

---

## 12. Prevenção de Vazamento

Riscos:

- duplicatas entre treino e teste;
- templates quase idênticos em treino e teste;
- mesma URL/domínio em treino e teste;
- mesma fonte determinando a classe;
- split aleatório inflando desempenho;
- dataset sintético fácil demais.

Mitigações:

- deduplicação exata;
- deduplicação aproximada quando possível;
- split por fonte;
- split por domínio/campanha quando houver dados;
- teste cross-source;
- teste externo em dataset não usado na fase exploratória;
- congelar arquitetura antes do teste final.

---

## 13. Estrutura do Artigo

### 1. Introduction

- phishing continua relevante;
- IA generativa torna mensagens mais naturais;
- modelos grandes têm custo e privacidade como barreiras;
- SLM-first é alternativa implantável;
- lacuna: benchmarks balanceados não medem custo operacional/falsos positivos.

### 2. Related Work

- lightweight ML para phishing;
- transformers para phishing;
- LLMs para phishing;
- SLMs/compact encoders;
- URL e metadados;
- datasets recentes e lacunas de avaliação.

### 3. Baseline Reproduction

- reprodução do paper base;
- resultados obtidos;
- observação honesta sobre Dandelion.

### 4. Methodology

- hipóteses;
- datasets;
- splits;
- modelos;
- métricas;
- prevenção de vazamento;
- fase exploratória e teste final.

### 5. Proposed SLM-first Framework

- texto;
- URL/metadados;
- fusão;
- calibração;
- decisão.

### 6. Experiments

- MeAJOR;
- compact encoders;
- URL/metadados;
- fusão;
- robustez;
- base rates.

### 7. Results

- desempenho;
- FPR@Recall;
- custo/latência;
- robustez;
- ablation.

### 8. Discussion

- quando modelo pequeno basta;
- quando URL/metadados ajudam;
- quando BERT ainda vale;
- limitações.

### 9. Threats to Validity

- datasets públicos;
- dados sintéticos;
- disponibilidade de metadados;
- generalização por idioma;
- diferença de hardware;
- incompletude do baseline Dandelion.

### 10. Conclusion

- SLM-first como alternativa viável;
- avaliação operacional mais realista;
- próximos passos.

---

## 14. Tabelas Planejadas

### Tabela 1 - Baseline reproduzido

| Modelo | Accuracy paper | Accuracy nosso | F1 paper | F1 nosso | Observação |
|---|---:|---:|---:|---:|---|
| Naive Bayes | 0.9618 | 0.9534 | 0.9627 | 0.9541 | Próximo |
| Dandelion-NB | 0.9868 | 0.9535 | 0.9873 | 0.9542 | Ganho não reproduzido |
| BERT | 0.9948 | 0.9937 | 0.9950 | 0.9939 | Alta fidelidade |
| DistilBERT | 0.9936 | 0.9923 | 0.9939 | 0.9926 | Alta fidelidade |

### Tabela 2 - Datasets

| Dataset | Tipo | Amostras | Features | Uso |
|---|---|---:|---|---|
| Baseline | E-mail texto | ~82k | texto/label | Reprodução |
| MeAJOR | E-mail multi-source | versão a registrar | texto/fonte/features | Dataset principal |
| PhiUSIIL | URL/tabular | 235k+ | URL/webpage features | URL/metadados |
| PhishFuzzer | E-mail enriquecido | a auditar | URL/anexo/intent | Robustez |
| E-PhishGen | E-mail moderno | a auditar | LLM/multilíngue | Teste externo |

### Tabela 3 - Ablation

| Configuração | F1 | MCC | AUC-PR | FPR@Recall>=98% | Custo/10k | Latência p95 |
|---|---:|---:|---:|---:|---:|---:|
| Text-only |  |  |  |  |  |  |
| URL-only |  |  |  |  |  |  |
| Text + URL |  |  |  |  |  |  |
| Text + URL + calibração |  |  |  |  |  |  |
| Sistema completo v1 |  |  |  |  |  |  |

### Tabela 4 - Simulação operacional

| Base rate phishing | Modelo | FP/10k | FN/10k | Custo/10k | Latência p95 |
|---:|---|---:|---:|---:|---:|
| 50% |  |  |  |  |  |
| 10% |  |  |  |  |  |
| 5% |  |  |  |  |  |
| 1% |  |  |  |  |  |
| 0.5% |  |  |  |  |  |

---

## 15. Figuras Planejadas

1. **Arquitetura SLM-first v1**  
   E-mail -> texto/URL -> modelos -> fusão -> calibração -> decisão.

2. **Fluxo experimental baseado em hipóteses**  
   Baseline -> MeAJOR -> URL/metadados -> fusão -> robustez -> base rates.

3. **Custo x desempenho**  
   Comparar NB, DistilBERT, BERT, compact encoder, híbrido.

4. **FPR@Recall>=98%**  
   Comparar text-only, URL-only e híbrido.

5. **Robustness gap**  
   Original vs AI-rewrite vs cross-source.

---

## 16. Checklist de Execução

### Fase 1 - Baseline congelado

- [x] Reproduzir Naive Bayes.
- [x] Reproduzir Dandelion-NB.
- [x] Reproduzir BERT.
- [x] Reproduzir DistilBERT.
- [x] Comparar com Tabela IV do paper.
- [x] Registrar resultados finais em tabela no repositório.
- [ ] Rodar novamente após validar ambiente Kaggle da máquina atual.

### Fase 2 - MeAJOR

- [x] Baixar versão reproduzível.
- [x] Auditar colunas, rótulos e fontes.
- [x] Criar split aleatório.
- [x] Criar split por fonte, se possível.
- [x] Rodar smoke test com NB/LR/SVM leves.
- [ ] Rodar experimento completo com NB, LR e SVM.
- [ ] Rodar DistilBERT e MiniLM.

### Fase 3 - URL/metadados

- [x] Implementar extrator de features de URL.
- [x] Implementar features de metadados disponíveis.
- [x] Rodar Logistic Regression com text-only, URL/meta-only e combinado.
- [ ] Testar PhiUSIIL como benchmark tabular.

### Fase 4 - Fusão e calibração

- [x] Criar fusão por média ponderada.
- [x] Criar stacking com Logistic Regression.
- [x] Calibrar scores.
- [x] Otimizar threshold por FPR@Recall>=98%.

### Fase 5 - Robustez e base rates

- [ ] Auditar PhishFuzzer/E-PhishGen.
- [x] Criar avaliação controlada de rewrite sanitizada.
- [x] Simular base rates.
- [x] Reportar FP/10k, FN/10k e latência.

### Fase 6 - Teste final

- [ ] Congelar arquitetura.
- [ ] Congelar thresholds.
- [ ] Rodar teste final.
- [ ] Aplicar bootstrap/McNemar.
- [ ] Escrever resultados e limitações.

---

## 17. Riscos e Mitigações

| Risco | Impacto | Mitigação |
|---|---|---|
| Baseline saturado | Dificil superar em accuracy | Usar FPR, custo, latência e robustez como contribuição |
| Dandelion não reproduzido | Fragiliza comparação com paper | Relatar honestamente e tratar como baseline não reprodutível |
| MeAJOR já vir pré-processado | Limita features/metadados | Registrar versão e complementar com PhiUSIIL/PhishFuzzer |
| Dataset sem metadados | Enfraquece H2 | Separar texto+URL de metadados completos |
| PhishFuzzer sintético demais | Robustez artificial | Usar como stress test, não como único resultado |
| Termo SLM ambíguo | Crítica de revisor | Definir como compact text encoders |
| LLM fallback caro | Baixa viabilidade | Fora do v1 |
| PT-BR pequeno | Resultado frágil | Usar como stress test, não conclusão principal |
| Split aleatório inflado | Métricas irreais | Usar split por fonte/cross-source |

---

## 18. Extensões Futuras

Estas ideias continuam relevantes, mas devem vir depois:

- contexto local/RAG com histórico de remetentes e domínios;
- LLM fallback seletivo para casos ambíguos;
- distilação LLM -> SLM;
- PT-BR como estudo principal;
- anexos, HTML e QR phishing com CIC-Trap4Phish;
- explicabilidade com SHAP/LIME;
- estudo de utilidade das explicações com usuários.

---

## 19. Contribuições Esperadas

O paper v1 deve reivindicar quatro contribuições:

1. **Reprodução crítica do baseline:** BERT/DistilBERT reproduzidos com alta fidelidade; Dandelion não reproduzido.
2. **Avaliação mais realista:** uso de dataset multi-source, split por fonte e base rates desbalanceados.
3. **Pipeline SLM-first híbrido:** compact encoder + URL/metadados + fusão calibrada.
4. **Métricas operacionais:** FPR@Recall>=98%, FP/10k, custo/10k, latência p95 e robustness gap.

---

## 20. Frase de Contribuição

> Diferentemente de trabalhos que avaliam phishing como uma tarefa balanceada de classificação textual, este trabalho propõe e avalia um pipeline SLM-first, URL-aware e calibrado para detecção de phishing por e-mail sob base rates realistas, ataques reescritos por IA e restrições operacionais de custo, latência e falso positivo.

---

## 21. Conclusão da Proposta

A estratégia mais forte é não prometer um sistema universal. O paper deve provar uma tese menor e mais defensável:

```text
Compact encoders + URL/metadados + calibração
podem ser mais adequados que modelos grandes isolados
quando avaliamos phishing como problema operacional,
desbalanceado e adversarial.
```

Se os experimentos confirmarem essa hipótese, teremos um paper mais forte que uma simples troca de modelo, porque a contribuição será metodológica, operacional e cientificamente defensável.
