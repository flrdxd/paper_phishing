# Instrucoes de Execucao do Paper

Este documento define como rodar, registrar e interpretar os experimentos do
projeto.

O documento-base do nosso paper e:

```text
docs/proposta_paper_slm_phishing.md
```

Ele contem a pergunta de pesquisa, as hipoteses, os datasets, os modelos e os
criterios de decisao. Este arquivo aqui e o runbook operacional: o que fazer,
em qual ordem, quais comandos executar e onde salvar/verificar os resultados.

## 1. Ideia Geral

Existem dois blocos diferentes:

| Bloco | Comando principal | Objetivo |
|---|---|---|
| Baseline do PDF | `python3 run.py train` | Reproduzir o paper em `docs/Artigo_Phishing.pdf` |
| Nosso paper | `python3 run.py experiment ...` | Testar as hipoteses da proposta SLM-first |

O baseline do PDF nao deve ser misturado com os experimentos novos. Ele serve
como ponto de comparacao.

## 2. Preparar Ambiente

Ative o ambiente e instale o projeto:

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

Validar o projeto:

```bash
python3 run.py test
python3 run.py check
```

Observacao:

- `run.py test` valida codigo, auditoria de dados e experimentos.
- `run.py check` valida estrutura e Kaggle.
- O baseline do PDF precisa do Kaggle configurado.
- Os experimentos MeAJOR baixam dados do Zenodo e nao dependem do Kaggle.

## 3. Onde Cada Coisa Fica

O repositorio deve ficar limpo. Tudo que for gerado em execucao fica em
`.artifacts/`, que nao deve ser commitado.

Principais diretorios:

```text
.artifacts/data/       datasets baixados ou cacheados
.artifacts/models/     modelos treinados
.artifacts/plots/      graficos e figuras
.artifacts/results/    metricas, resumos e metadados
.artifacts/logs/       logs de execucao
```

Resultados por execucao ficam em:

```text
.artifacts/results/runs/<experimento>/<timestamp>/
```

Tabela agregada dos experimentos do nosso paper:

```text
.artifacts/results/research/all_experiments_metrics.csv
```

Resumos prontos para analise:

```text
.artifacts/results/research/summaries/
```

## 4. Ordem Recomendada

### Passo 1 - Validacao rapida

```bash
python3 run.py test
python3 run.py check
```

Se `run.py check` falhar por Kaggle, resolva antes de rodar o baseline do PDF.
Se o objetivo for apenas testar MeAJOR, pode continuar mesmo sem Kaggle.

### Passo 2 - Baseline do PDF

```bash
python3 run.py train
```

Este comando roda:

- Naive Bayes;
- Naive Bayes + Dandelion;
- BERT;
- DistilBERT.

Ele deve ser usado para comparar com `docs/Artigo_Phishing.pdf`.

Resultados principais:

```text
.artifacts/results/model_results.csv
.artifacts/results/model_results.json
.artifacts/results/results_summary.txt
.artifacts/results/runs/baseline_paper/<timestamp>/
```

Ao terminar, registre no caderno de analise:

- data/hora da execucao;
- commit usado;
- GPU/CPU;
- tamanho final do dataset;
- metricas de cada modelo;
- diferencas em relacao ao PDF.

### Passo 3 - Smoke test do nosso paper

```bash
python3 run.py experiment paper-v1 --profile smoke
```

Esse comando testa rapidamente a infraestrutura do nosso paper com uma amostra
menor. Ele roda H1, H2, H3, H5 e H6. Ele nao roda H4 porque H4 baixa e treina
transformers compactos.

Use o smoke para responder:

- os comandos estao funcionando?
- os resultados estao sendo salvos?
- os splits estao coerentes?
- existe queda entre split aleatorio e split por fonte?

### Passo 4 - Experimentos individuais

H1 - generalizacao no MeAJOR:

```bash
python3 run.py experiment h1-meajor --models nb,logreg,svm --split random
python3 run.py experiment h1-meajor --models nb,logreg,svm --split leave-one-source-out
```

H2 - ablation de texto versus URL/metadados:

```bash
python3 run.py experiment h2-url-meta --split random
python3 run.py experiment h2-url-meta --split leave-one-source-out
```

H3 - fusao e calibracao:

```bash
python3 run.py experiment h3-fusion --split random
```

H4 - encoders compactos:

```bash
python3 run.py experiment h4-compact-encoders --models distilbert,minilm --epochs 1
```

H5 - robustez contra reescrita controlada:

```bash
python3 run.py experiment h5-robustness
```

H6 - simulacao com base rates realistas:

```bash
python3 run.py experiment h6-base-rates --split random
```

Para testes rapidos, use `--sample-size 1000`.

Exemplo:

```bash
python3 run.py experiment h2-url-meta --sample-size 1000 --split random
```

### Passo 5 - Suite completa

Quando os testes individuais estiverem coerentes:

```bash
python3 run.py experiment paper-v1 --profile full
```

Esse perfil e mais pesado e inclui H4. Use GPU se possivel.

### Passo 6 - Execucao confirmatoria

Depois de analisar a fase exploratoria, congele:

- modelos;
- datasets;
- splits;
- seeds;
- thresholds;
- metricas primarias;
- comandos finais.

Depois rode:

```bash
python3 run.py experiment paper-v1 --profile confirmatory
```

Essa execucao deve ser tratada como resultado final do paper, nao como mais uma
tentativa exploratoria.

## 5. O Que Salvar Depois de Cada Rodada

Para cada rodada importante, guardar:

```text
.artifacts/results/runs/<experimento>/<timestamp>/metrics.csv
.artifacts/results/runs/<experimento>/<timestamp>/metrics.json
.artifacts/results/runs/<experimento>/<timestamp>/metadata.json
.artifacts/results/runs/<experimento>/<timestamp>/command.txt
.artifacts/results/runs/<experimento>/<timestamp>/summary.md
```

Tambem conferir:

```text
.artifacts/results/research/all_experiments_metrics.csv
.artifacts/results/research/summaries/*.csv
```

Os arquivos em `.artifacts/` nao entram no git. Para preservar resultados
importantes, compacte a pasta da rodada ou copie os CSVs finais para um local de
backup fora do repositorio.

## 6. Como Analisar

A analise deve seguir `docs/proposta_paper_slm_phishing.md`.

Perguntas principais:

1. O baseline do PDF foi reproduzido de forma aceitavel?
2. O split por fonte derruba as metricas em relacao ao split aleatorio?
3. URL/metadados reduzem falsos positivos mantendo recall alto?
4. A fusao calibrada melhora `FPR@Recall>=98%`?
5. DistilBERT/MiniLM entregam bom desempenho com custo menor que BERT?
6. Reescritas controladas reduzem o desempenho de modelos text-only?
7. Em base rates realistas, qual modelo gera menos falsos positivos esperados?

Metricas que devem aparecer na analise:

- Accuracy;
- Precision;
- Recall;
- F1;
- MCC;
- ROC-AUC quando disponivel;
- AUC-PR quando disponivel;
- FPR;
- FNR;
- `FPR@Recall>=98%`;
- matriz de confusao;
- tempo de treino;
- tempo de inferencia;
- custo operacional estimado em H6.

## 7. Decisoes Esperadas

Depois dos experimentos, atualizar a proposta com uma decisao objetiva:

| Evidencia | Decisao |
|---|---|
| Split por fonte cai muito | Generalizacao vira eixo central |
| URL/metadados reduzem FPR | Manter ramo tabular |
| Fusao calibrada melhora FPR com recall alto | Manter fusao/calibracao |
| H4 entrega resultado proximo com menor custo | Priorizar SLM/compact encoder |
| H5 mostra queda forte com reescrita | Robustez vira contribuicao importante |
| H6 mostra muitos falsos positivos em base rate baixo | Priorizar metrica operacional, nao so accuracy |

Se um componente nao melhorar resultado de forma clara, ele deve sair do nucleo
do paper v1.

## 8. Checklist Antes de Escrever o Paper

- [ ] `python3 run.py test` passou.
- [ ] Baseline do PDF foi executado com Kaggle configurado.
- [ ] `paper-v1 --profile smoke` passou.
- [ ] Experimentos individuais H1-H6 foram revisados.
- [ ] H4 foi executado pelo menos uma vez em GPU ou ambiente adequado.
- [ ] Resultados finais foram salvos fora de `.artifacts/` ou em backup.
- [ ] `docs/proposta_paper_slm_phishing.md` foi atualizado com as decisoes.
- [ ] Tabelas finais foram geradas a partir dos CSVs de resumo.
- [ ] A conclusao do paper foi escrita com base nos resultados, nao definida antes.

## 9. Diario de Resultados

Este runbook explica o que fazer e como fazer. Ele nao substitui o registro dos
resultados observados depois das execucoes.

Depois de rodar os experimentos, crie ou atualize um arquivo de resultados, por
exemplo:

```text
docs/RESULTADOS_EXPERIMENTAIS.md
```

Esse arquivo deve registrar:

- comando executado;
- data/hora;
- commit usado;
- ambiente de execucao;
- dataset e split;
- caminho da pasta em `.artifacts/results/runs/`;
- metricas principais;
- observacoes sobre falhas, anomalias ou diferencas;
- decisao tomada para a hipotese testada.

A proposta em `docs/proposta_paper_slm_phishing.md` deve ser atualizada apenas
com as conclusoes e decisoes consolidadas. O diario de resultados deve guardar o
historico detalhado das rodadas.
