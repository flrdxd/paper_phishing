# 📚 GUIA DE CONFIGURAÇÃO DE DADOS REAIS

## 🚨 PROBLEMA RESOLVIDO

O sistema detectou e removeu dados sintéticos que causavam accuracy artificial.
Agora você precisa configurar dados reais do Kaggle.

---

## 🔧 PASSO 1: CONFIGURAR API KAGGLE

### 1.1 Criar conta Kaggle
1. Acesse: https://www.kaggle.com/
2. Crie uma conta (gratuita)
3. Faça login

### 1.2 Obter API Token
1. Vá para: https://www.kaggle.com/settings
2. Role até "API" section
3. Clique em "Create New Token"
4. Baixe o arquivo `kaggle.json`

### 1.3 Instalar API Token
```bash
# No Linux/Mac:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# No Windows:
# Crie pasta: C:\Users\<SeuUsuario>\.kaggle
# Mova kaggle.json para essa pasta
```

### 1.4 Verificar instalação
```bash
source venv/bin/activate
pip install kagglehub
kaggle datasets list
```

Se funcionar, a API está configurada corretamente!

---

## 📥 PASSO 2: BAIXAR DADOS REAIS

### Opção A: Automático (recomendado)
```bash
cd /home/kalleb/Projects/Paper
source venv/bin/activate
python src/data_preprocessing.py
```

O sistema vai:
1. Detectar que não há dados em cache
2. Baixar dataset real do Kaggle
3. Executar quality checks
4. Salvar dados em cache

### Opção B: Manual (para debug)
```bash
# Baixar dataset de phishing
kaggle datasets download -d subhajournal/phishingemails
unzip phishingemails.zip -d data/

# Baixar dataset de spam/ham
kaggle datasets download -d uciml/sms-spam-collection-dataset
unzip sms-spam-collection-dataset.zip -d data/
```

---

## ✅ PASSO 3: VERIFICAR QUALIDADE DOS DADOS

```bash
python test_data_quality.py
```

**Resultado esperado:**
```
✓ Dataset appears to be real
✓ Risk Level: LOW or MEDIUM (not CRITICAL)
✓ Text diversity > 70%
✓ Template repetition < 10%
✓ Vocabulary > 2000 words
```

Se aparecer "CRITICAL", os dados ainda são sintéticos ou têm problemas.

---

## 🚀 PASSO 4: EXECUTAR PIPELINE COM DADOS REAIS

```bash
# Versão scikit-learn only (recomendado para começar)
python src/main_sklearn_only.py

# OU versão completa (requer PyTorch/GPU)
python src/main.py
```

---

## 📊 RESULTADOS ESPERADOS COM DADOS REAIS

### Métricas Realistas:
- **Accuracy:** 85-98% (não 100%)
- **Tempo de treino:** 10-60 segundos (não 1 segundo)
- **Diversidade textual:** >70%
- **Vocabulário:** >2000 palavras únicas
- **Cross-validation:** Std > 0.01 (variação real)

### Erros Esperados:
- Falsos positivos (emails legítimos classificados como phishing)
- Falsos negativos (phishing não detectados)
- Ambiguidade em casos de borda

### Features Importantes:
- Palavras contextualmente relevantes
- Padrões linguísticos complexos
- Não apenas palavras "mágicas" triviais

---

## 🐛 SOLUÇÃO DE PROBLEMAS

### Erro: "401 Unauthorized"
**Problema:** API token incorreto ou expirado
**Solução:** Recriar token no Kaggle e reinstalar

### Erro: "Dataset not found"
**Problema:** Nome do dataset incorreto
**Solução:** Verificar nome correto no Kaggle

### Erro: "Connection timeout"
**Problema:** Problema de conexão
**Solução:** Verificar internet, tentar novamente

### Erro: "Synthetic dataset detected"
**Problema:** Dados em cache são sintéticos
**Solução:** Deletar cache e baixar novamente
```bash
rm -f data/phishing_emails.csv data/legitimate_emails.csv
python src/data_preprocessing.py
```

### Erro: "ModuleNotFoundError: kagglehub"
**Problema:** kagglehub não instalado
**Solução:** `pip install kagglehub`

---

## 🔍 VERIFICAÇÃO DE SAÚDE DO SISTEMA

Execute este checklist antes de treinar:

```bash
# 1. Verificar ambiente virtual
source venv/bin/activate

# 2. Verificar dependências
pip list | grep -E "(pandas|scikit|kaggle)"

# 3. Verificar API Kaggle
kaggle datasets list | head -5

# 4. Verificar dados em cache
ls -lh data/

# 5. Executar quality checks
python test_data_quality.py

# 6. Verificar que Risk Level não é CRITICAL
```

---

## 📈 MONITORAMENTO DURANTE O TREINO

### Sinais de ✅ SUCESSO:
- Tempo de treino > 10 segundos
- Accuracy < 99%
- Erros na confusion matrix
- Cross-validation com variação
- Features importantes fazem sentido

### Sinais de ❌ PROBLEMA:
- Tempo de treino < 2 segundos
- Accuracy = 100%
- Confusion matrix perfeita
- Cross-validation sem variação
- Features triviais ("urgent", "bank")

Se detectar sinais de problema:
1. Pare o treino imediatamente
2. Execute `python test_data_quality.py`
3. Verifique se dados são sintéticos
4. Delete cache e baixe novamente

---

## 🎯 PRÓXIMOS PASSOS

1. ✅ Configurar API Kaggle
2. ✅ Baixar dados reais
3. ✅ Verificar qualidade dos dados
4. ✅ Executar pipeline
5. ✅ Analisar resultados
6. ✅ Comparar com literatura
7. ✅ Documentar limitações

---

## 📚 RECURSOS ADICIONAIS

- **Kaggle API Docs:** https://github.com/Kaggle/kaggle-api
- **Phishing Dataset:** https://www.kaggle.com/datasets/subhajournal/phishingemails
- **Spam Dataset:** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- **Scikit-learn Docs:** https://scikit-learn.org/

---

**Última atualização:** 2026-05-12
**Status:** 🟡 Sistema pronto para uso com dados reais