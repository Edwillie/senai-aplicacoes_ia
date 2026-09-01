# Aula 03 — Redes Neurais MLP aplicadas à Qualidade Industrial

## Aplicações em Inteligência Artificial

Nesta prática, vamos reaproveitar o pipeline de dados desenvolvido anteriormente e comparar três modelos aplicados ao mesmo problema de inspeção de qualidade:

- Árvore de Decisão;
- MLP com scikit-learn;  (Multi-Layer Perceptron, ou Perceptron Multicamadas)
- MLP com TensorFlow/Keras.  (Multi-Layer Perceptron, ou Perceptron Multicamadas)

O objetivo é perceber que o pipeline de preparação dos dados continua válido mesmo quando trocamos o algoritmo de aprendizagem.

Fluxo geral:

```text
Dado bruto
   ↓
Bronze
   ↓
Pré-processamento
   ↓
Prata
   ↓
Transformação
   ↓
Treino e teste
   ↓
Escalonamento
   ↓
Modelos
   ↓
Avaliação
   ↓
Ouro
   ↓
Dashboard
```

---

# 1. Estrutura do projeto

A pasta deve ficar aproximadamente assim:

```text
AULA3_RNA_MLP/
├── .venv/
├── .vscode/
├── data/
│   ├── raw/
│   │   └── dados_qualidade_pecas_kdd.csv
│   └── processed/
├── database/
├── src/
│   ├── inspecionar_banco.py
│   ├── pipeline_qualidade.py
│   └── pipeline_rna_mlp_aula3.py
├── web/
├── INICIAR_DASHBOARD.bat
├── requirements.txt
├── ROTEIRO_AULA.md
└── streamlit_app.py
```

O arquivo principal desta aula é:

```text
src/pipeline_rna_mlp_aula3.py
```

O painel visual é executado a partir de:

```text
streamlit_app.py
```

---

# 2. Preparar o ambiente

## 2.1 Abrir a pasta no VS Code

Abra o VS Code e selecione:

```text
File → Open Folder
```

Escolha a pasta do projeto:

```text
AULA3_RNA_MLP
```

Depois abra:

```text
Terminal → New Terminal
```

---

## 2.2 Conferir o Python

Execute:

```bat
python --version
```

O projeto foi preparado para Python 3.12.

---

## 2.3 Criar o ambiente virtual

Execute:

```bat
python -m venv .venv
```

Depois da criação, a pasta `.venv` aparecerá no projeto.

> No laboratório, utilizaremos os comandos sempre por meio de `python -m ...`, evitando depender do comando `pip` isolado.

---

## 2.4 Ativar o ambiente virtual

No Prompt de Comando (CMD):

```bat
.venv\Scripts\activate.bat
```

No PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Quando o ambiente estiver ativo, o terminal deverá mostrar algo semelhante a:

```text
(.venv) C:\IA\AULA3_RNA_MLP>
```

Podemos conferir qual Python está sendo utilizado:

```bat
where python
```

O primeiro caminho deve apontar para:

```text
AULA3_RNA_MLP\.venv\Scripts\python.exe
```

---

# 3. Instalar as dependências

Com `(.venv)` aparecendo no terminal, execute:

```bat
python -m pip install --upgrade pip setuptools wheel
```

Depois:

```bat
python -m pip install -r requirements.txt
```

Se for necessário instalar manualmente as principais bibliotecas:

```bat
python -m pip install pandas numpy scikit-learn tensorflow streamlit graphviz
```

Para conferir o ambiente:

```bat
python -c "import pandas, sklearn, tensorflow, streamlit; print('Ambiente OK')"
```

Resultado esperado:

```text
Ambiente OK
```

Também podemos conferir especificamente o TensorFlow:

```bat
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

# 4. Atenção ao TensorFlow no Windows

O TensorFlow possui uma estrutura interna de arquivos extensa. Em caminhos muito longos do Windows, a instalação pode falhar.

Por isso, para esta prática, prefira um caminho curto, por exemplo:

```text
C:\IA\AULA3_RNA_MLP
```

Evite instalar o projeto em caminhos muito extensos.

Mensagens informativas do TensorFlow sobre `oneDNN` ou ausência de GPU no Windows não impedem a execução desta prática. O treinamento pode ser realizado pela CPU.

---

# 5. Situação-problema

Continuaremos utilizando o conjunto de dados de inspeção de qualidade industrial.

Arquivo:

```text
data/raw/dados_qualidade_pecas_kdd.csv
```

A variável que queremos prever continua sendo:

```text
resultado_inspecao
```

Classes:

```text
0 = conforme
1 = retrabalho
```

O projeto mantém as etapas de limpeza e transformação usadas anteriormente.

---

# 6. Bronze, Prata e Ouro

## Bronze

Preserva o dado bruto recebido.

```text
CSV original
   ↓
Bronze
```

Nesta etapa ainda existem duplicidades, ausências e inconsistências.

## Prata

Representa o dado depois da limpeza e padronização.

```text
Bronze
   ↓
limpeza
   ↓
Prata
```

Nesta etapa são realizadas operações como:

- remoção de duplicidades;
- padronização de categorias;
- conversão de valores numéricos;
- validação de medições;
- tratamento de valores ausentes.

## Ouro

Organiza os resultados produzidos pelos modelos para análise.

Nesta aula, a camada Ouro passa a armazenar também a comparação entre os diferentes modelos.

Podemos resumir:

```text
Bronze preserva.
Prata organiza.
Ouro responde.
```

---

# 7. Transformação dos dados

A variável-alvo é convertida para:

```text
conforme = 0
retrabalho = 1
```

Os atributos categóricos são transformados por meio de `get_dummies()`.

O identificador:

```text
id_lote
```

não é utilizado como atributo preditivo.

O conjunto final possui 15 atributos.

---

# 8. Separar treino e teste

O projeto utiliza:

```python
train_test_split(
    ...,
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

No conjunto atual:

```text
96 registros tratados

72 registros de treino
24 registros de teste
```

Treino:

```text
dados utilizados para o algoritmo aprender
```

Teste:

```text
dados separados para avaliar a capacidade de generalização
```

---

# 9. StandardScaler

Antes de treinar as MLPs, os atributos são padronizados com:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

O `fit` ocorre apenas no conjunto de treino.

Isso evita que informações do conjunto de teste participem da preparação do modelo.

A Árvore de Decisão não exige essa padronização, mas redes neurais normalmente se beneficiam de atributos em escalas comparáveis.

---

# 10. Executar o pipeline da Aula 03

Execute:

```bat
python src\pipeline_rna_mlp_aula3.py
```

O terminal deverá mostrar os principais blocos:

```text
KDD 1 - SELEÇÃO / CAMADA BRONZE

KDD 2 - PRÉ-PROCESSAMENTO / CAMADA PRATA

KDD 3 - TRANSFORMAÇÃO / X E y

AULA 03 - ESCALONAMENTO PARA A MLP

BASELINE - ÁRVORE DE DECISÃO DA AULA 02

AULA 03 - MLP COM SCIKIT-LEARN

AULA 03 - MLP COM TENSORFLOW / KERAS

KDD 5 - INTERPRETAÇÃO / CAMADA OURO
```

---

# 11. Modelo 1 — Árvore de Decisão

A Árvore de Decisão é mantida como referência para comparação.

Ela funciona como nosso modelo baseline.

No experimento atual:

```text
Acurácia: 91,67%

Falso positivo: 1
Falso negativo: 1
```

Esse resultado será comparado com os resultados das redes neurais.

---

# 12. Modelo 2 — MLP com scikit-learn

A primeira rede neural é criada com:

```python
MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    ...
)
```

Arquitetura simplificada:

```text
15 atributos
     ↓
16 neurônios
     ↓
8 neurônios
     ↓
saída
```

No experimento atual:

```text
Acurácia: 70,83%

Falso positivo: 3
Falso negativo: 4
```

---

# 13. Modelo 3 — MLP com TensorFlow/Keras

A segunda rede neural utiliza TensorFlow/Keras.

Estrutura:

```python
keras.Sequential([
    keras.Input(shape=(15,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
```

Podemos visualizar:

```text
15 atributos
     ↓
Dense(16)
ReLU
     ↓
Dense(8)
ReLU
     ↓
Dense(1)
Sigmoid
     ↓
Conforme / Retrabalho
```

A saída utiliza `sigmoid` porque estamos tratando um problema de classificação binária.

No experimento atual:

```text
Acurácia: 79,17%

Falso positivo: 1
Falso negativo: 4
```

---

# 14. Épocas de treinamento

Ao contrário da árvore, a rede neural ajusta seus pesos ao longo de várias épocas.

No experimento atual, o TensorFlow/Keras executou:

```text
88 épocas
```

O histórico é salvo em:

```text
data/processed/historico_treinamento_keras.csv
```

Nesse arquivo podemos acompanhar:

```text
loss
accuracy
val_loss
val_accuracy
```

---

# 15. EarlyStopping

O modelo Keras utiliza `EarlyStopping`.

A ideia é acompanhar o desempenho nos dados de validação e interromper o treinamento quando ele deixa de melhorar durante várias épocas.

Isso evita continuar treinando indefinidamente um modelo que já deixou de apresentar melhora relevante na validação.

---

# 16. Comparação dos modelos

No experimento atual:

| Modelo | Acurácia | Falso positivo | Falso negativo |
|---|---:|---:|---:|
| Árvore de Decisão | 91,67% | 1 | 1 |
| MLP TensorFlow/Keras | 79,17% | 1 | 4 |
| MLP scikit-learn | 70,83% | 3 | 4 |

A primeira conclusão importante é:

> Um modelo mais complexo não é automaticamente um modelo melhor.

Neste conjunto pequeno e tabular, a Árvore de Decisão apresentou melhor resultado.

---

# 17. Acurácia não é suficiente

Não devemos comparar os modelos apenas pela acurácia.

Precisamos observar também os tipos de erro.

Neste problema:

```text
Falso positivo
Conforme → Retrabalho
```

representa um lote conforme enviado desnecessariamente para retrabalho.

Já:

```text
Falso negativo
Retrabalho → Conforme
```

representa um lote que deveria receber retrabalho, mas foi classificado como conforme.

Neste cenário, o falso negativo merece atenção especial.

---

# 18. Resultados gerados

Após a execução, os arquivos são gravados em:

```text
data/processed/
```

Entre eles:

```text
qualidade_prata.csv
ouro_lotes_qualidade.csv
ouro_metricas_modelos.csv
ouro_resultados_teste.csv
ouro_comparacao_modelos.csv
historico_treinamento_keras.csv
```

O banco SQLite é salvo em:

```text
database/aula3_pipeline.db
```

---

# 19. Abrir o painel visual

Depois de executar o pipeline, inicie o Streamlit:

```bat
python -m streamlit run streamlit_app.py
```

O navegador deverá abrir em:

```text
http://localhost:8501
```

Se não abrir automaticamente, copie esse endereço para o navegador.

---

# 20. O que observar no dashboard

O painel permite visualizar:

- quantidade de registros na Bronze e na Prata;
- quantidade de registros usados no teste;
- comparação de acurácia entre os três modelos;
- falsos positivos e falsos negativos;
- matrizes de confusão;
- registros classificados incorretamente;
- arquitetura da MLP;
- evolução de `loss` e `accuracy` ao longo das épocas;
- comparação entre treinamento e validação.

---

# 21. Curvas de treinamento

No dashboard, observe principalmente:

```text
loss
val_loss
```

e:

```text
accuracy
val_accuracy
```

Essas curvas ajudam a entender se o modelo está melhorando durante o treinamento e como ele se comporta nos dados de validação.

Uma redução de `loss` no treinamento não garante, sozinha, bom desempenho em dados que o modelo nunca viu.

---

# 22. Reflexões da Aula 03

Ao final da prática, devemos conseguir explicar:

1. Por que o pipeline da Aula 02 continua sendo útil mesmo após a troca do algoritmo?
2. Por que aplicamos `StandardScaler` antes de treinar a MLP?
3. Qual é a diferença entre treino, validação e teste?
4. O que representa uma época de treinamento?
5. Por que a camada de saída usa `sigmoid`?
6. Por que a Árvore de Decisão apresentou melhor resultado neste experimento?
7. Por que não devemos escolher o modelo apenas pela acurácia?
8. Qual é o impacto de um falso negativo neste cenário?

---

# 23. Sequência de execução no laboratório

Sempre que o projeto já estiver configurado:

```bat
.venv\Scripts\activate.bat
```

Execute o pipeline:

```bat
python src\pipeline_rna_mlp_aula3.py
```

Depois abra o dashboard:

```bat
python -m streamlit run streamlit_app.py
```

Se for necessário recriar o ambiente:

```bat
python -m venv .venv

.venv\Scripts\activate.bat

python -m pip install --upgrade pip setuptools wheel

python -m pip install -r requirements.txt
```

Se as dependências precisarem ser instaladas manualmente:

```bat
python -m pip install pandas numpy scikit-learn tensorflow streamlit graphviz
```

---

# 24. Encerramento

Nesta prática, mantivemos o mesmo problema e o mesmo fluxo de preparação de dados, mas alteramos o mecanismo utilizado para aprender padrões.

Comparamos:

```text
Árvore de Decisão
        ×
MLP scikit-learn
        ×
MLP TensorFlow/Keras
```

O resultado mostra que a escolha do modelo precisa considerar o conjunto de dados, os tipos de erro e a capacidade de generalização.

O principal aprendizado não é apenas como construir uma rede neural, mas como integrá-la corretamente a um pipeline de dados, treiná-la, avaliar seus resultados e compará-la com outras alternativas.
