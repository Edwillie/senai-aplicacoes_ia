
# LEIA-ME — Aula 05: Manutenção Preditiva com Redes Neurais

## Objetivo do projeto

Nesta prática, vamos utilizar dados de operação industrial para:

1. preparar uma base de dados de manutenção preditiva;
2. treinar uma Árvore de Decisão e duas redes neurais MLP;
3. comparar os resultados dos modelos;
4. usar um painel Streamlit para visualizar métricas;
5. inserir novos valores no simulador e receber uma previsão de possível falha.

---

## Organização esperada do projeto

A pasta deve possuir aproximadamente esta estrutura:

```text
AULA5_MANUTENCAO_PREDITIVA/
├── data/
│   └── raw/
│       └── ai4i2020.csv
├── src/
│   └── pipeline_mlp_aula5.py
├── LEIA-ME_AMBIENTE.md
├── requirements.txt
└── streamlit_app.py
```

Após a execução do pipeline, outras pastas serão criadas automaticamente:

```text
data/
└── processed/

database/
└── aula5_pipeline.db

modelos/
└── modelos_aula5.joblib
```

> Regra importante: todos os comandos deste roteiro devem ser executados a partir da pasta raiz do projeto, nunca de dentro de `src/`.

---

## Passo 1 — Confirmar que a pasta está correta

Abra a pasta do projeto no VS Code.

Depois abra:

```text
Terminal → New Terminal
```

No terminal, execute:

```bat
dir
```

Você deve visualizar, no mesmo nível:

```text
requirements.txt
streamlit_app.py
src
data
```

Também confirme que o dataset está neste local:

```text
data\raw\ai4i2020.csv
```

---

## Passo 2 — Instalar Python 3.12

Nas máquinas do laboratório existe Python 3.14. Porém, esta prática utiliza TensorFlow, que precisa ser instalado com Python 3.12.

Não desinstale nem altere o Python 3.14. O Python 3.12 será instalado ao lado dele e será usado apenas dentro do ambiente virtual deste projeto.

1. Baixe o instalador do Python 3.12:

   https://www.python.org/downloads/release/python-31210/

2. Escolha o arquivo:

   ```text
   Windows installer (64-bit)
   ```

3. Durante a instalação, marque:

   ```text
   Add python.exe to PATH
   ```

4. Não desinstale versões anteriores do Python.

Após instalar, feche e abra novamente o terminal do VS Code.

Execute:

```bat
py -3.12 --version
```

Resultado esperado:

```text
Python 3.12.10
```

Se necessário, execute:

```bat
py --list
```

Esse comando apresenta as versões de Python disponíveis na máquina.

---

## Passo 3 — Criar o ambiente virtual

Ainda na pasta raiz do projeto, execute:

```bat
py -3.12 -m venv .venv
```

Esse comando cria uma pasta `.venv` na raiz do projeto.

A estrutura ficará assim:

```text
AULA5_MANUTENCAO_PREDITIVA/
├── .venv/
├── data/
├── src/
├── requirements.txt
└── streamlit_app.py
```

> Não crie uma pasta `.venv` dentro de `src/`.

---

## Passo 4 — Ativar o ambiente virtual

No Prompt de Comando (CMD), execute:

```bat
.venv\Scripts\activate.bat
```

No PowerShell, execute:

```powershell
.\.venv\Scripts\Activate.ps1
```

Depois da ativação, o terminal deve iniciar com algo semelhante a:

```text
(.venv) C:\IA\AULA5_MANUTENCAO_PREDITIVA>
```

---

## Passo 5 — Confirmar a versão do Python no ambiente

Com `(.venv)` aparecendo no terminal, execute:

```bat
python --version
```

Resultado esperado:

```text
Python 3.12.10
```

Se aparecer Python 3.14, o ambiente virtual foi criado com a versão errada. Exclua somente a pasta `.venv` e repita o Passo 3:

```bat
rmdir /s /q .venv
py -3.12 -m venv .venv
```

---

## Passo 6 — Instalar as dependências

Com o ambiente virtual ativo, execute:

```bat
python -m pip install --upgrade pip setuptools wheel
```

Em seguida:

```bat
python -m pip install -r requirements.txt
```

A instalação pode demorar alguns minutos, principalmente por causa do TensorFlow.

---

## Passo 7 — Verificar o ambiente

Execute:

```bat
python -c "import pandas, sklearn, tensorflow, streamlit; print('Ambiente OK'); print('TensorFlow', __import__('tensorflow').__version__)"
```

Resultado esperado:

```text
Ambiente OK
TensorFlow 2.17.0
```

Mensagens sobre `oneDNN`, GPU ou CPU não impedem a atividade. Nesta aula, o treinamento pode ser executado pela CPU.

---

## Passo 8 — Executar o pipeline de manutenção preditiva

Com `(.venv)` ativo e o terminal na pasta raiz do projeto, execute:

```bat
python src\pipeline_mlp_aula5.py
```

O pipeline irá:

- ler o dataset `ai4i2020.csv`;
- criar a camada Bronze com os dados originais;
- criar a camada Prata com os dados preparados;
- separar os dados em treino e teste;
- treinar uma Árvore de Decisão;
- treinar uma MLP com scikit-learn;
- treinar uma MLP com TensorFlow/Keras;
- gerar métricas para comparação;
- salvar os modelos usados pelo simulador.

Ao final, procure esta mensagem:

```text
Modelos salvos para o simulador:
C:\IA\AULA5_MANUTENCAO_PREDITIVA\modelos\modelos_aula5.joblib
```

Esse arquivo é necessário para que o painel possa receber novos valores e realizar previsões.

---

## Passo 9 — Abrir o painel visual

Após o pipeline terminar sem erros, execute:

```bat
python -m streamlit run streamlit_app.py
```

O terminal exibirá um endereço semelhante a:

```text
Local URL: http://localhost:8501
```

Abra esse endereço no navegador.

Para encerrar o painel, volte ao terminal e pressione:

```text
Ctrl + C
```

---

## Passo 10 — Usar o Simulador de Manutenção Preditiva

No painel, abra a aba:

```text
Simulador
```

Informe valores hipotéticos de operação:

- variante de qualidade do produto;
- temperatura do ar;
- temperatura do processo;
- velocidade de rotação;
- torque;
- desgaste da ferramenta.

Em seguida, clique em:

```text
Executar previsão
```

O painel apresentará:

- a previsão da Árvore de Decisão;
- a previsão da MLP scikit-learn;
- a probabilidade estimada de falha pela MLP.

O objetivo é perceber que um modelo treinado pode receber novos dados e gerar uma estimativa de risco. Em uma operação real, a previsão apoiaria a decisão da equipe de manutenção; ela não substituiria inspeção técnica, análise do histórico do equipamento ou critérios de segurança.

---

## O que representam L, M e H?

As letras representam a variante de qualidade do produto presente no dataset:

- **L — Low:** qualidade baixa;
- **M — Medium:** qualidade média;
- **H — High:** qualidade alta.

Elas não representam baixo, médio ou alto risco de falha. São características do produto e do processo que o modelo analisa junto com temperatura, torque, velocidade de rotação e desgaste da ferramenta.

No dataset, as variantes também influenciam o desgaste: produtos **H, M e L** acrescentam, respectivamente, **5, 3 e 2 minutos** de desgaste à ferramenta durante o processo.

---

## Checklist final

- [ ] Projeto extraído em um caminho curto, preferencialmente `C:\IA\AULA5_MANUTENCAO_PREDITIVA`
- [ ] Arquivo `data\raw\ai4i2020.csv` está presente
- [ ] `py -3.12 --version` funciona
- [ ] `.venv` foi criado na raiz do projeto
- [ ] `(.venv)` aparece no terminal
- [ ] `python --version` mostra Python 3.12.x
- [ ] `python -m pip install -r requirements.txt` terminou sem erros
- [ ] `python src\pipeline_mlp_aula5.py` terminou sem erros
- [ ] A pasta `modelos` foi criada
- [ ] O arquivo `modelos\modelos_aula5.joblib` existe
- [ ] `python -m streamlit run streamlit_app.py` abriu o painel
- [ ] O simulador retornou uma previsão após o preenchimento dos campos
````
