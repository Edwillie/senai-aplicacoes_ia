# Aula 02 — KDD e Pipeline de Dados com VS Code

## Aplicações em Inteligência Artificial

Nesta prática, vamos retomar o mesmo raciocínio visto anteriormente no Orange Data Mining, mas agora implementando cada etapa diretamente em Python, no VS Code.

O fluxo continua sendo:

**Seleção → Pré-processamento → Transformação → Mineração → Interpretação**

A diferença é que, em vez de conectar widgets visualmente, vamos identificar no código onde cada etapa acontece, acompanhar os dados sendo transformados e observar como o modelo é treinado e avaliado.

O projeto usa um conjunto de dados que simula **inspeções de qualidade em uma indústria de autopeças**.

---

# 1. Antes de começar: Bronze, Prata e Ouro

Ao longo do projeto, vamos organizar os dados em três camadas: **Bronze, Prata e Ouro**.

Essa organização ajuda a separar claramente o dado original, o dado tratado e a informação já preparada para análise ou tomada de decisão.

## Camada Bronze — o dado como chegou

A camada **Bronze** representa o dado bruto, preservado o mais próximo possível da origem.

Neste projeto, ela é criada logo após a leitura do arquivo CSV, ainda na etapa de **Seleção** do KDD.

Exemplo:

```text
CSV original
    ↓
Camada Bronze
```

Na Bronze ainda podemos encontrar:

- duplicidades;
- valores ausentes;
- grafias inconsistentes;
- valores inválidos;
- categorias escritas de formas diferentes.

É importante preservarmos essa camada porque ela funciona como uma referência do dado original.

Se modificarmos o dado antes de salvar a Bronze, perdemos parte da rastreabilidade do que realmente foi recebido.

**Neste projeto:**

```text
data/raw/dados_qualidade_pecas_kdd.csv
                ↓
bronze_qualidade_pecas
```

---

## Camada Prata — o dado tratado

A camada **Prata** representa o dado depois das operações de limpeza e padronização.

Ela aparece principalmente na etapa de **Pré-processamento** do KDD.

Aqui vamos:

- remover duplicidades;
- padronizar textos;
- tratar valores numéricos inválidos;
- identificar valores impossíveis;
- tratar valores ausentes;
- deixar as categorias consistentes.

Exemplo:

```text
Linha C
linha c
LINHA C
```

passam a representar:

```text
linha c
```

Outro exemplo:

```text
Forn A
Fornecedor A
forn a
```

passam a representar:

```text
fornecedor a
```

Podemos pensar assim:

```text
Bronze
  ↓
limpeza e padronização
  ↓
Prata
```

A Prata ainda não é necessariamente a resposta final para o negócio. Ela é uma base mais confiável e consistente para as próximas etapas.

---

## Camada Ouro — informação preparada para uso

A camada **Ouro** representa dados já organizados para responder perguntas, alimentar indicadores, relatórios ou aplicações.

Neste projeto, ela aparece principalmente na etapa de **Interpretação** do KDD.

Depois que o modelo é treinado e avaliado, vamos gerar informações como:

- resultado das previsões;
- acurácia do modelo;
- falsos positivos e falsos negativos;
- importância dos atributos;
- taxa de retrabalho por linha;
- base refinada de lotes.

Podemos resumir o fluxo assim:

```text
BRONZE
Dado bruto preservado
        ↓
PRATA
Dado limpo e padronizado
        ↓
OURO
Informação preparada para análise e decisão
```

Uma forma simples de lembrar é:

> **Bronze preserva. Prata organiza. Ouro responde.**

Neste projeto, as três camadas são gravadas em um banco SQLite:

```text
database/aula2_pipeline.db
```

---

# 2. Estrutura do projeto

A pasta deve ficar aproximadamente assim:

```text
AULA2_KDD_VSCODE/
├── .venv/
├── .vscode/
│   └── settings.json
├── data/
│   ├── raw/
│   │   └── dados_qualidade_pecas_kdd.csv
│   └── processed/
├── database/
│   └── aula2_pipeline.db
├── src/
│   ├── pipeline_qualidade.py
│   └── inspecionar_banco.py
├── INICIAR_DASHBOARD.bat
├── requirements.txt
├── ROTEIRO_AULA.md
└── streamlit_app.py
```

Vamos manter **apenas este arquivo `.md`** como guia da prática.

---

# 3. Preparar o ambiente

## 3.1 Abrir o projeto

1. Abra o **VS Code**.
2. Vá em **File → Open Folder**.
3. Selecione a pasta:

```text
AULA2_KDD_VSCODE
```

4. Abra:

```text
Terminal → New Terminal
```

---

## 3.2 Conferir o Python

No terminal:

```bat
python --version
```

Neste projeto, Python 3.12 funciona normalmente.

---

## 3.3 Criar o ambiente virtual

Execute:

```bat
python -m venv .venv
```

O ambiente virtual cria um espaço isolado para as bibliotecas deste projeto.

Depois de criado, veremos uma pasta:

```text
.venv
```

---

## 3.4 Ativar o ambiente virtual

Se estivermos usando **Command Prompt (CMD)**:

```bat
.venv\Scripts\activate.bat
```

Se estivermos usando **PowerShell**:

```powershell
.\.venv\Scripts\Activate.ps1
```

Quando o ambiente estiver ativo, o início da linha deverá mostrar algo semelhante a:

```text
(.venv) C:\...\AULA2_KDD_VSCODE>
```

Podemos confirmar qual Python está sendo utilizado:

```bat
where python
```

O primeiro caminho deverá apontar para:

```text
AULA2_KDD_VSCODE\.venv\Scripts\python.exe
```

---

## 3.5 Instalar as dependências

Com `(.venv)` aparecendo no terminal:

```bat
python -m pip install -r requirements.txt
```

Podemos conferir rapidamente:

```bat
python -c "import pandas, sklearn, streamlit; print('Ambiente OK')"
```

Resultado esperado:

```text
Ambiente OK
```

---

# 4. Conhecer a situação-problema

Vamos trabalhar com registros de inspeção de qualidade de peças.

Abra:

```text
data/raw/dados_qualidade_pecas_kdd.csv
```

As colunas são:

| Coluna | O que representa |
|---|---|
| `id_lote` | identificador do lote |
| `linha` | linha de produção |
| `turno` | turno da produção |
| `fornecedor` | fornecedor da matéria-prima |
| `temp_forno_c` | temperatura do forno |
| `tempo_ciclo_s` | tempo de ciclo |
| `pressao_bar` | pressão do processo |
| `umidade_pct` | umidade |
| `resultado_inspecao` | resultado conhecido: `conforme` ou `retrabalho` |

Agora vamos procurar alguns problemas propositalmente inseridos no dataset:

- textos escritos com maiúsculas e minúsculas diferentes;
- `Forn A` e `Fornecedor A`;
- células vazias;
- temperatura de `310 °C`;
- tempo de ciclo negativo;
- valores incompatíveis em colunas numéricas;
- registros duplicados.

## Vamos pensar antes de executar

Quais desses valores podem ser entregues diretamente a um algoritmo?

Quais precisam ser tratados antes?

Esse é um ponto importante: **o modelo aprende a partir dos dados que fornecemos**. Se os dados estiverem inconsistentes, o resultado também poderá ser inconsistente.

---

# 5. Executar o pipeline completo

Agora vamos executar o projeto sem alterar o código.

No terminal:

```bat
python src\pipeline_qualidade.py
```

Devemos observar cinco blocos:

```text
KDD 1 — SELEÇÃO / CAMADA BRONZE
KDD 2 — PRÉ-PROCESSAMENTO / CAMADA PRATA
KDD 3 — TRANSFORMAÇÃO / PREPARAÇÃO DE X E y
KDD 4 — MINERAÇÃO / ÁRVORE DE DECISÃO
KDD 5 — INTERPRETAÇÃO / CAMADA OURO
```

Ao final, o banco deverá existir em:

```text
database/aula2_pipeline.db
```

Agora vamos voltar ao código e entender o que aconteceu em cada etapa.

---

# 6. KDD 1 — Seleção e Camada Bronze

Abra:

```text
src/pipeline_qualidade.py
```

Localize:

```python
def etapa_1_selecao_e_bronze(...):
```

O ponto principal é:

```python
bronze = pd.read_csv(RAW_PATH)
salvar_tabela(bronze, "bronze_qualidade_pecas", conn)
```

Aqui estamos fazendo duas coisas:

1. lendo o arquivo original;
2. preservando esse conteúdo na camada Bronze.

A Bronze ainda não corrige nada.

Ela mantém:

- duplicidades;
- valores ausentes;
- grafias inconsistentes;
- valores suspeitos.

Também mantemos:

```text
id_lote
```

porque ele é útil para rastrear qual registro estamos analisando.

## Vamos observar

No terminal, compare:

```text
Formato bruto
Valores ausentes
Duplicidades completas
```

Pergunta para pensarmos:

> Se corrigíssemos o CSV antes de salvar a Bronze, como saberíamos exatamente o que chegou da origem?

---

# 7. KDD 2 — Pré-processamento e Camada Prata

Agora localize:

```python
def etapa_2_preprocessamento(...):
```

Nesta etapa vamos transformar o dado bruto em uma base mais consistente.

---

## 7.1 Remover duplicidades

Observe:

```python
prata = prata.drop_duplicates().copy()
```

No nosso conjunto:

```text
Bronze: 99 registros
Prata: 96 registros
```

A diferença ocorre porque três registros estavam duplicados.

É importante notarmos que não estamos removendo linhas aleatoriamente. Estamos removendo repetições completas identificadas no conjunto.

---

## 7.2 Padronizar categorias

O projeto consolida valores equivalentes.

Exemplos:

```text
NOITE / noite / Noite
```

passam para:

```text
noite
```

Também:

```text
Forn A / Fornecedor A / forn a
```

passam para:

```text
fornecedor a
```

Sem essa padronização, o computador poderia interpretar:

```text
Forn A
```

e:

```text
Fornecedor A
```

como categorias diferentes.

Para nós o significado é o mesmo, mas para o algoritmo são textos diferentes.

---

## 7.3 Converter valores numéricos

Observe o uso de:

```python
pd.to_numeric(..., errors="coerce")
```

Quando encontramos algo incompatível com uma coluna numérica, como:

```text
erro
```

não queremos tratá-lo como uma medição válida.

Com:

```python
errors="coerce"
```

o valor incompatível passa a ser considerado ausente.

Assim podemos tratá-lo de forma explícita na próxima etapa.

---

## 7.4 Identificar valores impossíveis

Neste dataset também existem valores propositalmente incompatíveis com o processo, como:

```text
temperatura = 310 °C
tempo de ciclo negativo
pressão = 0
```

O código utiliza regras de validade para identificar essas situações.

Em um projeto real, esses limites não deveriam ser escolhidos arbitrariamente.

Eles precisam ser definidos a partir de:

- especificações do processo;
- engenharia;
- documentação do equipamento;
- especialistas da área.

É importante diferenciarmos:

```text
valor incomum
```

de:

```text
valor impossível ou inválido
```

Nem todo valor raro deve ser eliminado.

---

# 8. Valores ausentes e imputação

Depois das conversões e regras de validade, alguns valores passam a ser considerados ausentes.

Para os atributos numéricos, adotamos a **mediana**.

Observe:

```python
prata[coluna] = prata[coluna].fillna(
    prata[coluna].median()
)
```

No nosso experimento, aparecem valores como:

```text
temp_forno_c: 188.20
tempo_ciclo_s: 45.70
pressao_bar: 6.46
umidade_pct: 48.15
```

Esses são os valores usados para substituir as ausências numéricas.

## Por que não colocar zero?

Vamos considerar uma temperatura ausente.

Se escrevermos:

```text
0 °C
```

estaremos dizendo que a temperatura foi medida e o resultado foi zero.

# 9. Persistir a Camada Prata

Depois do tratamento, a base é salva no SQLite como:

```text
prata_qualidade_pecas
```

Também geramos:

```text
data/processed/qualidade_prata.csv
```

Agora temos:

```text
Bronze
99 registros brutos
        ↓
Pré-processamento
        ↓
Prata
96 registros tratados
```

---

# 10. KDD 3 — Transformação

Agora localize:

```python
def etapa_3_transformacao(...):
```

Até aqui estávamos limpando dados.

Agora precisamos transformá-los para o formato utilizado pelo algoritmo.

---

## 10.1 Separar X e y

Observe:

```python
y = prata["resultado_inspecao"].map(
    {"conforme": 0, "retrabalho": 1}
)
```

`y` representa a resposta que já conhecemos nos dados históricos.

Neste caso:

```text
0 = conforme
1 = retrabalho
```

Agora observe:

```python
atributos = prata.drop(
    columns=["id_lote", "resultado_inspecao"]
)
```

O `resultado_inspecao` não pode entrar como entrada porque ele é exatamente aquilo que queremos prever.

Também retiramos:

```text
id_lote
```

do conjunto de atributos preditivos.

## Por que retirar `id_lote`?

Porque ele é apenas um identificador.

Exemplo:

```text
Lote 2001
Lote 2002
Lote 2003
```

O número do lote não representa uma característica física do processo.

Queremos que o modelo aprenda com informações como:

```text
temperatura
tempo de ciclo
pressão
umidade
linha
turno
fornecedor
```

e não com o número usado para identificar cada registro.

---

# 11. Transformar categorias em números

Algoritmos de Machine Learning trabalham melhor com representações numéricas.

Por isso utilizamos:

```python
pd.get_dummies(...)
```

Uma categoria como:

```text
linha = linha a
```

pode gerar colunas semelhantes a:

```text
linha_linha a
linha_linha b
linha_linha c
```

Para um registro da Linha A:

```text
linha_linha a = 1
linha_linha b = 0
linha_linha c = 0
```

Esse processo também acontece com:

```text
turno
fornecedor
```

Depois dessa transformação, nosso conjunto passa a ter **15 atributos**.

---

# 12. Separar treino e teste

Observe:

```python
train_test_split(
    ...,
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

No nosso experimento:

```text
Treino: 72 registros
Teste: 24 registros
```

O conjunto de treino é utilizado para o algoritmo aprender.

O conjunto de teste fica separado para verificarmos depois se o modelo consegue classificar registros que não foram utilizados durante o aprendizado.

Podemos pensar assim:

```text
96 registros
     ↓
┌───────────────┬───────────────┐
│               │               │
72 treino       24 teste
│               │
aprendizado     avaliação
```

Pergunta importante:

> Se treinássemos e avaliássemos o modelo exatamente nos mesmos dados, como saberíamos se ele realmente aprendeu um padrão ou apenas se ajustou aos exemplos que já conhecia?

---

# 13. KDD 4 — Mineração

Agora chegamos à etapa em que o algoritmo aprende padrões.

Localize:

```python
modelo = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
```

e:

```python
modelo.fit(X_train, y_train)
```

Na aula anterior, no Orange, utilizamos o widget:

```text
Tree
```

Agora estamos fazendo algo equivalente por código:

```python
DecisionTreeClassifier(...)
modelo.fit(...)
```

O ponto central continua o mesmo:

> **Nós não escrevemos manualmente as regras de classificação.**

O algoritmo analisa os exemplos e encontra divisões que ajudam a separar:

```text
conforme
```

de:

```text
retrabalho
```

---

# 14. Visualizar as regras encontradas

Depois da execução, abra:

```text
data/processed/regras_arvore.txt
```

Vamos localizar os atributos utilizados nas divisões da árvore.

No nosso experimento, os atributos mais relevantes foram:

```text
tempo_ciclo_s
temp_forno_c
pressao_bar
```

É importante notarmos que isso descreve **esta árvore, neste conjunto de dados, com esta configuração**.

Não significa que todas as outras variáveis sejam irrelevantes em qualquer processo industrial.

---

# 15. KDD 5 — Interpretação

Treinar o modelo não encerra o trabalho.

Agora precisamos verificar se ele conseguiu classificar corretamente os registros do conjunto de teste.

No nosso experimento:

```text
24 registros de teste
22 previsões corretas
2 previsões incorretas
```

A acurácia foi:

```text
91,67%
```

Podemos interpretar:

> Aproximadamente 91,67% das classificações realizadas no conjunto de teste foram corretas.

Mas precisamos ir além desse número.

---

# 16. Matriz de confusão

O resultado aparece aproximadamente assim:

```text
                 PREVISTO
                 Conforme   Retrabalho

REAL
Conforme             15          1
Retrabalho             1          7
```

Vamos ler célula por célula.

---

## 16.1 Conforme → Conforme

```text
15
```

O registro era realmente:

```text
conforme
```

e o modelo previu:

```text
conforme
```

Temos um acerto.

---

## 16.2 Conforme → Retrabalho

```text
1
```

O registro era:

```text
conforme
```

mas o modelo previu:

```text
retrabalho
```

Esse caso é chamado de:

```text
falso positivo
```

No nosso contexto, uma peça boa poderia ser enviada desnecessariamente para retrabalho.

Possíveis consequências:

- custo adicional;
- nova inspeção;
- perda de tempo;
- atraso no processo.

---

## 16.3 Retrabalho → Conforme

```text
1
```

O registro realmente precisava de:

```text
retrabalho
```

mas o modelo previu:

```text
conforme
```

Esse caso é chamado de:

```text
falso negativo
```

No nosso cenário, esse erro merece bastante atenção porque um item que deveria ser corrigido poderia seguir adiante no processo.

---

## 16.4 Retrabalho → Retrabalho

```text
7
```

O registro precisava de:

```text
retrabalho
```

e o modelo identificou corretamente.

Temos outro acerto.

---

# 17. Acurácia não conta toda a história

Agora podemos entender por que não devemos observar apenas:

```text
91,67%
```

Dois modelos podem apresentar a mesma acurácia e cometer erros diferentes.

Neste problema, vamos comparar:

```text
Falso positivo
Peça conforme → retrabalho
```

com:

```text
Falso negativo
Peça de retrabalho → conforme
```

A pergunta passa a ser:

> Qual desses erros produz maior impacto no processo?

A resposta depende do contexto real do negócio.

Neste exemplo de inspeção de qualidade, liberar como conforme uma peça que deveria ir para retrabalho pode representar um risco maior do que enviar uma peça boa para uma inspeção adicional.

---

# 18. Importância dos atributos

O projeto também calcula a importância utilizada pela árvore.

No nosso resultado:

```text
tempo_ciclo_s     ≈ 53,5%
temp_forno_c      ≈ 27,6%
pressao_bar       ≈ 18,9%
```

Isso significa que, **nas divisões realizadas por esta árvore**, esses atributos tiveram maior participação.

Não devemos transformar esse resultado automaticamente em uma afirmação causal.

Por exemplo:

```text
tempo_ciclo_s foi o atributo mais importante na árvore
```

não significa necessariamente:

```text
tempo_ciclo_s é a causa do retrabalho
```

O modelo identifica padrões estatísticos no conjunto fornecido.

Causalidade exige investigação adicional.

---

# 19. Camada Ouro

Agora chegamos às informações preparadas para análise.

O projeto gera tabelas como:

```text
ouro_lotes_qualidade
ouro_indicadores_linha
ouro_resultado_teste
ouro_metricas_modelo
ouro_importancia_atributos
```

Também produz arquivos em:

```text
data/processed/
```

Nesta etapa já conseguimos responder perguntas como:

- qual foi a acurácia?
- quantos erros ocorreram?
- quais atributos tiveram maior participação na árvore?
- qual linha apresentou maior taxa de retrabalho?
- quais registros foram classificados incorretamente?

É aqui que a frase fica mais clara:

> **Bronze preserva. Prata organiza. Ouro responde.**

---

# 20. Taxa de retrabalho por linha

No nosso conjunto, encontramos aproximadamente:

```text
Linha C: 50,0%
Linha B: 32,0%
Linha A: 19,4%
```

Podemos afirmar:

> A Linha C apresentou a maior taxa de retrabalho neste conjunto de dados.

Mas não podemos afirmar apenas com esse resultado:

> A Linha C causa o retrabalho.

Essa diferença é importante.

O resultado nos mostra **onde investigar primeiro**.

A partir daqui poderíamos analisar:

- temperatura;
- tempo de ciclo;
- pressão;
- fornecedor;
- turno;
- condições do processo;
- manutenção dos equipamentos.

A informação da camada Ouro ajuda a direcionar novas perguntas.

---

# 21. Inspecionar o banco SQLite

Agora vamos observar as tabelas gravadas no banco.

Execute:

```bat
python src\inspecionar_banco.py
```

Devemos encontrar tabelas como:

```text
bronze_qualidade_pecas
prata_qualidade_pecas
ouro_importancia_atributos
ouro_indicadores_linha
ouro_lotes_qualidade
ouro_metricas_modelo
ouro_resultado_teste
```

O script também mostra amostras das camadas Bronze, Prata e Ouro no terminal.

---

# 22. Abrir o painel visual

O terminal é útil para acompanhar o processamento, mas também vamos visualizar os resultados em uma interface.

Execute:

```bat
python -m streamlit run streamlit_app.py
```

O navegador deverá abrir:

```text
http://localhost:8501
```

Também podemos iniciar pelo arquivo:

```text
INICIAR_DASHBOARD.bat
```

---

# 23. Navegar pelo painel

O painel está dividido em seis partes.

## 23.1 Visão geral

Primeiro vamos observar:

- quantidade de registros na Bronze;
- quantidade de registros na Prata;
- quantidade de registros no teste;
- acurácia.

Depois vamos relacionar os cards às cinco etapas do KDD.

---

## 23.2 Bronze → Prata

Aqui conseguimos comparar visualmente:

```text
dado bruto
```

com:

```text
dado tratado
```

Vamos observar principalmente:

- duplicidades;
- valores ausentes;
- padronização;
- redução de 99 para 96 registros.

---

## 23.3 Modelo e matriz

Nesta parte vamos observar:

- quantidade de previsões corretas;
- quantidade de previsões incorretas;
- falso positivo;
- falso negativo;
- atributos mais utilizados pela árvore.

Aqui conseguimos transformar a matriz de confusão em situações concretas do processo.

---

## 23.4 Onde o modelo errou?

Agora vamos observar os próprios registros classificados incorretamente.

Em vez de olhar apenas para:

```text
falso positivo = 1
falso negativo = 1
```

podemos ver os valores de:

- linha;
- turno;
- fornecedor;
- temperatura;
- tempo de ciclo;
- pressão;
- umidade.

Isso ajuda a investigar o comportamento do modelo.

---

## 23.5 Decisão de negócio

Nesta parte vamos visualizar:

- taxa de retrabalho por linha;
- linha com maior taxa;
- dados da camada Ouro.

É importante mantermos a diferença entre:

```text
encontrar uma associação
```

e:

```text
provar uma causa
```

O painel mostra onde podemos investigar. Ele não encerra a análise.

---

## 23.6 Revisão do fluxo

Na última parte vamos retomar o raciocínio completo:

```text
Problema
   ↓
Dado bruto
   ↓
Bronze
   ↓
Limpeza
   ↓
Prata
   ↓
Transformação
   ↓
Treinamento
   ↓
Avaliação
   ↓
Ouro
   ↓
Informação para análise e decisão
```

---

# 24. Vamos consolidar o que fizemos

Ao final desta prática, devemos conseguir identificar:

## Seleção

```text
Leitura do CSV
Preservação da Bronze
```

## Pré-processamento

```text
Duplicidades
Padronização
Validação
Ausências
Imputação
Prata
```

## Transformação

```text
Separação X e y
Retirada de id_lote
Codificação das categorias
Treino e teste
```

## Mineração

```text
DecisionTreeClassifier
fit()
```

## Interpretação

```text
Acurácia
Matriz de confusão
Importância dos atributos
Indicadores
Camada Ouro
```

---

# 25. Registro final da prática

Vamos registrar em um parágrafo de **5 a 8 linhas**:

1. Quais problemas encontramos no dado bruto?
2. O que foi tratado para formar a camada Prata?
3. Por que `id_lote` não foi utilizado como atributo preditivo?
4. Qual foi a acurácia obtida?
5. Quantos falsos positivos e falsos negativos apareceram?
6. Qual desses erros teria maior impacto neste cenário?
7. Qual decisão de pré-processamento precisaria ser validada com um especialista do processo antes de levar a solução para produção?

---

# 26. Desafio extra

Se terminarmos antes, vamos alterar:

```python
max_depth=3
```

Primeiro para:

```python
max_depth=2
```

Execute novamente:

```bat
python src\pipeline_qualidade.py
```

Depois teste:

```python
max_depth=5
```

Execute novamente.

Vamos comparar:

- acurácia;
- matriz de confusão;
- quantidade de erros;
- regras em `regras_arvore.txt`;
- quantidade de atributos com importância maior que zero.

Depois de cada execução, atualize o painel do Streamlit.

A pergunta final é:

> Qual configuração apresenta resultados mais adequados para este experimento e quais evidências sustentam essa conclusão?

---

# 27. Encerramento

Nesta prática percorremos o fluxo completo:

```text
Dado bruto
   ↓
Seleção
   ↓
Bronze
   ↓
Pré-processamento
   ↓
Prata
   ↓
Transformação
   ↓
Mineração
   ↓
Modelo
   ↓
Interpretação
   ↓
Ouro
```

Na aula anterior, esse processo foi visualizado por meio de widgets.

Agora implementamos o mesmo raciocínio diretamente em código.

O próximo passo é avançarmos da preparação e mineração dos dados para modelos de Inteligência Artificial mais aprofundados.
