const q = (selector, scope = document) => scope.querySelector(selector);
const qa = (selector, scope = document) => Array.from(scope.querySelectorAll(selector));

const fallback = {
  bronze: 99,
  prata: 96,
  teste: 24,
  acuracia: 0.916667,
  tn: 15,
  fp: 1,
  fn: 1,
  tp: 7,
  featureImportance: [
    { atributo: "tempo_ciclo_s", importancia: 0.534836 },
    { atributo: "temp_forno_c", importancia: 0.276358 },
    { atributo: "pressao_bar", importancia: 0.188806 }
  ],
  retrabalhoLinha: [
    { linha: "linha a", taxa: 0.193548 },
    { linha: "linha b", taxa: 0.320000 },
    { linha: "linha c", taxa: 0.500000 }
  ]
};

const kddSteps = {
  1: {
    eyebrow: "ETAPA 1",
    title: "Seleção: preservar a origem",
    body: "Começamos lendo o CSV e salvando uma cópia lógica do dado bruto. Ainda não corrigimos nada: primeiro preservamos a origem.",
    bullets: [
      "Leitura com pd.read_csv()",
      "Contagem de ausências e duplicidades",
      "Persistência em bronze_qualidade_pecas"
    ],
    code: `bronze = pd.read_csv(RAW_PATH)

salvar_tabela(
    bronze,
    "bronze_qualidade_pecas",
    conn
)`
  },
  2: {
    eyebrow: "ETAPA 2",
    title: "Pré-processamento: construir a Prata",
    body: "Agora tratamos problemas de qualidade do dado. Removemos duplicidades, padronizamos categorias, validamos números e imputamos ausências.",
    bullets: [
      "drop_duplicates() remove repetições completas",
      "pd.to_numeric(..., errors='coerce') trata valores incompatíveis",
      "fillna(median()) preserva registros sem transformar ausência em zero"
    ],
    code: `prata = bronze.drop_duplicates().copy()

prata[coluna] = pd.to_numeric(
    prata[coluna],
    errors="coerce"
)

prata[coluna] = prata[coluna].fillna(
    prata[coluna].median()
)`
  },
  3: {
    eyebrow: "ETAPA 3",
    title: "Transformação: preparar X e y",
    body: "Separamos as entradas do modelo e a resposta conhecida. Depois transformamos categorias em colunas numéricas e dividimos treino e teste.",
    bullets: [
      "X contém atributos preditivos",
      "y representa conforme ou retrabalho",
      "id_lote permanece como metadado, não como atributo de aprendizado"
    ],
    code: `y = prata["resultado_inspecao"].map({
    "conforme": 0,
    "retrabalho": 1
})

atributos = prata.drop(
    columns=["id_lote", "resultado_inspecao"]
)

X = pd.get_dummies(atributos)`
  },
  4: {
    eyebrow: "ETAPA 4",
    title: "Mineração: treinar a árvore",
    body: "Aqui o algoritmo analisa exemplos de treino e encontra divisões que ajudam a separar conforme de retrabalho. As regras não são escritas manualmente por nós.",
    bullets: [
      "DecisionTreeClassifier cria o algoritmo",
      "max_depth limita a complexidade da árvore",
      "fit() realiza o aprendizado com X_train e y_train"
    ],
    code: `modelo = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

modelo.fit(X_train, y_train)`
  },
  5: {
    eyebrow: "ETAPA 5",
    title: "Interpretação: transformar resultado em informação",
    body: "Depois do treinamento, avaliamos o conjunto de teste. Agora observamos acurácia, matriz de confusão, atributos utilizados e indicadores da camada Ouro.",
    bullets: [
      "accuracy_score() mede a proporção de acertos",
      "confusion_matrix() mostra como os erros aconteceram",
      "as tabelas Ouro organizam resultados para análise"
    ],
    code: `acuracia = accuracy_score(
    y_test,
    previsoes
)

matriz = confusion_matrix(
    y_test,
    previsoes
)`
  }
};

const matrixDescriptions = {
  tn: {
    title: "Conforme → Conforme",
    body: "O registro era realmente conforme e o modelo também previu conforme. Temos um acerto.",
    label: "ACERTO"
  },
  fp: {
    title: "Conforme → Retrabalho",
    body: "O registro era conforme, mas o modelo previu retrabalho. É um falso positivo: podemos gerar inspeção, custo ou retrabalho desnecessário.",
    label: "FALSO POSITIVO"
  },
  fn: {
    title: "Retrabalho → Conforme",
    body: "O registro precisava de retrabalho, mas o modelo previu conforme. É um falso negativo: um problema real pode seguir adiante no processo.",
    label: "FALSO NEGATIVO"
  },
  tp: {
    title: "Retrabalho → Retrabalho",
    body: "O registro precisava de retrabalho e o modelo identificou corretamente. Temos um acerto.",
    label: "ACERTO"
  }
};

function parseCSV(text) {
  const rows = [];
  let row = [];
  let value = "";
  let quoted = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"' && quoted && next === '"') {
      value += '"';
      i++;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(value);
      value = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i++;
      row.push(value);
      if (row.some(v => v !== "")) rows.push(row);
      row = [];
      value = "";
    } else {
      value += char;
    }
  }

  if (value || row.length) {
    row.push(value);
    rows.push(row);
  }

  if (!rows.length) return [];

  const headers = rows[0].map(h => h.trim());

  return rows.slice(1).map(values => {
    const obj = {};

    headers.forEach((header, index) => {
      obj[header] = (values[index] ?? "").trim();
    });

    return obj;
  });
}

async function fetchText(path) {
  const response = await fetch(path, { cache: "no-store" });

  if (!response.ok) {
    throw new Error(`Falha ao carregar ${path}`);
  }

  return response.text();
}

async function loadProjectData() {
  const status = q("#data-status");
  const text = q("#data-status-text");

  try {
    const [
      bronzeText,
      prataText,
      metricasText,
      importanciaText,
      linhasText,
      testeText
    ] = await Promise.all([
      fetchText("data/raw/dados_qualidade_pecas_kdd.csv"),
      fetchText("data/processed/qualidade_prata.csv"),
      fetchText("data/processed/ouro_metricas_modelo.csv"),
      fetchText("data/processed/ouro_importancia_atributos.csv"),
      fetchText("data/processed/ouro_indicadores_linha.csv"),
      fetchText("data/processed/ouro_resultado_teste.csv")
    ]);

    const bronze = parseCSV(bronzeText);
    const prata = parseCSV(prataText);
    const metricas = parseCSV(metricasText);
    const importancias = parseCSV(importanciaText);
    const linhas = parseCSV(linhasText);
    const teste = parseCSV(testeText);

    const metricMap = Object.fromEntries(
      metricas.map(row => [row.metrica, Number(row.valor)])
    );

    const loaded = {
      bronze: bronze.length,
      prata: prata.length,
      teste: teste.length,

      acuracia:
        metricMap.acuracia ??
        fallback.acuracia,

      tn:
        metricMap.verdadeiro_negativo ??
        fallback.tn,

      fp:
        metricMap.falso_positivo ??
        fallback.fp,

      fn:
        metricMap.falso_negativo ??
        fallback.fn,

      tp:
        metricMap.verdadeiro_positivo ??
        fallback.tp,

      featureImportance: importancias
        .map(row => ({
          atributo: row.atributo,
          importancia: Number(row.importancia)
        }))
        .filter(
          row =>
            Number.isFinite(row.importancia) &&
            row.importancia > 0
        )
        .sort(
          (a, b) =>
            b.importancia - a.importancia
        ),

      retrabalhoLinha: linhas
        .map(row => ({
          linha: row.linha,
          taxa: Number(row.taxa_retrabalho)
        }))
        .filter(
          row =>
            Number.isFinite(row.taxa)
        )
    };

    applyData(loaded);

    status.classList.add("ok");

    text.textContent =
      "Arquivos do projeto carregados. Os gráficos usam os resultados atuais.";
  } catch (error) {
    applyData(fallback);

    status.classList.add("demo");

    text.textContent =
      "Modo demonstrativo. Para ler os CSVs automaticamente, abra este guia por um servidor local.";
  }
}

function formatPct(value, digits = 1) {
  return `${(value * 100)
    .toFixed(digits)
    .replace(".", ",")}%`;
}

function setMetric(name, value) {
  qa(`[data-metric="${name}"]`).forEach(el => {
    el.textContent = value;
  });
}

function applyData(data) {
  setMetric(
    "bronze",
    String(data.bronze)
  );

  setMetric(
    "prata",
    String(data.prata)
  );

  setMetric(
    "teste",
    String(data.teste)
  );

  setMetric(
    "acuracia",
    formatPct(data.acuracia, 2)
  );

  setMetric(
    "tn",
    String(Math.round(data.tn))
  );

  setMetric(
    "fp",
    String(Math.round(data.fp))
  );

  setMetric(
    "fn",
    String(Math.round(data.fn))
  );

  setMetric(
    "tp",
    String(Math.round(data.tp))
  );

  const gauge = q("#accuracy-gauge");

  if (gauge) {
    gauge.style.setProperty(
      "--value",
      (data.acuracia * 100).toFixed(2)
    );
  }

  updateFeatureChart(
    data.featureImportance
  );

  updateLineChart(
    data.retrabalhoLinha
  );
}

function updateFeatureChart(items) {
  const root = q("#feature-chart");

  if (!root || !items.length) {
    return;
  }

  root.innerHTML = items
    .slice(0, 6)
    .map(item => {
      const pct =
        item.importancia * 100;

      return `
        <div class="bar-row">
          <span>${escapeHTML(item.atributo)}</span>

          <div>
            <i
              data-target-width="${pct.toFixed(2)}%"
              style="width:0"
            ></i>
          </div>

          <strong>
            ${pct
              .toFixed(1)
              .replace(".", ",")}%
          </strong>
        </div>
      `;
    })
    .join("");

  requestAnimationFrame(() => {
    qa(
      "i[data-target-width]",
      root
    ).forEach(bar => {
      bar.style.width =
        bar.dataset.targetWidth;
    });
  });
}

function updateLineChart(items) {
  const root = q("#line-chart");

  if (!root || !items.length) {
    return;
  }

  const max = Math.max(
    ...items.map(i => i.taxa),
    0.01
  );

  const sorted = [...items].sort(
    (a, b) =>
      a.linha.localeCompare(b.linha)
  );

  root.innerHTML = sorted
    .map(item => {
      const pct =
        item.taxa * 100;

      const height =
        (item.taxa / max) * 100;

      const emphasis =
        item.taxa === max
          ? " emphasis"
          : "";

      const label = item.linha
        .replace("linha ", "Linha ")
        .toUpperCase()
        .replace("LINHA", "Linha");

      return `
        <div
          class="column${emphasis}"
          data-label="${escapeHTML(label)}"
        >
          <strong>
            ${pct
              .toFixed(1)
              .replace(".", ",")}%
          </strong>

          <i
            data-target-height="${height.toFixed(2)}%"
            style="height:0"
          ></i>

          <span>
            ${escapeHTML(label)}
          </span>
        </div>
      `;
    })
    .join("");

  requestAnimationFrame(() => {
    qa(
      "i[data-target-height]",
      root
    ).forEach(bar => {
      bar.style.height =
        bar.dataset.targetHeight;
    });
  });
}

function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function setupLayerSwitcher() {
  qa(".layer-tab").forEach(button => {
    button.addEventListener(
      "click",
      () => {
        const layer =
          button.dataset.layer;

        qa(".layer-tab").forEach(
          b =>
            b.classList.toggle(
              "active",
              b === button
            )
        );

        qa(
          "[data-layer-panel]"
        ).forEach(panel => {
          panel.classList.toggle(
            "active",
            panel.dataset.layerPanel ===
              layer
          );
        });
      }
    );
  });
}

function setupCopyButtons() {
  qa(".copy-btn").forEach(button => {
    button.addEventListener(
      "click",
      async () => {
        const value =
          button.dataset.copy;

        try {
          await navigator.clipboard.writeText(
            value
          );
        } catch {
          const area =
            document.createElement(
              "textarea"
            );

          area.value = value;

          document.body.appendChild(
            area
          );

          area.select();

          document.execCommand(
            "copy"
          );

          area.remove();
        }

        showToast(
          "Comando copiado"
        );
      }
    );
  });
}

function showToast(message) {
  const toast =
    q("#toast");

  toast.textContent =
    message;

  toast.classList.add(
    "show"
  );

  clearTimeout(
    showToast.timer
  );

  showToast.timer =
    setTimeout(
      () =>
        toast.classList.remove(
          "show"
        ),
      1700
    );
}

function setupProblems() {
  const button =
    q("#reveal-problems");

  const table =
    q("#fake-table");

  button.addEventListener(
    "click",
    () => {
      const active =
        table.classList.toggle(
          "revealed"
        );

      button.textContent =
        active
          ? "Ocultar marcações"
          : "Mostrar problemas";
    }
  );
}

function setupSchema() {
  qa(
    "#schema-list button"
  ).forEach(button => {
    button.addEventListener(
      "click",
      () => {
        qa(
          "#schema-list button"
        ).forEach(
          b =>
            b.classList.remove(
              "active"
            )
        );

        button.classList.add(
          "active"
        );
      }
    );
  });
}

function setupKDD() {
  const detail =
    q("#kdd-detail");

  function renderStep(step) {
    const info =
      kddSteps[step];

    qa(
      ".kdd-node"
    ).forEach(node => {
      node.classList.toggle(
        "active",
        Number(
          node.dataset.step
        ) === step
      );
    });

    const text =
      q(
        ".kdd-detail-text",
        detail
      );

    text.innerHTML = `
      <span class="eyebrow">
        ${info.eyebrow}
      </span>

      <h3>
        ${info.title}
      </h3>

      <p>
        ${info.body}
      </p>

      <ul>
        ${info.bullets
          .map(
            item =>
              `<li>${item}</li>`
          )
          .join("")}
      </ul>
    `;

    q("#kdd-code").textContent =
      info.code;

    detail.classList.remove(
      "reveal"
    );

    void detail.offsetWidth;

    detail.classList.add(
      "reveal"
    );
  }

  qa(
    ".kdd-node"
  ).forEach(node => {
    node.addEventListener(
      "click",
      () =>
        renderStep(
          Number(
            node.dataset.step
          )
        )
    );
  });

  q("#run-tour").addEventListener(
    "click",
    () => {
      q("#kdd").scrollIntoView({
        behavior: "smooth"
      });

      let step = 1;

      renderStep(step);

      clearInterval(
        setupKDD.timer
      );

      setupKDD.timer =
        setInterval(() => {
          step += 1;

          if (step > 5) {
            clearInterval(
              setupKDD.timer
            );

            return;
          }

          renderStep(step);
        }, 1700);
    }
  );
}

function setupEncoding() {
  qa(
    "#line-selector button"
  ).forEach(button => {
    button.addEventListener(
      "click",
      () => {
        qa(
          "#line-selector button"
        ).forEach(
          b =>
            b.classList.remove(
              "active"
            )
        );

        button.classList.add(
          "active"
        );

        const line =
          button.dataset.line;

        q("#bin-a").textContent =
          line === "a"
            ? "1"
            : "0";

        q("#bin-b").textContent =
          line === "b"
            ? "1"
            : "0";

        q("#bin-c").textContent =
          line === "c"
            ? "1"
            : "0";
      }
    );
  });
}

function setupMatrix() {
  qa(
    ".matrix-cell"
  ).forEach(cell => {
    cell.addEventListener(
      "click",
      () => {
        qa(
          ".matrix-cell"
        ).forEach(
          c =>
            c.classList.remove(
              "active"
            )
        );

        cell.classList.add(
          "active"
        );

        const info =
          matrixDescriptions[
            cell.dataset.matrix
          ];

        const box =
          q(
            "#matrix-explanation"
          );

        box.innerHTML = `
          <span class="eyebrow">
            ${info.label}
          </span>

          <h3>
            ${info.title}
          </h3>

          <p>
            ${info.body}
          </p>
        `;

        box.classList.remove(
          "reveal"
        );

        void box.offsetWidth;

        box.classList.add(
          "reveal"
        );
      }
    );
  });
}

function setupDepth() {
  const slider =
    q("#depth-slider");

  const value =
    q("#depth-value");

  const description =
    q("#depth-description");

  const text = {
    2:
      "Profundidade 2: árvore mais simples. Podemos ganhar interpretabilidade, mas talvez perder capacidade de separar padrões mais complexos.",

    3:
      "Profundidade 3: configuração usada no experimento principal. Equilibra simplicidade e capacidade de separar padrões.",

    4:
      "Profundidade 4: permite mais divisões. Precisamos verificar se o ganho de ajuste realmente melhora a avaliação em dados de teste.",

    5:
      "Profundidade 5: árvore mais complexa. Maior complexidade não garante melhor generalização e pode aumentar o risco de sobreajuste."
  };

  slider.addEventListener(
    "input",
    () => {
      value.textContent =
        slider.value;

      description.textContent =
        text[slider.value];
    }
  );
}

function setupOpenStreamlit() {
  q(
    "#open-streamlit"
  ).addEventListener(
    "click",
    () => {
      window.open(
        "http://localhost:8501",
        "_blank",
        "noopener"
      );
    }
  );
}

function setupScrollProgress() {
  const progress =
    q("#page-progress");

  const label =
    q("#progress-label");

  const sections =
    qa("main section[id]");

  const links =
    qa(".nav-link");

  const onScroll = () => {
    const doc =
      document.documentElement;

    const max =
      doc.scrollHeight -
      doc.clientHeight;

    const pct =
      max > 0
        ? (
            doc.scrollTop /
            max
          ) * 100
        : 0;

    progress.style.width =
      `${pct}%`;

    label.textContent =
      `${Math.round(pct)}%`;

    let current =
      sections[0]?.id;

    sections.forEach(
      section => {
        const rect =
          section.getBoundingClientRect();

        if (
          rect.top <= 160
        ) {
          current =
            section.id;
        }
      }
    );

    links.forEach(link => {
      link.classList.toggle(
        "active",
        link.getAttribute(
          "href"
        ) ===
          `#${current}`
      );
    });
  };

  window.addEventListener(
    "scroll",
    onScroll,
    {
      passive: true
    }
  );

  onScroll();
}

function setupObserver() {
  const observer =
    new IntersectionObserver(
      entries => {
        entries.forEach(
          entry => {
            if (
              !entry.isIntersecting
            ) {
              return;
            }

            entry.target.classList.add(
              "reveal"
            );

            if (
              entry.target.id ===
              "ouro"
            ) {
              qa(
                "#feature-chart i[data-target-width]"
              ).forEach(bar => {
                bar.style.width =
                  bar.dataset.targetWidth;
              });

              qa(
                "#line-chart i[data-target-height]"
              ).forEach(bar => {
                bar.style.height =
                  bar.dataset.targetHeight;
              });
            }
          }
        );
      },
      {
        threshold: 0.12
      }
    );

  qa(".section").forEach(
    section =>
      observer.observe(
        section
      )
  );
}

document.addEventListener(
  "DOMContentLoaded",
  () => {
    setupLayerSwitcher();
    setupCopyButtons();
    setupProblems();
    setupSchema();
    setupKDD();
    setupEncoding();
    setupMatrix();
    setupDepth();
    setupOpenStreamlit();
    setupScrollProgress();
    setupObserver();
    loadProjectData();
  }
);