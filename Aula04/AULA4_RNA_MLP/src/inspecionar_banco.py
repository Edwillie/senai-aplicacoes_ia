from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "database" / "aula2_pipeline.db"

if not DB_PATH.exists():
    raise FileNotFoundError(
        "Banco ainda não existe. Execute primeiro: python src/pipeline_qualidade.py"
    )

with sqlite3.connect(DB_PATH) as conn:
    tabelas = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
    )["name"].tolist()

    print("Tabelas encontradas no SQLite:")
    for tabela in tabelas:
        qtd = pd.read_sql_query(f'SELECT COUNT(*) AS n FROM "{tabela}"', conn).iloc[0, 0]
        print(f"  - {tabela}: {qtd} linha(s)")

    print("\nAmostra da camada bronze:")
    print(pd.read_sql_query("SELECT * FROM bronze_qualidade_pecas LIMIT 6", conn).to_string(index=False))

    print("\nAmostra da camada prata:")
    print(pd.read_sql_query("SELECT * FROM prata_qualidade_pecas LIMIT 6", conn).to_string(index=False))

    print("\nIndicadores da camada ouro:")
    print(pd.read_sql_query("SELECT * FROM ouro_indicadores_linha", conn).to_string(index=False))

    print("\nMétricas do modelo:")
    print(pd.read_sql_query("SELECT * FROM ouro_metricas_modelo", conn).to_string(index=False))
