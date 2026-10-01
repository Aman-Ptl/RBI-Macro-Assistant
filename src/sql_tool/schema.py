"""
Auto-generates a schema description from the DuckDB table (column names,
types, and optionally sample values) as an LLM-friendly text block.

This means columns never need to be typed out by hand -- whatever is in the
CSV is automatically reflected in the schema.
"""
import duckdb

from src.config import DUCKDB_PATH, MACRO_TABLE_NAME


def get_schema_description(
    db_path=DUCKDB_PATH,
    table_name=MACRO_TABLE_NAME,
    include_samples: bool = True,
    n_samples: int = 2,
) -> str:
    """
    Build an LLM-friendly schema description: column name + type, and
    optionally a couple of sample values per column.

    include_samples=False is used in the agent's system prompt to keep the
    token count low (this prompt is resent on every single API call, so
    trimming it directly extends how many questions fit in a rate-limited
    budget like Groq's free tier). The standalone CLI run below keeps
    samples on since it's just for human inspection.
    """
    con = duckdb.connect(str(db_path), read_only=True)

    columns_info = con.execute(f"DESCRIBE {table_name}").fetchall()
    # columns_info: [(column_name, column_type, null, key, default, extra), ...]

    row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    lines = [
        f"Table: {table_name}",
        f"Total rows: {row_count}",
        "",
        "Columns:",
    ]

    for col_name, col_type, *_ in columns_info:
        if not include_samples:
            lines.append(f'  - "{col_name}" ({col_type})')
            continue

        try:
            samples = con.execute(
                f'SELECT DISTINCT "{col_name}" FROM {table_name} '
                f'WHERE "{col_name}" IS NOT NULL LIMIT {n_samples}'
            ).fetchall()
            sample_vals = [str(s[0]) for s in samples]
            sample_str = ", ".join(sample_vals)
        except Exception:
            sample_str = "N/A"

        lines.append(f'  - "{col_name}" ({col_type})  e.g. {sample_str}')

    con.close()
    return "\n".join(lines)


if __name__ == "__main__":
    print(get_schema_description())
