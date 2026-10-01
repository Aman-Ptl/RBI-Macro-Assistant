"""
Agent isi function ko "tool" ke roop me call karega. Ye SQL ko safely
execute karta hai -- sirf SELECT allow, row limit lagta hai, aur errors
ko clean message me wrap karta hai taaki agent retry kar sake.
"""
import re

import duckdb

from src.config import DUCKDB_PATH, MACRO_TABLE_NAME, SQL_ROW_LIMIT

# Sirf ye keywords allowed hain query ki shuruaat me
ALLOWED_START = ("select", "with")

# Ye keywords kahin bhi dikhe to reject -- data modify/delete karne wale
BLOCKED_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create",
    "truncate", "attach", "copy", "pragma", "install", "load",
)


class SQLGuardrailError(Exception):
    """Jab query safety check me fail ho jaye."""
    pass


def _validate_query(sql: str) -> None:
    cleaned = sql.strip().lower()

    if not cleaned.startswith(ALLOWED_START):
        raise SQLGuardrailError(
            "Sirf SELECT (ya WITH ... SELECT) queries allowed hain."
        )

    for kw in BLOCKED_KEYWORDS:
        # word-boundary check, taaki 'created_at' jaisa column name flag na ho
        if re.search(rf"\b{kw}\b", cleaned):
            raise SQLGuardrailError(
                f"Query me blocked keyword mila: '{kw}'. Sirf read-only SELECT chalega."
            )

    # ek query me ek hi statement allowed (semicolon-separated multi-statement block karo)
    if cleaned.rstrip(";").count(";") > 0:
        raise SQLGuardrailError("Ek baar me sirf ek SQL statement bhej sakte ho.")


def run_query(sql: str, db_path=DUCKDB_PATH, row_limit=SQL_ROW_LIMIT) -> dict:
    """
    SQL query ko validate karke execute karta hai.

    Returns dict:
      {"success": True, "columns": [...], "rows": [...], "row_count": N}
      ya
      {"success": False, "error": "..."}
    Agent isi dict ko dekh kar decide karega ki answer banana hai ya retry.
    """
    try:
        _validate_query(sql)
    except SQLGuardrailError as e:
        return {"success": False, "error": str(e)}

    # agar query me already LIMIT nahi hai, to apna row_limit laga do
    if "limit" not in sql.lower():
        sql = f"{sql.rstrip(';')} LIMIT {row_limit}"

    try:
        con = duckdb.connect(str(db_path), read_only=True)
        result = con.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        con.close()

        return {
            "success": True,
            "columns": columns,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
        }
    except Exception as e:
        return {"success": False, "error": f"SQL execution error: {e}"}


if __name__ == "__main__":
    # quick manual test
    test_sql = f"SELECT * FROM {MACRO_TABLE_NAME} LIMIT 5"
    print(run_query(test_sql))
