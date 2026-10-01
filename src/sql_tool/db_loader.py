"""
CSV ko DuckDB me load karta hai, ek baar chalana hai (ya jab dataset update ho).

Kyun DuckDB?
- Zero setup — ek file-based DB hai, Postgres jaisa server nahi chahiye
- Pandas se seedha CSV read kar sakta hai
- Fast analytical SQL (macro data jaisi wide tables ke liye acha)

Run: python -m src.sql_tool.db_loader
"""
import duckdb
import pandas as pd

from src.config import MASTER_CSV_PATH, DUCKDB_PATH, MACRO_TABLE_NAME


def load_csv_to_duckdb(csv_path=MASTER_CSV_PATH, db_path=DUCKDB_PATH, table_name=MACRO_TABLE_NAME):
    """CSV ko padhta hai aur DuckDB table me daal deta hai (overwrite karta hai)."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Master CSV nahi mila: {csv_path}\n"
            f"Check karo ki impp_master_v3.csv data/structured/ me hai."
        )

    df = pd.read_csv(csv_path)
    print(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    # Sirf explicitly known date columns ko datetime me convert karna.
    # Pehle "month" substring match kar rahe the, jo 'month_num' (1-12 ka int)
    # aur 'import_cover_months' (duration, date nahi) ko bhi galat convert kar
    # raha tha. Isliye ab whitelist use kar rahe hain -- agar CSV me koi aur
    # date-type column ho to yahan naam add kar dena.
    KNOWN_DATE_COLUMNS = ("date", "year_month")
    for col in df.columns:
        if col.lower() in KNOWN_DATE_COLUMNS:
            try:
                df[col] = pd.to_datetime(df[col], errors="raise")
                print(f"  -> '{col}' ko datetime me convert kiya")
            except (ValueError, TypeError):
                print(f"  -> '{col}' datetime me convert nahi ho paya, string hi rahega")

    con = duckdb.connect(str(db_path))
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.register("df_temp", df)
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_temp")

    row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"DuckDB me loaded: {row_count} rows table '{table_name}' me, DB file: {db_path}")

    con.close()
    return df.columns.tolist(), df.dtypes.to_dict()


if __name__ == "__main__":
    load_csv_to_duckdb()
