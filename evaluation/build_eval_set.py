"""
Eval set ko manually likhne ki jagah dataset se hi auto-generate karta hai:
random rows sample karke un par natural-language question banata hai, aur
ground-truth answer seedha DB se (SQL se) nikalta hai -- isse hum sure hote
hain ki expected value galat nahi hai.

Har template ek (question_format, column, unit) hai. Run karne par
N questions per template generate honge, total eval set data/eval/
sql_eval_set.json me save hoga.

Run: python -m evaluation.build_eval_set
"""
import json
import random

import duckdb

from src.config import DUCKDB_PATH, MACRO_TABLE_NAME, EVAL_DIR

RANDOM_SEED = 42
QUESTIONS_PER_TEMPLATE = 4

# (question_template, column, unit_label)
# question_template me {month_year} placeholder date ke liye hai
TEMPLATES = [
    ("What was the USD/INR exchange rate in {month_year}?", "usd_inr", "INR/USD"),
    ("What were India's forex reserves in {month_year}?", "reserves_bn", "billion USD"),
    ("What was the CPI inflation rate (YoY) in {month_year}?", "cpi_inflation_yoy", "%"),
    ("What was the Brent crude oil price in {month_year}?", "crude_brent_usd", "USD/barrel"),
    ("What were India's exports in {month_year}?", "exports_bn", "billion USD"),
    ("What were India's imports in {month_year}?", "imports_bn", "billion USD"),
    ("What was the trade balance in {month_year}?", "trade_balance_bn", "billion USD"),
    ("What was the net FII inflow in {month_year}?", "fii_net_cr", "INR crore"),
    ("What was the gold price in {month_year}?", "gold_inr_oz", "INR/oz"),
    ("What was India's import cover in {month_year}?", "import_cover_months", "months"),
]

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def build_eval_set(n_per_template=QUESTIONS_PER_TEMPLATE, seed=RANDOM_SEED):
    random.seed(seed)
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    eval_items = []
    item_id = 1

    for question_template, column, unit in TEMPLATES:
        # us column ke non-null rows me se random sample lo
        rows = con.execute(
            f'SELECT "date", "{column}" FROM {MACRO_TABLE_NAME} '
            f'WHERE "{column}" IS NOT NULL ORDER BY random() LIMIT {n_per_template}'
        ).fetchall()

        for date_val, expected_value in rows:
            month_year = f"{MONTH_NAMES[date_val.month - 1]} {date_val.year}"
            question = question_template.format(month_year=month_year)

            eval_items.append({
                "id": f"q{item_id:03d}",
                "question": question,
                "column": column,
                "unit": unit,
                "date": str(date_val.date()),
                "expected_value": round(float(expected_value), 4),
            })
            item_id += 1

    con.close()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "sql_eval_set.json"
    with open(out_path, "w") as f:
        json.dump(eval_items, f, indent=2)

    print(f"{len(eval_items)} eval questions generated -> {out_path}")
    return eval_items


if __name__ == "__main__":
    build_eval_set()
