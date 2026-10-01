"""
Sends every question in the eval set to the agent, extracts a number from
its answer, and compares it against the ground-truth expected_value
(with tolerance, since the agent may round).

Tracks two metrics:
  1. Accuracy: percentage of answers close to the expected value
  2. Tool-usage rate: percentage of answers where the agent actually called
     the SQL tool (if this is below 100%, some answer is likely hallucinated)

Handles two real-world constraints of a free-tier API:
  - Resume: if a previous run was interrupted, results already saved to
    disk are skipped on the next run instead of being re-answered.
  - Rate limits: short waits are retried automatically; a long wait stops
    the run cleanly (progress is already saved) instead of blocking for
    a long time or crashing with a traceback.

Usage:
  python -m evaluation.run_sql_eval            # run all remaining questions
  python -m evaluation.run_sql_eval --limit 10  # run at most 10 this session
"""
import argparse
import json
import re
import time

from groq import RateLimitError

from src.agent.orchestrator import ask_agent
from src.config import EVAL_DIR

RESULTS_DIR = EVAL_DIR.parent.parent / "evaluation" / "results"
RELATIVE_TOLERANCE = 0.02  # up to 2% difference is accepted (rounding)
MAX_RETRIES_PER_QUESTION = 3
MAX_INLINE_WAIT_SECONDS = 90  # only auto-retry for waits shorter than this


def _parse_wait_seconds(error_message: str, default: float = 30.0) -> float:
    """Extracts 'try again in Xm Ys' from a Groq error message, in seconds."""
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", error_message)
    if not match:
        return default
    minutes = float(match.group(1)) if match.group(1) else 0.0
    seconds = float(match.group(2))
    return minutes * 60 + seconds + 2  # small buffer


def extract_numbers(text: str) -> list[float]:
    """Extracts all numbers from an answer string (handles commas/negatives)."""
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    numbers = []
    for m in matches:
        cleaned = m.replace(",", "")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return numbers


def is_close_enough(expected: float, answer_numbers: list[float]) -> bool:
    """
    Passes if any number in the answer is within tolerance of expected.

    Also accepts a sign flip (e.g. trade_balance_bn = -17.58, but the agent
    might say "a trade deficit of 17.58 billion" -- that's correct, just
    phrased as a positive magnitude instead of a signed value).
    """
    if not answer_numbers:
        return False
    tolerance = max(abs(expected) * RELATIVE_TOLERANCE, 0.05)
    return any(
        abs(n - expected) <= tolerance or abs(n - (-expected)) <= tolerance
        for n in answer_numbers
    )


def _save_results(out_path, eval_items, results):
    correct_count = sum(1 for r in results if r["passed"])
    tool_used_count = sum(1 for r in results if r["used_tool"])
    n = len(results)
    accuracy = (correct_count / n * 100) if n else 0.0
    tool_usage_rate = (tool_used_count / n * 100) if n else 0.0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "accuracy_pct": round(accuracy, 2),
            "tool_usage_rate_pct": round(tool_usage_rate, 2),
            "total_questions": len(eval_items),
            "answered_so_far": n,
            "correct_count": correct_count,
            "results": results,
        }, f, indent=2)
    return accuracy, tool_usage_rate


def run_eval(limit: int | None = None):
    eval_path = EVAL_DIR / "sql_eval_set.json"
    if not eval_path.exists():
        print(f"Eval set not found: {eval_path}")
        print("Run this first: python -m evaluation.build_eval_set")
        return

    with open(eval_path) as f:
        eval_items = json.load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "sql_eval_results.json"

    # Resume from a previous run's partial results, if any, instead of
    # re-answering questions that already succeeded.
    results = []
    if out_path.exists():
        with open(out_path) as f:
            prev = json.load(f)
        results = prev.get("results", [])
        print(f"Resuming: {len(results)} questions already answered.")

    done_ids = {r["id"] for r in results}
    remaining = [item for item in eval_items if item["id"] not in done_ids]

    if limit is not None:
        remaining = remaining[:limit]

    if not remaining:
        print("All questions already answered.")
    else:
        print(f"{len(remaining)} questions to run this session.\n")

    stopped_early = False

    for item in remaining:
        print(f"[{len(results) + 1}/{len(eval_items)}] {item['question']}")

        answer_text, used_tool = None, False
        for attempt in range(1, MAX_RETRIES_PER_QUESTION + 1):
            try:
                response = ask_agent(item["question"], verbose=False)
                answer_text = response["answer"]
                used_tool = len(response["trace"]) > 0
                break
            except RateLimitError as e:
                wait_s = _parse_wait_seconds(str(e))
                if wait_s > MAX_INLINE_WAIT_SECONDS:
                    print(f"  Rate limit hit, wait is {wait_s:.0f}s -- too long to "
                          f"wait inline. Stopping here; progress is saved, "
                          f"just rerun this command later.")
                    stopped_early = True
                    break
                print(f"  Rate limit hit (attempt {attempt}/{MAX_RETRIES_PER_QUESTION}). "
                      f"Waiting {wait_s:.0f}s and retrying...")
                time.sleep(wait_s)
        else:
            print("  Max retries reached, skipping this question.")
            answer_text = ""

        if stopped_early:
            break

        answer_numbers = extract_numbers(answer_text or "")
        passed = is_close_enough(item["expected_value"], answer_numbers)

        results.append({
            "id": item["id"],
            "question": item["question"],
            "expected_value": item["expected_value"],
            "answer_text": answer_text,
            "used_tool": used_tool,
            "passed": passed,
        })

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] expected={item['expected_value']}, tool_used={used_tool}")

        # Save after every question so progress isn't lost on a crash or
        # rate limit.
        _save_results(out_path, eval_items, results)

    accuracy, tool_usage_rate = _save_results(out_path, eval_items, results)

    print("\n=== EVAL SUMMARY ===")
    print(f"Total questions: {len(eval_items)}")
    print(f"Answered so far: {len(results)}")
    print(f"Correct: {sum(1 for r in results if r['passed'])} ({accuracy:.1f}%)")
    print(f"Tool usage rate: {tool_usage_rate:.1f}%")
    print(f"\nDetailed results saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of (unanswered) questions to run in this session.",
    )
    args = parser.parse_args()
    run_eval(limit=args.limit)
