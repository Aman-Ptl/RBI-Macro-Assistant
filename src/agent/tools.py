"""
Tool definitions given to Groq (OpenAI-compatible function-calling format):
{"type": "function", "function": {name, description, parameters}}
-- this differs slightly from Anthropic's "input_schema" key, so switching
providers later would only mean restructuring these dicts, not the rest
of the agent logic.
"""

SQL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_macro_data",
        "description": (
            "Runs a read-only SQL SELECT query against the 'macro_data' DuckDB "
            "table. This table holds monthly Indian macroeconomic data from 1990 "
            "to 2026 (USD/INR rate, forex reserves, trade, crude oil, inflation, "
            "gold, BoP, etc). Use this for any number, date-specific value, "
            "trend, or comparison question -- never state a number from memory, "
            "always pull it from this tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": (
                        "A valid DuckDB SELECT query. Only SELECT or WITH...SELECT "
                        "will run. Use double quotes around column names if a "
                        "query fails (columns can be case-sensitive)."
                    )
                }
            },
            "required": ["sql"],
        },
    },
}

RAG_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_rbi_documents",
        "description": (
            "Searches RBI policy documents (MPC Minutes, RBI Bulletins, "
            "Monetary Policy Reports, State of the Economy articles) for "
            "qualitative context, reasoning, and analysis. Use this for "
            "'why' questions -- policy rationale, economic commentary, "
            "explanations behind a decision or trend -- as opposed to raw "
            "numbers, which come from query_macro_data instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query -- phrase it as the key topic or question.",
                },
                "date_filter": {
                    "type": "string",
                    "description": (
                        "Optional. A 'YYYY-MM' (or 'YYYY') prefix to restrict the "
                        "search to a specific month/year, when the question names "
                        "one (e.g. 'August 2026' -> '2026-08'). Narrowing to the "
                        "right date avoids pulling similar-sounding content from "
                        "the wrong month's document. Omit if no date is mentioned."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

ALL_TOOLS = [SQL_TOOL_SCHEMA, RAG_TOOL_SCHEMA]


def execute_tool(tool_name: str, tool_input: dict) -> dict:
    """
    Routes a tool_use call to the right implementation. The agent
    orchestrator calls this whenever the model requests a tool.
    """
    if tool_name == "query_macro_data":
        from src.sql_tool.query_engine import run_query
        return run_query(tool_input["sql"])

    if tool_name == "search_rbi_documents":
        from src.rag_tool.retriever import hybrid_retrieve

        try:
            results = hybrid_retrieve(
                tool_input["query"],
                date_filter=tool_input.get("date_filter"),
            )
        except Exception as e:
            return {"success": False, "error": f"Retrieval error: {e}"}

        chunks = [
            {
                "text": r["text"],
                "doc_type": r["metadata"]["doc_type"],
                "date": r["metadata"]["date"],
                "section": r["metadata"]["section"],
                "source_file": r["metadata"]["source_file"],
            }
            for r in results
        ]
        return {"success": True, "chunks": chunks}

    return {"success": False, "error": f"Unknown tool: {tool_name}"}