"""
Agent's core loop (Groq version):
  1. Send the user's question to Groq (with system prompt + tool schema)
  2. If the model requests tool_calls, execute them and send results back
  3. Loop until the model gives a final text answer
  4. Return the full trace (which SQL ran, what came back) -- for
     showing "show your work" in a future Streamlit UI

Groq's API is OpenAI-compatible, so tool-calling's shape differs from
Anthropic (message has a "tool_calls" list, response role="tool").

Run demo: python -m src.agent.orchestrator "What was the USD/INR rate in March 2013?"
"""
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()  # loads GROQ_API_KEY from .env

from groq import Groq

from src.agent.prompts import build_system_prompt
from src.agent.tools import ALL_TOOLS, execute_tool
from src.config import LLM_MODEL, MAX_TOKENS

MAX_TOOL_ITERATIONS = 5  # safety cap against infinite loops


def ask_agent(user_question: str, verbose: bool = True) -> dict:
    """
    Returns:
      {
        "answer": "final text answer",
        "trace": [{"tool": "...", "input": {...}, "result": {...}}, ...]
      }
    """
    client = Groq()  # picks up GROQ_API_KEY from the environment
    system_prompt = build_system_prompt()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]
    trace = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
            tools=ALL_TOOLS,
            tool_choice="auto",
            reasoning_effort="low",  # keeps gpt-oss's internal thinking short,
                                      # leaving more of max_tokens for the answer
        )

        message = response.choices[0].message

        # If the model didn't request a tool, this is the final answer.
        if not message.tool_calls:
            answer_text = message.content or ""
            if not answer_text and verbose:
                # This shouldn't normally happen -- if it does, it's worth
                # knowing why (e.g. the model ran out of tokens mid-thought).
                finish_reason = response.choices[0].finish_reason
                print(f"  Warning: empty answer, finish_reason={finish_reason}")
            return {"answer": answer_text, "trace": trace}

        # Record the assistant's tool_calls message in history.
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        })

        # Execute every tool call and feed the result back into messages.
        for tc in message.tool_calls:
            tool_name = tc.function.name
            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_input = {}

            if verbose:
                print(f"[iteration {iteration + 1}] Tool call: {tool_name}")
                print(f"  Input: {tool_input}")

            result = execute_tool(tool_name, tool_input)
            trace.append({"tool": tool_name, "input": tool_input, "result": result})

            if verbose:
                print(f"  Result: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    return {
        "answer": "Sorry, couldn't produce an answer even after multiple tool calls.",
        "trace": trace,
    }


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY env var is not set. Check your .env file.")
        sys.exit(1)

    question = sys.argv[1] if len(sys.argv) > 1 else "What was the USD/INR rate in March 2013?"
    result = ask_agent(question)
    print("\n=== FINAL ANSWER ===")
    print(result["answer"])