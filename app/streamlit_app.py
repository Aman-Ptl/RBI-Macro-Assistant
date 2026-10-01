"""
Streamlit chat UI for the RBI Macro Assistant.

Shows a normal chat interface, plus an expandable "Show tool calls" panel
under each answer so the person using it can see exactly which SQL ran
or which documents were retrieved -- useful both as a demo and as a way
to sanity-check that answers are actually grounded in the tools, not
hallucinated.

Run locally: streamlit run app/streamlit_app.py
Deployed on Streamlit Community Cloud: same command, GROQ_API_KEY comes
from the app's Secrets settings instead of a local .env file.
"""
import sys
from pathlib import Path

# Streamlit adds this script's own directory (app/) to sys.path, not the
# project root -- without this, "src" wouldn't be importable regardless
# of which directory the command is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

import streamlit as st
from groq import RateLimitError

# Streamlit Community Cloud secrets are only exposed via st.secrets, not
# automatically copied into os.environ -- but the rest of this project
# (orchestrator.py's Groq() client, python-dotenv locally) reads the key
# via os.environ. Bridging it here means no other file needs to know or
# care whether it's running locally (.env) or on Community Cloud
# (Secrets UI). Wrapped in try/except because st.secrets raises an error
# (not just an empty dict) when no secrets.toml exists at all -- which is
# the normal case for local development, where the key comes from .env
# instead.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from src.config import DUCKDB_PATH
from src.agent.orchestrator import ask_agent

st.set_page_config(page_title="RBI Macro Assistant", page_icon="🇮🇳", layout="wide")

# Global font-size bump -- Streamlit's defaults look small on large
# desktop monitors, especially combined with "wide" layout. Scaling the
# base font size up (rather than individually tweaking every element)
# keeps spacing/proportions consistent everywhere.
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 1.1rem; }
    section[data-testid="stSidebar"] * { font-size: 1.05rem !important; }
    h1 { font-size: 2.6rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def ensure_data_ready():
    """
    Builds the DuckDB file and vector store on first run if they don't
    already exist. The Docker image builds these at image-build time, but
    Streamlit Community Cloud just pip-installs requirements.txt and runs
    this script directly -- it never runs the Dockerfile -- so without
    this check, a Community Cloud deploy would start with no data at all.

    @st.cache_resource makes this run at most once per app session
    (across reruns from user interaction), not on every single rerun.
    """
    if not DUCKDB_PATH.exists():
        from src.sql_tool.db_loader import load_csv_to_duckdb
        load_csv_to_duckdb()

    from src.rag_tool.vectorstore import get_collection
    if get_collection().count() == 0:
        from scripts.ingest_pdfs import ingest_all_pdfs
        ingest_all_pdfs()

    return True


with st.spinner("Setting up the dataset and document index (first run only)... ⏳"):
    ensure_data_ready()

# Custom header banner -- a plain CSS/HTML gradient in the Indian tricolor
# rather than a hotlinked photo. Hotlinked images from random sites carry
# copyright risk (especially for a project that may end up public on
# GitHub) and can silently break later if the source removes the image --
# not something you want failing mid-demo in an interview. This has
# neither problem and loads instantly.
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #FF9933 0%, #FF9933 33%, #FFFFFF 33%, #FFFFFF 67%, #138808 67%, #138808 100%);
        height: 6px;
        border-radius: 3px;
        margin-bottom: 1.2rem;
    "></div>
    """,
    unsafe_allow_html=True,
)

st.title("🇮🇳 RBI Macro Assistant 📈")
st.caption(
    "💵 Ask about Indian macroeconomic data (1990-2026) or RBI policy decisions. "
    "Answers are grounded in a structured dataset and RBI documents -- not model memory. ₹"
)

with st.sidebar:
    st.markdown("### 🏦 About")
    st.markdown(
        "This agent uses two tools:\n\n"
        "- 📈 **query_macro_data**: SQL over monthly macro data (USD/INR, "
        "reserves, trade, inflation, gold, etc.)\n"
        "- 📄 **search_rbi_documents**: hybrid search (dense + BM25) over "
        "MPC Minutes, Bulletins, Monetary Policy Reports, and State of "
        "the Economy articles\n\n"
        "Every numeric or \"why\" claim is required to come from a tool "
        "call -- never from the model's memory."
    )
    st.divider()
    st.markdown("### 💡 Try asking")
    st.markdown(
        "- 💱 What was the USD/INR rate in March 2013?\n"
        "- 🏛️ Why did the MPC keep the repo rate unchanged in August 2026?\n"
        "- 📉 How did USD/INR trend during the 2008 financial crisis?\n"
        "- 💰 What were India's forex reserves in 2020, and what did the "
        "RBI say about reserves that year?"
    )
    st.divider()
    st.caption("Built with Groq · DuckDB · ChromaDB · Streamlit")

AVATARS = {"user": "🧑", "assistant": "🏦"}

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "trace"}

# Center the chat column even in wide layout, so chat bubbles don't
# stretch uncomfortably across a huge monitor.
chat_col, _ = st.columns([3, 1])

with chat_col:
    # Replay chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
            st.markdown(msg["content"])
            if msg.get("trace"):
                with st.expander("🔍 Show tool calls"):
                    for step in msg["trace"]:
                        st.markdown(f"**Tool:** `{step['tool']}`")
                        st.json(step["input"])
                        st.json(step["result"])

user_question = st.chat_input("Ask a question... 💬")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with chat_col:
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(user_question)

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            with st.spinner("Crunching the numbers... 📊"):
                try:
                    response = ask_agent(user_question, verbose=False)
                    answer = response["answer"] or "I couldn't produce an answer for that -- try rephrasing the question."
                    trace = response["trace"]
                except RateLimitError:
                    answer = (
                        "The model's rate limit was hit. This app runs on Groq's "
                        "free tier, which has a daily token budget -- please try "
                        "again in a few minutes."
                    )
                    trace = []
                except Exception as e:
                    answer = f"Something went wrong: {e}"
                    trace = []

            st.markdown(answer)
            if trace:
                with st.expander("🔍 Show tool calls"):
                    for step in trace:
                        st.markdown(f"**Tool:** `{step['tool']}`")
                        st.json(step["input"])
                        st.json(step["result"])

    st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace})

# Footer attribution -- fixed in the bottom-right corner, unobtrusive
st.markdown(
    """
    <style>
    .footer-credit {
        position: fixed;
        bottom: 8px;
        right: 16px;
        font-size: 0.8rem;
        color: rgba(250, 250, 250, 0.5);
        z-index: 100;
    }
    .footer-credit a {
        color: inherit;
        text-decoration: none;
        border-bottom: 1px dotted rgba(250, 250, 250, 0.4);
    }
    .footer-credit a:hover {
        color: rgba(250, 250, 250, 0.9);
    }
    </style>
    <div class="footer-credit">
        Built by <a href="https://github.com/Aman-Ptl" target="_blank">Aman Patel</a>
    </div>
    """,
    unsafe_allow_html=True,
)