"""
app.py — a browser chat UI for the RAG pipeline.

Run with:
    streamlit run app.py

Opens at http://localhost:8501
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st
from rag import ask
import config

st.set_page_config(page_title="Victorian Law Assistant", page_icon="⚖️")
st.title("⚖️ Victorian Law Assistant")
st.caption("Ask a question about the Crimes Act 1958, Drugs, Poisons and Controlled "
           "Substances Act 1981, Infringements Act 2006, Road Safety Act 1986, "
           "Sentencing Act 1991, Summary Offences Act 1966, or Victoria Police Act 2013.")

with st.sidebar:
    st.header("Settings")
    use_dense = st.checkbox("Use dense retrieval", value=True)
    use_rerank = st.checkbox("Use reranker", value=True)
    extractive = st.checkbox("Extractive mode (skip LLM)", value=False)
    llm_model = st.text_input("Ollama model", value=config.OLLAMA_MODEL)
    top_k = st.slider("Sections to retrieve", 1, 10, config.FINAL_TOP_K)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

question = st.chat_input("Ask a question...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                result = ask(question, top_k=top_k, use_dense=use_dense,
                              use_rerank=use_rerank, llm=llm_model,
                              extractive=extractive)
                answer = result["answer"]
                sources = result["sources"]
            except RuntimeError as e:
                answer = f"⚠️ {e}"
                sources = []

        st.write(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"**{s['act']}, s. {s['section']}** — {s['heading']}")
                    st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))

    st.session_state.history.append({"role": "assistant", "content": answer})
