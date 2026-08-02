"""Streamlit demo for the RAG Document Assistant."""
import streamlit as st
from dotenv import load_dotenv

from rag_assistant.config import Config
from rag_assistant.pipeline import RAGPipeline

load_dotenv()
st.set_page_config(page_title="RAG Document Assistant", page_icon="📚")


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(Config.from_yaml("configs/config.yaml"))


st.title("📚 RAG Document Assistant")
st.caption("Ask questions about your documents. Answers cite their sources.")

question = st.text_input("Your question:")
if question:
    with st.spinner("Retrieving and generating..."):
        result = get_pipeline().answer(question)
    st.subheader("Answer")
    st.write(result["answer"])
    with st.expander("Retrieved sources"):
        for c in result["contexts"]:
            st.markdown(f"**{c['source']}** · score `{c['score']:.3f}`")
            st.text(c["text"][:400] + "...")
