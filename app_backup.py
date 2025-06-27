import os
import streamlit as st
from dotenv import load_dotenv
import nest_asyncio
import re
from config import Config
from utils import get_pdf_text, get_text_chunks, highlight_relevant_text
from vector_store import VectorStore
import hybrid_rerank
from mistral_client import ask_mistral

nest_asyncio.apply()

def main():
    load_dotenv()
    st.set_page_config(page_title="Multi PDF Chatbot", page_icon="🤖")
    st.header("Research Assistant")

    for key in ["vector_store", "corpus", "chat_history"]:
        if key not in st.session_state:
            st.session_state[key] = None if key != "chat_history" else []

    with st.sidebar:
        st.header("Chat History")
        for idx, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{idx+1}:** {q}")
            st.markdown(f"**A{idx+1}:** {a}")
            st.divider()
        if st.button("Clear History"):
            st.session_state.chat_history = []

    pdf_docs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    user_question = st.text_input("Ask about your documents")

    if st.button("Build Knowledge Base") and pdf_docs:
        with st.spinner("Processing PDFs..."):
            print("Starting PDF extraction")
            raw_docs = get_pdf_text(pdf_docs)
            print(f"Extracted {len(raw_docs)} pages")
            text_chunks = get_text_chunks(raw_docs)
            print(f"Chunked into {len(text_chunks)} chunks")
            vector_store = VectorStore()
            print("Ensuring collection")
            vector_store.ensure_collection()
            progress_bar = st.progress(0, text="Processing document chunks...")

            def update_progress(percent):
                progress_bar.progress(percent, text=f"Processing... {percent}%")

            print("Adding documents to vector store")
            corpus = vector_store.add_documents(text_chunks, progress_callback=update_progress)
            print("Documents added, updating session state")
            st.session_state.vector_store = vector_store
            st.session_state.corpus = corpus
            st.success(f"Loaded {len(text_chunks)} chunks from {len(pdf_docs)} PDF(s)!")
            progress_bar.empty()

            


    if user_question and st.session_state.vector_store:
        corpus = st.session_state.vector_store.get_corpus()
        if not corpus:
            st.warning("No documents found in the knowledge base. Please upload PDFs and build the knowledge base first.")
        else:
            try:
                results = hybrid_rerank.hybrid_search(
                    user_question,
                    st.session_state.vector_store.client,
                    st.session_state.vector_store.collection_name,
                    corpus,
                    limit=8  
                )
            except ValueError as e:
                st.error(str(e))
                results = []

            if results:
                context_chunks = []
                seen_sections = set()
                for idx, hits in enumerate(results):
                    if isinstance(hits, (list, tuple)):
                        for jdx, hit in enumerate(hits):
                            entity = hit.get("entity", hit)
                            metadata = entity.get("metadata", {})
                            text = entity.get("text", "")
                            section = metadata.get("section", "").lower()
                            # Prioritize abstract/introduction
                            if section in {"abstract", "introduction"} and section not in seen_sections:
                                context_chunks.insert(0, text)
                                seen_sections.add(section)
                            else:
                                context_chunks.append(text)
                    else:
                        st.write(f"[ERROR] Unhandled result type: {type(hits)} - {hits}")

                full_context = "\n\n".join(context_chunks)

                if st.checkbox("Show retrieved context (debug)"):
                    st.markdown("#### Retrieved Context")
                    st.markdown(full_context)

                prompt = (
                    "You are a knowledgeable research assistant. "
                    "Answer the user's question using ONLY the provided context. "
                    "Structure your response naturally: "
                    "Use paragraphs, bullet points, or section headers as needed for clarity. "
                    "Always cite sources inline like (Section X.Y) or (Page Z). "
                    "If information is missing, say: 'This topic is not mentioned in the paper.' "
                    "Never repeat the question or include unrelated content. "
                    f"Context:\n{full_context}\n\n"
                    f"Question: {user_question}\n"
                    "Answer:"
                )



                answer = ask_mistral(prompt, max_tokens=768, temperature=0.2)
                

                st.session_state.chat_history.append((user_question, answer))

                st.markdown("### Answer")
                st.markdown(answer)

                with st.expander("Show source context"):
                  
                    flat_hits = []
                    for idx, hits in enumerate(results):
                        if isinstance(hits, (list, tuple)):
                            for jdx, hit in enumerate(hits):
                                entity = hit.get("entity", hit)
                                metadata = entity.get("metadata", {})       
                                text = entity.get("text", "")
                              
                                flat_hits.append((metadata, text))
                        else:
                            st.write(f"[ERROR] Unhandled result type: {type(hits)} - {hits}")

                    

                    for idx, (metadata, text) in enumerate(flat_hits[:3]):
                        source = metadata.get("source", "unknown")
                        page = metadata.get("page", "N/A")
                        line = metadata.get("line", "N/A")
                        st.markdown(f"**Source {idx+1}:** {source} (page {page}, line {line})")
                        st.markdown(highlight_relevant_text(text, user_question))
                        st.divider()


if __name__ == "__main__":
    main()
