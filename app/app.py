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
   
    st.set_page_config(
        page_title="Research Assistant",
        page_icon="📚",
        layout="centered"
    )
    

    st.markdown("""
    <div style="text-align:center; margin-bottom:30px">
        <h1 style="display:inline-block; border-bottom:2px solid #4f8bf9; padding-bottom:10px">
        📚 Research Assistant
        </h1>
    </div>
    """, unsafe_allow_html=True)

    # session state intialize kar diya
    for key in ["vector_store", "corpus", "chat_history"]:
        if key not in st.session_state:
            st.session_state[key] = None if key != "chat_history" else []

    # sidebar 
    with st.sidebar:
        st.subheader("Chat History")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
        st.divider()
        
        for idx, (q, a) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{idx+1}: {q[:50]}..." if len(q) > 50 else f"Q{idx+1}: {q}"):
                st.markdown(f"**Question:** {q}")
                st.markdown(f"**Answer:** {a}")
        st.divider()

    # File upload 
    with st.container():
        st.subheader("📂 Upload Documents")
        pdf_docs = st.file_uploader(
            "Drag and drop PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if st.button("⚙️ Build Knowledge Base", type="primary", disabled=not pdf_docs):
            with st.spinner("Processing PDFs..."):
                print("Starting PDF extraction")
                raw_docs = get_pdf_text(pdf_docs)
                print(f"Extracted {len(raw_docs)} pages")
                text_chunks = get_text_chunks(raw_docs)
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
                st.success(f"✅ Loaded {len(text_chunks)} chunks from {len(pdf_docs)} PDF(s)!")
                progress_bar.empty()

    # Question input 
    user_question = st.text_input(
        "Ask about your documents:",
        placeholder="Type your research question here...",
        label_visibility="collapsed"
    )

    # Answer 
    if user_question and st.session_state.vector_store:
        corpus = st.session_state.vector_store.get_corpus()
        if not corpus:
            st.warning("⚠️ No documents found. Please upload PDFs and build the knowledge base first.")
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
                            if section in {"abstract", "introduction"} and section not in seen_sections:
                                context_chunks.insert(0, text)
                                seen_sections.add(section)
                            else:
                                context_chunks.append(text)
                    else:
                        st.write(f"[ERROR] Unhandled result type: {type(hits)} - {hits}")

                full_context = "\n\n".join(context_chunks)

                # Debug blehh
                debug = st.checkbox("🐞 Show retrieved context (debug mode)")
                if debug:
                    with st.expander("Retrieved Context"):
                        st.markdown(full_context)

                # system prompt
                prompt = (
                    "You are a knowledgeable research assistant. "
                    "Answer the user's question using ONLY the provided context. "
                    "Structure your response naturally: "
                    "Use paragraphs, bullet points, or section headers as needed."
                    "Always cite sources inline like (Section X.Y) or (Page Z). "
                    "If information is missing, say: 'This topic is not mentioned in the paper.' "
                    "Never repeat the question or include unrelated content. "
                    f"Context:\n{full_context}\n\n"
                    f"Question: {user_question}\n"
                    "Answer:"
                )

                
                answer = ask_mistral(prompt, max_tokens=768, temperature=0.2)
                
                
                st.session_state.chat_history.append((user_question, answer))

                #Answer display
                st.markdown("---")
                st.subheader("💡 Answer")
                st.markdown(answer)
                st.divider()

                # Source context
                with st.expander("🔍 Show Source Context"):
                    flat_hits = []
                    for idx, hits in enumerate(results):
                        if isinstance(hits, (list, tuple)):
                            for jdx, hit in enumerate(hits):
                                entity = hit.get("entity", hit)
                                metadata = entity.get("metadata", {})       
                                text = entity.get("text", "")
                                flat_hits.append((metadata, text))
                    
                    # TOP 3 Context
                    for idx, (metadata, text) in enumerate(flat_hits[:3]):
                        source = metadata.get("source", "unknown")
                        page = metadata.get("page", "N/A")
                        line = metadata.get("line", "N/A")
                        
                        # Source card
                        with st.container():
                            st.markdown(f"**📄 Source {idx+1}:** `{source}` (Page {page}, Line {line})")
                            st.caption(highlight_relevant_text(text, user_question))
                            st.divider()

if __name__ == "__main__":
    main()
