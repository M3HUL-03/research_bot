import os
import streamlit as st
from dotenv import load_dotenv
import nest_asyncio
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections, utility
from mistral_client import ask_mistral
from langchain.schema import Document

nest_asyncio.apply()

# Modified PDF processing with metadata
def get_pdf_text(pdf_docs):
    documents = []
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            filename = pdf.name  # Get original filename
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": filename,
                            "page": page_num + 1
                        }
                    })
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
    return documents

def get_text_chunks(documents):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = []
    for doc in documents:
        text_chunks = splitter.split_text(doc["text"])
        for chunk in text_chunks:
            chunks.append({
                "text": chunk,
                "metadata": doc["metadata"]
            })
    return chunks

def get_vector_store(text_chunks):
    connections.connect(host="localhost", port="19530")
    
    if utility.has_collection("doc_chat"):
        utility.drop_collection("doc_chat")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create documents with metadata
    documents = [
        {"page_content": chunk["text"], "metadata": chunk["metadata"]}
        for chunk in text_chunks
    ]
    
    vector_store = Milvus.from_documents(
        documents=documents,
        embedding=embeddings,
        connection_args={"host": "localhost", "port": "19530"},
        collection_name="doc_chat",
        index_params={
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
    )
    return vector_store

def retrieve_relevant_chunks(query, vector_store, top_k=5):
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(query)
    return docs  # Now returns full documents with metadata

def main():
    load_dotenv()
    st.set_page_config(page_title="Multi PDF Chatbot", page_icon="🤖")
    st.header("Research Assistant")

    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for chat history
    with st.sidebar:
        st.header("Chat History")
        for idx, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{idx+1}:** {q}")
            st.markdown(f"**A{idx+1}:** {a}")
            st.divider()
        
        if st.button("Clear History"):
            st.session_state.chat_history = []

    # Main interface
    pdf_docs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    user_question = st.text_input("Ask about your documents")

    if st.button("Build Knowledge Base") and pdf_docs:
        with st.spinner("Processing..."):
            raw_docs = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_docs)
            vector_store = get_vector_store(text_chunks)
            st.session_state.vector_store = vector_store
            st.success(f"Loaded {len(text_chunks)} chunks from {len(pdf_docs)} PDF(s)!")

    if user_question and st.session_state.vector_store:
        # Retrieve documents with metadata
        docs = retrieve_relevant_chunks(user_question, st.session_state.vector_store)
        
        # Build context with source information
        context = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            context.append(f"[From {source}, page {page}]: {doc.page_content}")
        
        full_context = "\n\n".join(context)
        
        # Create prompt
        prompt = (
            "You are a research assistant. Answer using ONLY the context below. "
            "If the answer isn't in the context, say 'I don't know.'\n\n"
            f"Context:\n{full_context}\n\n"
            f"Question: {user_question}\n"
            "Answer:"
        )
        
        # Get answer
        answer = ask_mistral(prompt, max_tokens=512, temperature=0.2)
        
        # Add to chat history
        st.session_state.chat_history.append((user_question, answer))
        
        # Display answer
        st.markdown("### Answer")
        st.markdown(answer)
        
        # Show context sources
        with st.expander("Show source context"):
            for idx, doc in enumerate(docs):
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "N/A")
                st.markdown(f"**Source {idx+1}:** {source} (page {page})")
                st.markdown(doc.page_content)
                st.divider()

if __name__ == "__main__":
    main()




# # BEFORE HYBRID SEARCH, RE RANKING INTEGRATION 

# import os
# import streamlit as st
# from dotenv import load_dotenv
# import nest_asyncio
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_milvus import Milvus
# from pymilvus import connections, utility, MilvusClient
# from mistral_client import ask_mistral
# from langchain.schema import Document
# import hybrid_rerank
# from hybrid_rerank import embed_dense, embed_sparse

# nest_asyncio.apply()

# def get_pdf_text(pdf_docs):
#     documents = []
#     for pdf in pdf_docs:
#         try:
#             reader = PdfReader(pdf)
#             filename = pdf.name
#             for page_num, page in enumerate(reader.pages):
#                 text = page.extract_text()
#                 if text:
#                     documents.append({
#                         "text": text,
#                         "metadata": {
#                             "source": filename,
#                             "page": page_num + 1
#                         }
#                     })
#         except Exception as e:
#             st.error(f"Error processing {pdf.name}: {str(e)}")
#     return documents

# def get_text_chunks(documents):
#     splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1200,
#         chunk_overlap=400,
#         length_function=len
#     )
#     chunks = []
#     for doc in documents:
#         text_chunks = splitter.split_text(doc["text"])
#         for chunk in text_chunks:
#             chunks.append({
#                 "text": chunk,
#                 "metadata": doc["metadata"]
#             })
#     return chunks

# def get_vector_store(text_chunks):
#     connections.connect(host="localhost", port="19530")
#     client = MilvusClient(uri="http://localhost:19530")
#     collection_name = "doc_chat"


#     # Drop collection if it exists
#     if client.has_collection(collection_name):
#         client.drop_collection(collection_name)

#     # Create collection with schema
#     client.create_collection(
#         collection_name=collection_name,
#         fields=fields
#     )

#     # Prepare data for insertion
#     data = []
#     for idx, chunk in enumerate(text_chunks):
#         dense_vec = embed_dense(chunk["text"])
#         sparse_vec = embed_sparse(chunk["text"])
#         data.append({
#             "text": chunk["text"],
#             "dense_vector": dense_vec,
#             "sparse_vector": sparse_vec,
#             "metadata": chunk["metadata"]
#         })

#     client.insert(collection_name=collection_name, data=data)
#     return client, collection_name

#     if utility.has_collection("doc_chat"):
#         utility.drop_collection("doc_chat")
#     embeddings = HuggingFaceEmbeddings(
#         model_name="BAAI/bge-base-en-v1.5",
#         encode_kwargs={'normalize_embeddings': True}
#     )
#     documents = [
#         Document(
#             page_content=chunk["text"],
#             metadata=chunk["metadata"]
#         ) for chunk in text_chunks
#     ]
#     vector_store = Milvus.from_documents(
#         documents=documents,
#         embedding=embeddings,
#         connection_args={"host": "localhost", "port": "19530"},
#         collection_name="doc_chat",
#         index_params={
#             "metric_type": "L2",
#             "index_type": "IVF_FLAT",
#             "params": {"nlist": 128},
#         }
#     )
#     return vector_store

# def retrieve_relevant_chunks(query, vector_store, top_k=5):
#     retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
#     docs = retriever.get_relevant_documents(query)
#     query_terms = query.lower().split()
#     filtered_docs = [
#         doc for doc in docs 
#         if any(term in doc.page_content.lower() for term in query_terms)
#     ]
#     return filtered_docs[:top_k]


# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Multi PDF Chatbot", page_icon="🤖")
#     st.header("Research Assistant")

#     # Initialize session state
#     if "vector_store" not in st.session_state:
#         st.session_state.vector_store = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     # Sidebar for chat history
#     with st.sidebar:
#         st.header("Chat History")
#         for idx, (q, a) in enumerate(st.session_state.chat_history):
#             st.markdown(f"**Q{idx+1}:** {q}")
#             st.markdown(f"**A{idx+1}:** {a}")
#             st.divider()
#         if st.button("Clear History"):
#             st.session_state.chat_history = []

#     # Main interface
#     pdf_docs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
#     user_question = st.text_input("Ask about your documents")

#     if st.button("Build Knowledge Base") and pdf_docs:
#         with st.spinner("Processing..."):
#             raw_docs = get_pdf_text(pdf_docs)
#             text_chunks = get_text_chunks(raw_docs)
#             corpus = [chunk["text"] for chunk in text_chunks]
            
#             vector_store = get_vector_store(text_chunks)
#             results = hybrid_rerank.hybrid_search(query, vector_store, corpus)
#             st.session_state.vector_store = vector_store
#             st.success(f"Loaded {len(text_chunks)} chunks from {len(pdf_docs)} PDF(s)!")

#     if user_question and st.session_state.vector_store:

#         corpus = st.session_state.corpus

#         results = hybrid_rerank.hybrid_search(user_question, st.session_state.vector_store, corpus)

#         #Reranking results 
#         #
#         #

#         # Retrieve documents with metadata
#         docs = retrieve_relevant_chunks(user_question, st.session_state.vector_store)
#         # Build context with source information
#         context = []
#         for idx, doc in enumerate(docs):
#             source = doc.metadata.get("source", "unknown")
#             page = doc.metadata.get("page", "N/A")
#             context.append(f"### Source {idx+1} ({source}, page {page}):\n{doc.page_content}")
#         full_context = "\n\n".join(context)

#         # Debug: Show the actual retrieved context if checkbox ticked
#         if st.checkbox("Show retrieved context (debug)"):
#             st.markdown("#### Retrieved Context")
#             st.markdown(full_context)

#         # Improved prompt engineering
#         prompt = (
#         "You are a highly precise technical research assistant. You are given a user question and a document extract.\n"
#         "You must answer ONLY the user's question using the context below.\n"
#         "Do NOT repeat or reformat the context.\n"
#         "Do NOT answer any other questions you may see in the context.\n"
#         "Do NOT invent new Q&A pairs. If the answer isn't found, say 'Not found in context.'\n\n"
#         f"Context:\n{full_context}\n\n"
#         f"User Question: {user_question}\n"
#         "Answer:"
#     )


#         # Get answer
#         answer = ask_mistral(prompt, max_tokens=512, temperature=0.3)  # Lower temperature
        
#         # Post-process to keep only first sentence
#         import re
#         answer = re.split(r"[.!?]", answer, 1)[0] + "."

#         # Add to chat history
#         st.session_state.chat_history.append((user_question, answer))

#         # Display answer
#         st.markdown("### Answer")
#         st.markdown(answer)

#         # Show context sources in an expander
#         with st.expander("Show source context"):
#             for idx, doc in enumerate(docs):
#                 source = doc.metadata.get("source", "unknown")
#                 page = doc.metadata.get("page", "N/A")
#                 st.markdown(f"**Source {idx+1}:** {source} (page {page})")
#                 st.markdown(doc.page_content)
#                 st.divider()

                

# if __name__ == "__main__":
#     main()






#
#
#
#

#embeddings.py 

# from sentence_transformers import SentenceTransformer
# from langchain_milvus import Milvus
# from pymilvus import connections, utility

# def get_vector_store(text_chunks):
#     connections.connect(host="localhost", port="19530")
#     if utility.has_collection("doc_chat"):
#         utility.drop_collection("doc_chat")
#     embeddings = SentenceTransformer("BAAI/bge-m3")  # Use BGE-M3, runs on CPU
#     vector_store = Milvus.from_texts(
#         texts=text_chunks,
#         embedding=embeddings,
#         connection_args={"host": "localhost", "port": "19530"},
#         collection_name="doc_chat",
#         index_params={
#             "metric_type": "L2",
#             "index_type": "IVF_FLAT",
#             "params": {"nlist": 128},
#         }
#     )
#     return vector_store


#
#
#
# mistral_client.py

# from llama_cpp import Llama

# # Download and specify the path to your GGUF model file
# llm = Llama(model_path=r"D:\new_chatbot\mistral-7b-v0.1.Q4_K_M.gguf", n_ctx=4096, n_threads=10)

# prompt = "What is the capital of France?"
# output = llm(prompt, max_tokens=512)
# print(output['choices'][0]['text'].strip())
# def ask_mistral(prompt):
#     output = llm(prompt, max_tokens=512)
#     return output['choices'][0]['text']

# def ask_mistral(prompt, max_tokens=256, temperature=0.2):
#     output = llm(
#         prompt,
#         max_tokens=max_tokens,        # Limit answer length
#         temperature=temperature,      # Control creativity (lower = more factual)
#         top_p=0.95,                   # (optional) Nucleus sampling
#         top_k=40,                     # (optional) Top-k sampling
#         repeat_penalty=1.1            # (optional) Reduce repetition
#     )
#     return output['choices'][0]['text']



#UTIL 

# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter

# def get_pdf_text(pdf_docs):
#     return "".join(
#         page.extract_text()
#         for pdf in pdf_docs
#         for page in PdfReader(pdf).pages
#     )

# def get_text_chunks(text):
#     splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=500,
#         chunk_overlap=100,
#         length_function=len
#     )
#     return splitter.split_text(text)

