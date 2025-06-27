# Research Assistant: PDF Q&A with Hybrid Search

A modern, interactive research assistant built with Streamlit.  
Upload research papers (PDFs), build a knowledge base, and ask questions using advanced hybrid search (dense + sparse retrieval) and a local Mistral-7B language model.

---

## 🚀 Features

- **PDF Upload:** Drag and drop research papers for instant processing.
- **Hybrid Search:** Combines BM25 (sparse) and dense vector retrieval for accurate context.
- **Local LLM:** Answers generated using a local Mistral-7B model via `llama.cpp`.
- **Milvus Integration:** Uses Milvus vector database for scalable, fast search.
- **Chat History:** Sidebar displays previous questions and answers.
- **Context Highlighting:** See relevant source snippets for every answer.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/research_bot.git
cd research_bot

### 2. Create and Activate a Virtual Environment

**Windows:**

python -m venv venv
venv\Scripts\activate


### 3. Install Dependencies

All dependencies are listed in `req.txt`:

pip install -r req.txt


### 4. Set Up Environment Variables

Create a `.env` file in your project root with the following content (edit paths as needed):


MISTRAL_MODEL_PATH=path/to/mistral-7b-v0.1.Q4_K_M.gguf
MILVUS_URI=http://localhost:19530


### 5. Download Required Model Files

- Download the [Mistral-7B](https://mistral.ai/news/announcing-mistral-7b/) model and place it at the path specified in your `.env` file.
- **Do not commit large model files to the repository.**

---

## ▶️ Usage

1. **Start Milvus** (if using locally; see [Milvus Quick Start](https://milvus.io/docs/v2.3.x/install_standalone-docker.md))
2. **Run the Streamlit app:**

    ```
    streamlit run app.py
    ```

3. **Use the web interface to:**
    - Upload PDFs
    - Build the knowledge base
    - Ask questions and get context-rich answers!

---

## 📁 File Structure


app.py # Main Streamlit app
config.py # Configuration settings
embeddings.py # Dense embedding utilities
hybrid_rerank.py # Hybrid search logic
mistral_client.py # LLM interface
sparse_utils.py # Sparse vector helpers
utils.py # PDF/text utilities
vector_store.py # Milvus vector store logic
req.txt # Python dependencies (requirements)
.gitignore # Files/folders to ignore
README.md # This file


---

## ⚠️ 

- **Do not commit large model files or database volumes.**
- **Keep your `.env` file private; do not share secrets.**
- **For Milvus setup, follow the official [Milvus documentation](https://milvus.io/docs/).**
- **All dependencies are listed in `req.txt`.**  
  If you add new packages, update it with:
    ```
    pip freeze > req.txt
    ```

---


---

## 🙋 FAQ

**Q: Can I use this with my own PDFs?**  
A: Yes! Just upload them via the web interface.

**Q: Is the LLM fully local?**  
A: Yes, the Mistral-7B model runs locally via `llama.cpp`.

**Q: How do I add more dependencies?**  
A: Install them in your venv, then run `pip freeze > req.txt` to update the requirements file.

---

## 📫 Contact

For questions or suggestions, open an issue or contact [YOUR_EMAIL@example.com].

