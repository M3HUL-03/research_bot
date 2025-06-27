import re

def get_pdf_text(pdf_docs):
    """
    Extracts text from uploaded PDF files.
    Returns a list of dicts: [{"text": ..., "source": ...}, ...]
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError("PyPDF2 is required for PDF extraction. Install it with 'pip install PyPDF2'.")

    raw_docs = []
    for pdf_file in pdf_docs:
        try:
            reader = PdfReader(pdf_file)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    raw_docs.append({
                        "text": page_text,
                        "source": getattr(pdf_file, "name", "uploaded.pdf"),
                        "page": i + 1
                    })
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
    return raw_docs

def get_text_chunks(raw_docs, chunk_size=600, chunk_overlap=20):
    import re
    text_chunks = []
    for doc in raw_docs:
        text = doc["text"]
        source = doc.get("source", "unknown")
        page = doc.get("page", None)

        # --- Extract and add abstract/introduction as special chunks ---
        abstract_match = re.search(
            r'(abstract[\s\S]{0,1000}?)(?=\n\n|\n\s*\n|introduction|1\. )', text, re.IGNORECASE)
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            text_chunks.append({
                "text": abstract,
                "metadata": {"source": source, "section": "abstract", "page": page, "line": 1}
            })
        intro_match = re.search(
            r'(introduction[\s\S]{0,1500}?)(?=\n\n|\n\s*\n|2\. )', text, re.IGNORECASE)
        if intro_match:
            intro = intro_match.group(1).strip()
            text_chunks.append({
                "text": intro,
                "metadata": {"source": source, "section": "introduction", "page": page, "line": 1}
            })

        # --- Standard chunking with page and line number tracking ---
        lines = text.splitlines()
        char_ptr = 0
        while char_ptr < len(text):
            chunk_text = text[char_ptr:char_ptr + chunk_size]
            # Find the starting line number for this chunk
            chars_counted = 0
            start_line = 1
            for idx, line in enumerate(lines):
                chars_counted += len(line) + 1  # +1 for newline
                if chars_counted > char_ptr:
                    start_line = idx + 1
                    break

            chunk_metadata = {"source": source}
            if page is not None:
                chunk_metadata["page"] = page
            chunk_metadata["line"] = start_line

            text_chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
            char_ptr += chunk_size - chunk_overlap
    return text_chunks

def highlight_relevant_text(text, query, max_keywords=5):
    """
    Bold the most relevant keywords from the query in the text.
    """
    # Simple keyword extraction: split query, filter short/common words
    keywords = [w for w in re.findall(r'\w+', query) if len(w) > 2][:max_keywords]
    for kw in keywords:
        # Use regex for case-insensitive replacement, avoid double-highlighting
        text = re.sub(f'({re.escape(kw)})', r'**\1**', text, flags=re.IGNORECASE)
    return text

