import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import requests
import os
from sentence_transformers import SentenceTransformer

# -- Streamlit UI Setup --
st.set_page_config(page_title="PDF Chatbot (OpenRouter API)", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stFileUploader>div>div {
        background-color: #262730;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -- Load Embedder --
@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-base-en-v1.5")

embedder = load_embedder()

# -- PDF Text Extraction --
def extract_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# -- Text Chunking --
def chunk_text(text, max_tokens=200):
    sentences = text.split(". ")
    chunks, chunk = [], ""
    for sent in sentences:
        if len((chunk + sent).split()) <= max_tokens:
            chunk += sent + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sent + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# -- FAISS Index --
def build_faiss(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def get_top_k_chunks(query, chunks, index, k=5):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return [chunks[i] for i in I[0]]

# -- OpenRouter API-based Answer Generation --
def generate_answer(query, context, api_key):
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistralai/mistral-7b-instruct",  # You can also try "meta-llama/llama-3-8b-instruct"
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300
    }

    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {res.status_code} - {res.text}"

# -- Streamlit UI Logic --
st.title("📄💬 PDF Chatbot — Powered by OpenRouter API")
api_key = st.text_input("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", type="password")

uploaded_file = st.file_uploader("📎 Upload a PDF", type=["pdf"])

if uploaded_file and api_key:
    with st.spinner("📖 Reading and embedding PDF..."):
        text = extract_text(uploaded_file)
        chunks = chunk_text(text)
        index, _ = build_faiss(chunks)
        st.success("✅ PDF processed!")

    query = st.text_input("💬 Ask a question from the PDF:")
    if st.button("🧠 Get Answer") and query:
        top_chunks = get_top_k_chunks(query, chunks, index)
        context = " ".join(top_chunks)
        st.markdown("📚 *Retrieved Context:*")
        st.code("\n---\n".join(top_chunks))
        with st.spinner("🤖 Generating answer..."):
            answer = generate_answer(query, context, api_key)
            st.markdown("✅ *Answer:*")
            st.success(answer)
