import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import requests
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

# -- FAISS Indexing --
def build_faiss(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def get_top_k_chunks(query, chunks, index, k=5):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return [chunks[i] for i in I[0]]

# -- OpenRouter API Answer Generation --
def generate_answer(query, context, api_key):
    if not context.strip():
        return "❌ No relevant context found in the document."

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

    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
    }

    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ API Error {res.status_code}: {res.text}"
    except Exception as e:
        return f"❌ Exception: {str(e)}"

# -- Streamlit App UI Logic --
st.title("📄💬 PDF Chatbot — Powered by OpenRouter API")
api_key = st.text_input("🔑 Enter your OpenRouter API key", type="password")

uploaded_file = st.file_uploader("📎 Upload a PDF", type=["pdf"])

if uploaded_file and api_key:
    with st.spinner("📖 Reading and embedding PDF..."):
        text = extract_text(uploaded_file)
        chunks = chunk_text(text)
        index, _ = build_faiss(chunks)
        st.success("✅ PDF processed!")

    query = st.text_input("💬 Ask a question from the PDF:")
    if st.button("Get Answer") and query:
        top_chunks = get_top_k_chunks(query, chunks, index)
        context = " ".join(top_chunks)

        st.markdown("### 🔍 Retrieved Context")
        st.code("\n---\n".join(top_chunks))

        with st.spinner("🧠 Generating answer..."):
            answer = generate_answer(query, context, api_key)

        st.markdown("### 💬 Answer")
        st.success(answer)
