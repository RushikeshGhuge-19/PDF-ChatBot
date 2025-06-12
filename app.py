
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import faiss
import numpy as np

# Streamlit page config with dark theme style tweaks
st.set_page_config(page_title="PDF Chatbot", layout="centered", initial_sidebar_state="collapsed")

# Apply dark theme tweaks
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

@st.cache_resource
def load_models():
    qa_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return qa_tokenizer, qa_model, embed_model

tokenizer, model, embedder = load_models()

def extract_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

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

def build_faiss(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def get_top_k_chunks(query, chunks, index, k=3):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return [chunks[i] for i in I[0]]

def generate_answer(query, context):
    prompt = f"Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {query}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.title("📄💬 PDF Chatbot (Dark Mode)")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading PDF..."):
        text = extract_text(uploaded_file)
        chunks = chunk_text(text)
        index, _ = build_faiss(chunks)
        st.success("PDF processed!")

    query = st.text_input("Ask a question from the PDF:")
    if st.button("Get Answer") and query:
        top_chunks = get_top_k_chunks(query, chunks, index)
        context = " ".join(top_chunks)
        answer = generate_answer(query, context)
        st.markdown("**Answer:**")
        st.success(answer)
