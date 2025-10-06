import os
import io
import time
import numpy as np
import streamlit as st
from PIL import Image
import google.generativeai as genai
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# --- MODEL LOADING ---
@st.cache_resource
def load_local_embedding_model():
    """Loads the sentence-transformer model from Hugging Face, caching it for performance."""
    return SentenceTransformer("all-MiniLM-L6-v2")

# --- GEMINI CONFIGURATION ---
def configure_gemini(api_key):
    """Configures the Gemini API with the provided key."""
    genai.configure(api_key=api_key)

# --- DATA EXTRACTION & PROCESSING ---
def get_text_from_pdf(uploaded_pdf):
    """Extracts text from an uploaded PDF file."""
    texts = []
    pdf_document = fitz.open(stream=uploaded_pdf.getvalue(), filetype="pdf")
    for page in pdf_document:
        if page_text := page.get_text("text").strip():
            texts.append(page_text)
    return texts

def get_image_from_upload(uploaded_image):
    """Opens an uploaded image file."""
    return Image.open(io.BytesIO(uploaded_image.getvalue()))

def summarize_image(image):
    """Generates a text summary of an image using Gemini Pro Vision."""
    # --- FIX: Use the correct, modern model name ---
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt_parts = [
        "You are an expert at analyzing images and charts.",
        "Provide a detailed, objective summary of the contents of this image.",
        "Describe the key elements, data points, and overall conclusion the image conveys.",
        image
    ]
    try:
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        st.error(f"Error generating image summary: {e}")
        return None

# --- EMBEDDING & VECTOR STORE ---
def get_local_embeddings(texts, local_model):
    """Generates embeddings for a list of texts using a local model."""
    with st.spinner(f"Embedding {len(texts)} text chunks locally..."):
        return local_model.encode(texts).tolist()

def create_vector_store(embeddings, content_store):
    """Creates a FAISS vector store."""
    embeddings_np = np.array(embeddings).astype('float32')
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    return (index, content_store)

# --- MAIN PIPELINE ---
def process_files(uploaded_pdf, uploaded_image):
    """Complete pipeline to process files and create a vector store."""
    local_model = load_local_embedding_model()

    texts = get_text_from_pdf(uploaded_pdf)
    image = get_image_from_upload(uploaded_image)
    
    with st.spinner("Generating image summary with Gemini..."):
        image_summary = summarize_image(image)
        if not image_summary:
            return None # Stop if summary generation fails

    all_texts_to_embed = texts + [image_summary]
    all_embeddings = get_local_embeddings(all_texts_to_embed, local_model)
    
    # Store the original image, not its summary, for the final prompt
    content_store = texts + [image]
    
    return create_vector_store(all_embeddings, content_store)

# --- ANSWER GENERATION ---
def generate_answer(user_question, vector_store):
    """Generates an answer using the RAG pipeline."""
    index, content_store = vector_store
    local_model = load_local_embedding_model()

    question_embedding = local_model.encode([user_question]).astype('float32')
    
    k = 5
    _, indices = index.search(question_embedding, k)
    
    valid_indices = [i for i in indices[0] if i < len(content_store)]
    retrieved_context = [content_store[i] for i in valid_indices]
    
    prompt_parts = [
        "You are an expert analyst. Answer the user's question based ONLY on the following context. If the context does not contain the answer, state that you don't know.",
        "\n--- CONTEXT START ---\n"
    ]
    for item in retrieved_context:
        if isinstance(item, str):
            prompt_parts.append(item)
        elif isinstance(item, Image.Image):
            prompt_parts.append(item)
    prompt_parts.append(f"\n--- CONTEXT END ---\n\nQuestion: {user_question}\n\nAnswer:")
    
    # --- FIX: Use the correct, modern model name ---
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt_parts)
    return response.text

