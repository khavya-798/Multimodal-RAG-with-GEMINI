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
def extract_pdf_content(uploaded_pdf):
    """Extracts all text and images from an uploaded PDF file."""
    content_store = []
    pdf_document = fitz.open(stream=uploaded_pdf.getvalue(), filetype="pdf")
    
    for page_num, page in enumerate(pdf_document):
        # Extract text
        if page_text := page.get_text("text").strip():
            content_store.append(page_text)
        
        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                # Convert bytes to PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes))
                content_store.append(pil_image)
            except Exception as e:
                st.warning(f"Warning: Could not extract image {img_index} from page {page_num+1}. Skipping. Error: {e}")
                
    return content_store

def summarize_image(image):
    """Generates a text summary of an image using Gemini Pro Vision."""
   
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
    with st.spinner(f"Embedding {len(texts)} text/image chunks locally..."):
        return local_model.encode(texts).tolist()

def create_vector_store(embeddings, content_store):
    """Creates a FAISS vector store."""
    embeddings_np = np.array(embeddings).astype('float32')
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    return (index, content_store)

# --- MAIN PIPELINE ---
def process_files(uploaded_pdf):
    """Complete pipeline to process files and create a vector store."""
    local_model = load_local_embedding_model()

    content_store = extract_pdf_content(uploaded_pdf)
    
    all_texts_to_embed = []
    with st.spinner("Summarizing images with Gemini..."):
        for item in content_store:
            if isinstance(item, str):
                all_texts_to_embed.append(item)
            elif isinstance(item, Image.Image):
                image_summary = summarize_image(item)
                if image_summary:
                    all_texts_to_embed.append(image_summary)
                else:
                    st.error("Failed to summarize an image, it will be skipped in context.")

    if not all_texts_to_embed:
         st.error("No text or image content could be processed from the PDF.")
         return None

    all_embeddings = get_local_embeddings(all_texts_to_embed, local_model)
    
    # The content_store already contains the original text and images
    return create_vector_store(all_embeddings, content_store)

# --- ANSWER GENERATION ---
def generate_answer(user_question, vector_store):
    """Generates an answer using the RAG pipeline."""
    index, content_store = vector_store
    local_model = load_local_embedding_model()

    question_embedding = local_model.encode([user_question]).astype('float32')
    
    k = 5 # Retrieve top 5 relevant chunks
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
            # Pass the actual image object to Gemini
            prompt_parts.append("\n[Image context below]\n") 
            prompt_parts.append(item)
            prompt_parts.append("\n[End of image context]\n")
    prompt_parts.append(f"\n--- CONTEXT END ---\n\nQuestion: {user_question}\n\nAnswer:")
    
    model = genai.GenerativeModel('gemini-2.5-flash') # Use the appropriate model
    try:
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        st.error(f"Error generating answer from Gemini: {e}")
        return "Sorry, I encountered an error while generating the answer."