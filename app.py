import os
import streamlit as st
from dotenv import load_dotenv
import rag_utils  # Import our utility functions

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Multimodal RAG with Gemini",
    page_icon="📄",
    layout="wide"
)

# --- ENVIRONMENT AND API KEY ---
load_dotenv()
# --- FIX: Securely load the API key from the environment ---
api_key = os.getenv("GOOGLE_API_KEY")

# --- APPLICATION ---
st.title("📄 Multimodal RAG with Gemini")
st.write("Upload a PDF document and a related chart image, then ask questions about them.")

# --- SIDEBAR ---
with st.sidebar:
    # --- FIX: Removed the API key text_input from the UI ---
    st.header("Upload Files")
    uploaded_pdf = st.file_uploader("Upload your PDF document", type="pdf")
    uploaded_image = st.file_uploader("Upload your Chart/Image file", type=["png", "jpg", "jpeg"])

# --- MAIN LOGIC ---
# Stop the app if the API key is not configured
if not api_key:
    st.error("🚨 Google API Key not found!")
    st.info("Please create a .env file in the project's root directory and add your GOOGLE_API_KEY to it.")
    st.stop()
else:
    # Configure Gemini with the loaded key
    rag_utils.configure_gemini(api_key)

if uploaded_pdf and uploaded_image:
    if "vector_store" not in st.session_state:
        with st.spinner("Processing your documents... This may take a moment."):
            st.session_state.vector_store = rag_utils.process_files(uploaded_pdf, uploaded_image)
        st.success("Documents processed successfully! You can now ask questions.")
else:
    st.info("Please upload both a PDF and an image file to begin.")

# --- CHAT INTERFACE ---
if "vector_store" in st.session_state:
    user_question = st.chat_input("Ask a question about the content of your documents:")
    if user_question:
        with st.spinner("Generating answer..."):
            answer = rag_utils.generate_answer(user_question, st.session_state.vector_store)
            st.chat_message("user").write(user_question)
            st.chat_message("assistant").write(answer)

