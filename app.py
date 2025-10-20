import os
import streamlit as st
from dotenv import load_dotenv
import rag_utils  # Import our utility functions
# Removed base64 import as it's no longer needed

# --- REMOVED Background Image Functions ---
# get_base64_of_bin_file() removed
# set_background_image() removed

# --- CUSTOM STYLING (Loads style.css) ---
def load_css(file_name):
    """Loads a CSS file and injects it into the app."""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file '{file_name}' not found. Please create it in the same folder as app.py.")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Multimodal RAG with Gemini",
    page_icon="🧠", # Brain Icon
    layout="centered" # Keep centered for the card effect
)

# --- LOAD ASSETS ---
# Only load the CSS file now
load_css("style.css")

# --- ENVIRONMENT AND API KEY ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# --- APPLICATION TITLE ---
st.markdown(
    """
    <h1 style="text-align: center;">
        <img src="https://icons.getbootstrap.com/assets/icons/robot.svg" alt="AI" style="height: 1.0em; vertical-align: -0.1em; margin-right: 0.2em; filter: invert(20%) sepia(80%) saturate(500%) hue-rotate(190deg) brightness(95%) contrast(90%);">
        Multimodal RAG with Gemini
    </h1>
    """, unsafe_allow_html=True)
st.write("Upload a PDF document and a related chart image, then ask questions about them.")


# --- SIDEBAR ---
with st.sidebar:
    st.markdown(
        """
        <h3>
            <img src="https://icons.getbootstrap.com/assets/icons/file-earmark-arrow-up.svg" alt="Upload" class="icon" style="filter: brightness(0) invert(1); margin-bottom: -2px;">
            Upload Files
        </h3>
        """, unsafe_allow_html=True)

    uploaded_pdf = st.file_uploader(
        "Upload your PDF document",
        type="pdf",
        key="pdf_uploader"
    )
    uploaded_image = st.file_uploader(
        "Upload your Chart/Image file",
        type=["png", "jpg", "jpeg"],
        key="image_uploader"
    )

    st.markdown("---")

    st.info("💡 **Tip:** Change the theme from the **Settings** menu (top-right `...`)")

    # --- PROFILE LINKS ---
    st.markdown(
        """
        <div class="footer-links">
            <b>Connect with Khavya</b>
            <a href="https://github.com/khavya-798" target="_blank">
                <img src="https://icons.getbootstrap.com/assets/icons/github.svg" class="icon"> GitHub
            </a>
            <a href="https://www.linkedin.com/in/khavyanjali-gopisetty-019720254/" target="_blank">
                <img src="https://icons.getbootstrap.com/assets/icons/linkedin.svg" class="icon"> LinkedIn
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- MAIN LOGIC ---
if not api_key:
    st.error("🚨 Google API Key not found!")
    st.info("Please create a .env file and add your GOOGLE_API_KEY to it.")
    st.stop()
else:
    rag_utils.configure_gemini(api_key)

if uploaded_pdf and uploaded_image:
    pdf_name = uploaded_pdf.name
    image_name = uploaded_image.name
    if st.session_state.get("pdf_name") != pdf_name or st.session_state.get("image_name") != image_name or "vector_store" not in st.session_state:
        with st.spinner("Processing your documents... This may take a moment."):
            # Assumes rag_utils.process_files expects PDF and Image
            st.session_state.vector_store = rag_utils.process_files(uploaded_pdf, uploaded_image)

            st.session_state.pdf_name = pdf_name
            st.session_state.image_name = image_name
            if "messages" in st.session_state:
                 st.session_state.messages = []
        if st.session_state.vector_store:
            st.success("Documents processed successfully! You can now ask questions.")
        else:
            st.error("Failed to process documents. Please check the files or logs.")
            # Clear state if processing failed
            if "pdf_name" in st.session_state: del st.session_state.pdf_name
            if "image_name" in st.session_state: del st.session_state.image_name
            if "vector_store" in st.session_state: del st.session_state.vector_store
    else:
        if "messages" not in st.session_state or not st.session_state.messages:
             st.success("Documents already loaded. Ask your questions below.")
else:
    st.info("Please upload both a PDF and an image file to begin.")
    if "vector_store" in st.session_state:
        st.session_state.clear()


# --- CHAT INTERFACE ---
if "vector_store" in st.session_state and st.session_state.vector_store:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if user_question := st.chat_input("Ask a question about the content of your documents:"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                answer = rag_utils.generate_answer(user_question, st.session_state.vector_store)
                st.markdown(answer)
        # --- FIX: Removed trailing comma ---
        st.session_state.messages.append({"role": "assistant", "content": answer})