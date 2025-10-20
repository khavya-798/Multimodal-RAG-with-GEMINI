Multimodal RAG with Gemini
🚀 Project Overview
This project is a sophisticated, multimodal Question-Answering system that leverages the power of Google's Gemini model. It allows users to upload a PDF document and a related chart/image, and then ask complex questions that require reasoning over both the textual and visual information.

The application uses a Retrieval-Augmented Generation (RAG) pipeline to provide grounded, accurate answers based on the content of the uploaded files.

✨ Key Features
Multimodal Input: Processes both PDF documents and image files (.png, .jpg).

Hybrid Embedding Strategy: Uses a local sentence-transformers model for high-volume text embedding to avoid API rate limits, and the Gemini API for advanced image summarization.

Vector Search: Employs a high-speed FAISS vector database for efficient similarity search to retrieve relevant context.

Advanced Reasoning: Leverages the gemini-2.5-flash model to analyze the retrieved text and image context together to generate comprehensive answers.

Interactive UI: A user-friendly web interface built with Streamlit for file uploads and a real-time chat experience.

🛠️ Tech Stack
AI & Machine Learning: Google Gemini, Sentence-Transformers, FAISS, PyMuPDF, Pillow

Backend & Processing: Python, NumPy

Frontend: Streamlit

Environment: Python Virtual Environment, Dotenv for secrets management
