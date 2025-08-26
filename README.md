# GIKI Prospectus Q&A Chatbot using Retrieval-Augmented Generation (RAG)

![GIKI Prospectus Cover](UG-Prospectus-2024.pdf) <!--
A Retrieval-Augmented Generation (RAG) based chatbot for querying GIKI (Ghulam Ishaq Khan Institute of Engineering Sciences and Technology) documents, such as the UG Prospectus 2024, Fee Structure, and Academic Rules. Users can upload up to 5 documents (PDF, DOCX, TXT, etc.) and ask natural language questions. The system extracts text, embeds chunks using sentence-transformers, stores in FAISS, and generates answers via Llama3 on Groq API. Built with LangChain and a Gradio UI for easy interaction.

This project fulfills the academic requirements for a RAG-based Q&A system, supporting English/Urdu toggle (placeholder for Urdu) and conversational memory.

## Features

- **Document Ingestion**: Supports PDF (with PyMuPDF/PyPDF2/pdfplumber fallbacks), DOCX, TXT, PPTX, CSV, XLSX.
- **Text Processing**: Semantic chunking (fallback to recursive splitting, 1000-token chunks).
- **RAG Pipeline**: Embeddings with `all-MiniLM-L6-v2`, FAISS vector store, top-8 retrieval, Llama3-8b-8192 LLM via Groq.
- **Interface**: Gradio web UI with file upload, chat window, language dropdown (English/Urdu), and clear button.
- **Prompt Engineering**: Structured template to avoid hallucinations; answers formatted in Markdown (bullets, headings).
- **Evaluation**: Tested with UG Prospectus 2024; accuracy assessed (e.g., 3/5 for fee queries due to minor hallucinations).
- **Optional**: Urdu translation (placeholder); source metadata stored but not displayed in UI.

## Demo

- Run locally: See "Setup" below.
- Sample Query: "What is the fee structure for CS undergraduate?" → Retrieves and generates response from uploaded PDF.

## Project Structure

giki-chatbot/
├── main.py # Core logic: Document loading, processing, RAG chain, chatbot response
├── interface/
│ └── ui.py # Gradio UI components and styling
├── requirements.txt # Dependencies
├── UG-Prospectus-2024.pdf # Sample document (282 pages)
├── README.md # This file
└── .env # Environment variables (e.g., GROQ_API_KEY; add to .gitignore)
