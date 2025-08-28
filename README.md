# GIKI Prospectus Q&A Chatbot using Retrieval-Augmented Generation (RAG)

A Retrieval-Augmented Generation (RAG)-based chatbot for querying **GIKI (Ghulam Ishaq Khan Institute of Engineering Sciences and Technology)** documents, such as the **UG Prospectus 2024**, **Fee Structure**, and **Academic Rules**.  

Users can upload up to **5 documents** (PDF, DOCX, TXT, PPTX, CSV, XLSX, etc.) and ask natural language questions.  
The system:
- Extracts text from documents  
- Embeds chunks using **sentence-transformers**  
- Stores embeddings in **FAISS**  
- Generates answers using **Llama 3 on Groq API**  

Built with **LangChain** and a **Gradio UI** for easy interaction.  
This project fulfills the academic requirements for a RAG-based Q&A system, supporting an English/Urdu toggle (Urdu placeholder included) and conversational memory.

---

## 🚀 Features

- **Document Ingestion**  
  - Supports: PDF (PyMuPDF, PyPDF2, pdfplumber fallbacks), DOCX, TXT, PPTX, CSV, XLSX  

- **Text Processing**  
  - Semantic chunking (1000-token chunks, fallback to recursive splitting)  

- **RAG Pipeline**  
  - Embeddings: `all-MiniLM-L6-v2`  
  - Vector Store: **FAISS**  
  - Retrieval: Top-8 chunks  
  - LLM: **Llama3-8b-8192** via Groq  

- **Interface**  
  - Gradio Web UI  
  - File upload  
  - Chat window  
  - Language dropdown (English/Urdu)  
  - Clear button  

- **Prompt Engineering**  
  - Structured template to minimize hallucinations  
  - Markdown-formatted answers (headings, bullet points)  

- **Evaluation**  
  - Tested with **UG Prospectus 2024**  
  - Accuracy: ~3/5 for fee queries (minor hallucinations observed)  

- **Optional**  
  - Urdu translation (placeholder only)  
  - Source metadata stored but not displayed in UI  

## 📂 Project Structure
giki-chatbot/
├── main.py                 # Core logic: document loading, RAG chain, chatbot response
├── interface/
│   └── ui.py               # Gradio UI components and styling
├── requirements.txt        # Dependencies
├── UG-Prospectus-2024.pdf  # Sample document (282 pages)
├── README.md               # Documentation
└── .env                    # Environment variables (e.g., GROQ_API_KEY; add to .gitignore)
