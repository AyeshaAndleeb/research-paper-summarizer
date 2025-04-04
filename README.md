# Research Paper Summarizer & Q&A with RAG (Vector DB)

## 🚀 Overview

The **Research Paper Summarizer & Q&A** application allows users to upload research papers in PDF format, extract their content, and either summarize the document or answer specific questions based on the content. The application uses **Retrieval-Augmented Generation (RAG)** with **FAISS** for efficient similarity search and **Gemini AI** for content generation.

### Key Features:
- **Summarize**: Get concise summaries of the entire research paper.
- **Q&A**: Ask specific questions related to the research paper and get relevant answers.
- **Document Processing**: Efficiently extract and process text from PDF documents using embeddings.

---

## 🌐 Deployed Application

You can access the live version of the app on **Hugging Face**:  
[Research Paper Summarizer & Q&A](https://huggingface.co/spaces/Ayesha003/Research-papers-summarizer)

---

## 🛠️ Technologies Used

- **Streamlit**: Web framework for building interactive apps.
- **Gemini AI**: For summarization and answering questions.
- **FAISS**: Library for fast and efficient similarity search.
- **Sentence-Transformers**: For generating document embeddings.
- **PyPDF2**: For PDF text extraction.
- **dotenv**: For securely managing environment variables.

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AyeshaAndleeb/research-paper-summarizer.git
cd research-paper-summarizer
