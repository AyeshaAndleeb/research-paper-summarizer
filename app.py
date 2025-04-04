import streamlit as st
import requests
import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load the API key from environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if API key is set
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is not set. Please add it in the Secrets section.")
    st.stop()

# Define Gemini API endpoint
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Set up Sentence Transformer for embedding generation
model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="📄 Research Paper Summarizer & QA (RAG) with Vector DB")
st.title("📄 Research Paper Summarizer & Q&A (RAG) with Vector DB")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to break text into chunks (e.g., paragraphs)
def chunk_text(text, chunk_size=1000):
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 1 <= chunk_size:
            current_chunk += paragraph + "\n"
        else:
            chunks.append(current_chunk)
            current_chunk = paragraph + "\n"
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Function to generate embeddings for each chunk
def generate_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

# Function to call Gemini API for summary or Q&A
def call_gemini_api(prompt):
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    response = requests.post(GEMINI_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

# Upload section
uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type="pdf")

if uploaded_file:
    st.success("✅ PDF uploaded successfully!")
    with st.spinner("Extracting text from PDF..."):
        paper_text = extract_text_from_pdf(uploaded_file)
        # Break the text into chunks
        paper_chunks = chunk_text(paper_text)

        # Generate embeddings for chunks
        chunk_embeddings = generate_embeddings(paper_chunks)

        # Store embeddings in FAISS
        dimension = chunk_embeddings.shape[1]  # Get the dimension of the embeddings
        index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
        faiss_index = faiss.IndexIDMap(index)
        faiss_index.add_with_ids(np.array(chunk_embeddings), np.array(range(len(chunk_embeddings))))

    # Show options: summarize or ask a question
    option = st.radio("What would you like to do?", ("Summarize the paper", "Ask a question"))

    if option == "Summarize the paper":
        if st.button("Generate Summary"):
            with st.spinner("Generating summary using Gemini..."):
                # Use FAISS to retrieve relevant chunks (in this case, the entire document)
                summary = call_gemini_api(f"Please summarize the following research paper:\n{paper_text}")
                if summary:
                    st.subheader("📌 Summary")
                    st.write(summary)

    elif option == "Ask a question":
        user_question = st.text_input("Enter your question about the paper")
        if st.button("Get Answer") and user_question:
            with st.spinner("Thinking..."):
                # Convert the user's question to an embedding
                question_embedding = model.encode([user_question])

                # Perform similarity search in FAISS
                D, I = faiss_index.search(np.array(question_embedding), k=3)  # Retrieve top 3 relevant chunks

                # Combine relevant chunks
                relevant_chunks = "\n".join([paper_chunks[i] for i in I[0]])

                # Pass the relevant chunks to Gemini for question answering
                qa_prompt = f"Here is a research paper:\n{relevant_chunks}\n\nAnswer this question: {user_question}"
                answer = call_gemini_api(qa_prompt)
                if answer:
                    st.subheader("❓ Answer")
                    st.write(answer)