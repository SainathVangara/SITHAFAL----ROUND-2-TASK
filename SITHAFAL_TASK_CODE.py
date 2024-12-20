import os
import streamlit as st
import pickle
import time
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text_data = ""
    for page in reader.pages:
        text_data += page.extract_text() or ""
    return text_data

# Step 2: Chunk Data
def chunk_text(text, chunk_size=500):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=50
    )
    return text_splitter.split_text(text)

# Step 3: Create Embeddings and Vector Store
def create_vector_store(chunks, embeddings):
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# Step 4: Query Handling and Retrieval
def query_vector_store(vector_store, query, qa_model):
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=qa_model, chain_type="stuff", retriever=vector_store.as_retriever()
    )
    return retrieval_qa.run(query)

# Streamlit UI
def main():
    st.title("Task-1: Chat with PDF Using RAG Pipeline")
    st.sidebar.title("PDF Scraper")

    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    process_pdf_clicked = st.sidebar.button("Process PDFs")
    file_path = "faiss_store_openai.pkl"

    # Initialize components
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    qa_model = ChatGroq(model_name="mixtral-8x7b-32768")

    if process_pdf_clicked and uploaded_files:
        text_data = ""
        for uploaded_file in uploaded_files:
            text_data += extract_text_from_pdf(uploaded_file)
        
        # Chunk and Create Vector Store
        chunks = chunk_text(text_data)
        vector_store = create_vector_store(chunks, embeddings)

        # Save vector store to a file
        with open(file_path, "wb") as f:
            pickle.dump(vector_store, f)
        st.success("PDFs processed and embeddings stored!")

    # Query Interface
    main_placeholder = st.empty()
    query = st.text_input("Ask a question about the PDFs:")

    if query:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vector_store = pickle.load(f)
            answer = query_vector_store(vector_store, query, qa_model)
            main_placeholder.write(answer)
        else:
            st.warning("Please process the PDFs first.")

if name == "main":
    main()
