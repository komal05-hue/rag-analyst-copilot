# ingest.py - Reads PDFs and stores them in our database

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load API key
load_dotenv()

# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_dir, "..", "data")
vectorstore_path = os.path.join(current_dir, "..", "vectorstore")

print(f"📂 Looking for PDFs in: {data_folder}")
print(f"📂 Saving database to: {vectorstore_path}")

# Step 1 - Find all PDFs
def load_pdfs():
    all_documents = []
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
    
    if len(pdf_files) == 0:
        print("❌ No PDF files found in data folder!")
        return []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_folder, pdf_file)
        print(f"📄 Reading: {pdf_file}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)
    
    print(f"✅ Total pages loaded: {len(all_documents)}")
    return all_documents

# Step 2 - Split into chunks
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Total chunks created: {len(chunks)}")
    return chunks

# Step 3 - Store in ChromaDB
def store_in_database(chunks):
    print("⏳ Creating embeddings and storing in database...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=vectorstore_path
    )
    
    # Verify it saved correctly
    count = vectorstore._collection.count()
    print(f"📊 Total chunks saved in database: {count}")
    print("✅ All documents stored successfully!")
    return vectorstore

# Run everything
if __name__ == "__main__":
    print("🚀 Starting document ingestion...")
    docs = load_pdfs()
    if docs:
        chunks = split_documents(docs)
        store_in_database(chunks)
        print("🎉 Done! Your documents are ready to be searched.")