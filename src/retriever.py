# retriever.py - Searches documents and generates answers using Groq

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load API key
load_dotenv()

# Step 1 - Load our existing database
def load_vectorstore():
    print("⏳ Loading document database...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vectorstore_path = os.path.join(current_dir, "..", "vectorstore")
    print(f"📂 Database path: {vectorstore_path}")

    vectorstore = Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embeddings
    )
    
    # Debug - check how many chunks are stored
    count = vectorstore._collection.count()
    print(f"📊 Total chunks in database: {count}")
    
    print("✅ Database loaded successfully!")
    return vectorstore

# Step 2 - Set up Groq AI
def load_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )
    return llm

# Step 3 - Format retrieved documents
def format_docs(docs):
    print(f"📄 Retrieved {len(docs)} chunks")
    return "\n\n".join(doc.page_content for doc in docs)

# Step 4 - Build the QA chain
def build_qa_chain():
    vectorstore = load_vectorstore()
    llm = load_llm()
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )
    
    template = """
    You are a helpful analyst assistant working with multiple documents.
    The context below contains chunks from MULTIPLE different documents.
    When answering, look for information across ALL documents in the context.
    For general questions like "summarize" or "tell me about documents",
    describe EACH document separately with its own section.
    Always cite which document and page you used.
    If information is not in the context, say "I could not find this in the documents."

    Context:
    {context}

    Question:
    {question}

    Answer (cover all documents mentioned in context):
    """
    
    prompt = PromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever

# Step 5 - Ask a question
def ask_question(chain, retriever, question):
    print(f"\n❓ Question: {question}")
    print("⏳ Searching documents and generating answer...")
    
    answer = chain.invoke(question)
    print(f"\n✅ Answer:\n{answer}")
    
    source_docs = retriever.invoke(question)
    print("\n📄 Sources used:")
    for i, doc in enumerate(source_docs):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        print(f"  {i+1}. File: {source} | Page: {page}")

# Test it
if __name__ == "__main__":
    print("🚀 Building QA system...")
    chain, retriever = build_qa_chain()
    ask_question(chain, retriever, "What is this document about?")