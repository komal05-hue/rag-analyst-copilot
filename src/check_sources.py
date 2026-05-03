from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
vs_path = os.path.join(current_dir, "..", "vectorstore")

vs = Chroma(
    persist_directory=vs_path,
    embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

results = vs.get()
sources = list(set([m['source'] for m in results['metadatas']]))
print("Sources stored in database:")
for s in sources:
    print(s)