import os
import sys
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import DB_DIR, EMBEDDING_MODEL

def check_db():
    print(f"Checking database at {DB_DIR}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    count = db._collection.count()
    print(f"Number of documents in vectorstore: {count}")
    
    if count == 0:
        print("WARNING: Vectorstore is EMPTY.")
    else:
        results = db.similarity_search("preeclampsia", k=1)
        if results:
            print(f"Sample retrieval successful: {results[0].page_content[:100]}...")
        else:
            print("WARNING: Similarity search returned NO results.")

if __name__ == "__main__":
    check_db()
