import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import DATA_DIR, DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

class IngestionPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    def load_documents(self):
        """Loads PDFs from the data directory."""
        loader = DirectoryLoader(DATA_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")
        return documents

    def split_documents(self, documents):
        """Splits documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks.")
        return texts

    def create_vector_store(self):
        """Runs the full ingestion pipeline and creating the vector store."""
        print("Starting ingestion pipeline...")
        documents = self.load_documents()
        if not documents:
            print("No documents found in 'data/' directory. Please add PDF guidelines.")
            return None
        
        texts = self.split_documents(documents)
        
        print("Creating vector store (this may take a moment)...")
        db = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=DB_DIR
        )
        # db.persist() # Chroma 0.4+ persists automatically
        print(f"Vector store created at {DB_DIR}")
        return db

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.create_vector_store()
