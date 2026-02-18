import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from src.config import DB_DIR, EMBEDDING_MODEL, LLM_MODEL, RETRIEVER_K

class PregnancyRAG:
    def __init__(self):
        print("Loading RAG pipeline...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
             raise ValueError(f"Vector Database not found at {DB_DIR}. Please run ingest.py first.")

        self.db = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
        self.retriever = self.db.as_retriever(search_kwargs={"k": RETRIEVER_K})
        
        self.llm = Ollama(model=LLM_MODEL)
        
        # Strict Context-Based Prompt
        prompt_template = """
        You are a specialized Pregnancy Health Assistant using WHO and antenatal guidelines.
        
        Strictly follow these rules:
        1. Answer the question based ONLY on the following context.
        2. If the answer is not in the context, say "I cannot find this information in the provided guidelines."
        3. Do not make up information or use outside knowledge.
        4. Keep answers concise and clinical but empathetic.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])}
        )

    def ask(self, query):
        """
        Queries the RAG system.
        Returns: dict { "answer": str, "source_docs": list }
        """
        result = self.qa_chain.invoke({"query": query})
        return {
            "answer": result["result"],
            "source_docs": result["source_documents"]
        }
