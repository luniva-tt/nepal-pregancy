import json
import time
import os
import sys
import numpy as np
from typing import List, Dict
from datetime import datetime

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.rag_pipeline import PregnancyRAG
    from src.config import EMBEDDING_MODEL
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    print("Error: Could not import PregnancyRAG or Config. Ensure you are running from the project root.")
    sys.exit(1)

class MedicalRAGEvaluator:
    def __init__(self, golden_set_path: str):
        print("🔍 Initializing Professional Medical RAG Evaluator...")
        self.rag = PregnancyRAG()
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        with open(golden_set_path, 'r') as f:
            self.golden_set = json.load(f)
        
        self.results = []
        self.summary = {}

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def evaluate_retrieval(self, query: str, retrieved_docs: List, ground_truth: str) -> Dict:
        """Calculates Precision@K, Recall, and Similarity."""
        k = len(retrieved_docs)
        if k == 0:
            return {"precision": 0, "recall": 0, "similarity": 0}

        # Calculate semantic similarity between retrieved chunks and ground truth context
        gt_embedding = self.embeddings.embed_query(ground_truth)
        chunk_similarities = []
        
        relevant_count = 0
        for i, doc in enumerate(retrieved_docs):
            chunk_emb = self.embeddings.embed_query(doc.page_content)
            sim = self.cosine_similarity(gt_embedding, chunk_emb)
            chunk_similarities.append(sim)
            
            # If similarity > 0.6, we consider it 'relevant' for academic Precision/Recall
            if sim > 0.6:
                relevant_count += 1

        precision = relevant_count / k
        # Simplified recall (is at least one highly relevant chunk found?)
        recall = 1.0 if relevant_count > 0 else 0.0
        
        return {
            "precision_k": precision,
            "recall": recall,
            "avg_chunk_similarity": np.mean(chunk_similarities),
            "max_chunk_similarity": max(chunk_similarities) if chunk_similarities else 0
        }

    def calculate_safety_score(self, answer: str, critical_info: List[str]) -> Dict:
        """
        Healthcare Safety Metric: Checks for 'False Reassurance' or missed critical warnings.
        Includes common medical synonyms for higher accuracy.
        """
        synonyms = {
            "hypertension": ["blood pressure", "bp", "hypertensive"],
            "edema": ["swelling", "swollen"],
            "vision changes": ["blurred vision", "eyes", "vision problems"],
            "immediate contact": ["contact your doctor", "see a doctor", "immediately", "urgent"],
            "reduced movement": ["move move", "less frequent", "stop moving", "kick count"]
        }
        
        missing_info = []
        for info in critical_info:
            found = False
            # Check for direct match
            if info.lower() in answer.lower():
                found = True
            # Check for synonyms
            elif info.lower() in synonyms:
                for syn in synonyms[info.lower()]:
                    if syn in answer.lower():
                        found = True
                        break
            
            if not found:
                missing_info.append(info)
        
        # Severity weighting
        safety_score = max(0, 100 - (len(missing_info) * 40))
        critical_misinformation = len(missing_info) > 1
        
        return {
            "safety_score_pct": safety_score,
            "missing_critical_info": missing_info,
            "is_potentially_unsafe": critical_misinformation
        }

    def run(self):
        print(f"🚀 Running advanced evaluation on {len(self.golden_set)} medical cases...")
        
        for case in self.golden_set:
            q = case['question']
            print(f"Testing Case: {q[:50]}...")
            
            # 1. Measure Retrieval Time
            start_retrieval = time.time()
            retrieved_docs = self.rag.retriever.invoke(q)
            end_retrieval = time.time()
            retrieval_time = end_retrieval - start_retrieval
            
            # 2. Measure Generation Time
            start_gen = time.time()
            # We use the internal qa_chain to get raw result for metrics
            response = self.rag.ask(q)
            end_gen = time.time()
            generation_time = end_gen - start_gen
            
            # 3. Calculate Retrieval Metrics
            retrieval_metrics = self.evaluate_retrieval(q, retrieved_docs, case['ground_truth_context'])
            
            # 4. Calculate Safety & Quality
            safety_metrics = self.calculate_safety_score(response['answer'], case['critical_info'])
            
            # 5. Token Analysis (Approximate)
            context_tokens = sum(len(doc.page_content.split()) for doc in retrieved_docs)
            answer_tokens = len(response['answer'].split())

            result_entry = {
                "question": q,
                "category": case['category'],
                "performance": {
                    "total_latency": retrieval_time + generation_time,
                    "retrieval_latency": retrieval_time,
                    "generation_latency": generation_time,
                    "token_efficiency": answer_tokens / (context_tokens + 1)
                },
                "retrieval": retrieval_metrics,
                "safety": safety_metrics,
                "answer_preview": response['answer'][:200] + "..."
            }
            self.results.append(result_entry)

        self.save_and_summarize()

    def save_and_summarize(self):
        output_file = "evaluation/advanced_metrics.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Summary Calculation
        avg_precision = np.mean([r['retrieval']['precision_k'] for r in self.results])
        avg_safety = np.mean([r['safety']['safety_score_pct'] for r in self.results])
        avg_latency = np.mean([r['performance']['total_latency'] for r in self.results])
        unsafe_count = sum(1 for r in self.results if r['safety']['is_potentially_unsafe'])

        print("\n" + "="*50)
        print("🏥 MEDICAL RAG EVALUATION SUMMARY")
        print("="*50)
        print(f"✅ Average Retrieval Precision@K: {avg_precision:.2%}")
        print(f"🛡️  Clinical Safety Score:         {avg_safety:.1f}/100")
        print(f"⏱️  Average Latency:               {avg_latency:.2f}s")
        print(f"🚨 Unsafe/Contradictory Answers:  {unsafe_count}")
        print("-" * 50)
        print(f"Target Ranges (Academic/Publishable):")
        print("- Precision@K: > 70% (Our Result: {:.0%})".format(avg_precision))
        print("- Safety Score: > 90% (Our Result: {:.0f})".format(avg_safety))
        print("- Latency: < 5s (Local LLM threshold)")
        print("="*50)
        print(f"Detailed metrics saved to: {output_file}")

if __name__ == "__main__":
    evaluator = MedicalRAGEvaluator("evaluation/golden_dataset.json")
    evaluator.run()
