import json
import time
import os
import sys
from typing import List, Dict

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.rag_pipeline import PregnancyRAG
except ImportError:
    print("Error: Could not import PregnancyRAG. Ensure you are running from the project root.")
    sys.exit(1)

class RAGEvaluator:
    def __init__(self, test_set_path: str):
        with open(test_set_path, 'r') as f:
            self.test_set = json.load(f)
        self.rag = PregnancyRAG()
        self.results = []

    def evaluate_retrieval(self, query: str, retrieved_chunks: List[Dict]) -> Dict:
        """Evaluate retrieval precision and relevance (Placeholder for automated scoring)."""
        # In a real scenario, you'd compare chunk IDs against a ground truth
        return {
            "num_chunks": len(retrieved_chunks),
            "retrieval_latency": 0.0  # Captured in main loop
        }

    def run_evaluation(self):
        print(f"🚀 Starting Evaluation on {len(self.test_set)} queries...")
        print("-" * 50)

        for item in self.test_set:
            q = item['question']
            print(f"Testing: {q}")
            
            start_time = time.time()
            response = self.rag.ask(q)
            end_time = time.time()
            
            latency = end_time - start_time
            
            # Extract retrieved context if available in response
            # Note: PregnancyRAG response structure: {"answer": ..., "source_docs": [...]}
            sources = response.get("source_docs", [])
            answer = response.get("answer", "")

            eval_entry = {
                "question": q,
                "category": item['category'],
                "latency": latency,
                "num_sources": len(sources),
                "answer_length": len(answer),
                "sources": [str(doc.page_content[:200]) for doc in sources]
            }
            
            self.results.append(eval_entry)
            print(f"✅ Done (Latency: {latency:.2f}s, Sources: {len(sources)})")

        self.save_results()

    def save_results(self):
        output_path = "evaluation/results.json"
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("-" * 50)
        print(f"📊 Results saved to {output_path}")
        self.print_summary()

    def print_summary(self):
        avg_latency = sum(r['latency'] for r in self.results) / len(self.results)
        avg_sources = sum(r['num_sources'] for r in self.results) / len(self.results)
        
        print("\n--- PERFORMANCE SUMMARY ---")
        print(f"Average Latency: {avg_latency:.2f}s")
        print(f"Average Sources Retrieved: {avg_sources:.1f}")
        print("Note: Generation Quality (Faithfulness) requires a 'Judge' LLM or manual review.")

if __name__ == "__main__":
    evaluator = RAGEvaluator("evaluation/test_set.json")
    evaluator.run_evaluation()
