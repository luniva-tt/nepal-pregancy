from src.ingest import IngestionPipeline
from src.rag_pipeline import PregnancyRAG
from src.risk_engine import RiskEvaluator
from src.report_generator import PDFReportGenerator
import sys

def main():
    print("=== Pregnancy Health RAG System ===")
    
    # 1. Ingestion Check
    # In a real app, this might be a separate admin command
    print("Initializing System...")
    try:
        rag = PregnancyRAG()
    except ValueError:
        print("Vector store not found. Running ingestion...")
        ingestion = IngestionPipeline()
        ingestion.create_vector_store()
        rag = PregnancyRAG()

    risk_engine = RiskEvaluator()
    report_gen = PDFReportGenerator()
    
    # Session Data
    conversation_log = []
    patient_data = {"name": "Jane Doe", "age": 28, "week": 24} # Dummy data
    
    while True:
        print("\nOptions:")
        print("1. Ask a question")
        print("2. Check Vitals (Risk Assessment)")
        print("3. Generate Report & Exit")
        choice = input("Select option (1-3): ")

        if choice == "1":
            query = input("\nEnter your question: ")
            print("Thinking...")
            response = rag.ask(query)
            answer = response["answer"]
            print(f"\nResponse: {answer}")
            
            # Show sources for research-grade transparency
            print("\nSources:")
            for doc in response["source_docs"]:
                print(f"- {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', '?')})")
            
            conversation_log.append((query, answer))

        elif choice == "2":
            print("\nEnter Vitals:")
            try:
                bp_sys = int(input("Systolic BP (e.g., 120): "))
                bp_dia = int(input("Diastolic BP (e.g., 80): "))
                hr = int(input("Heart Rate (bpm): "))
                glucose = int(input("Glucose (mg/dL): "))
                
                vitals = {
                    "bp_systolic": bp_sys,
                    "bp_diastolic": bp_dia,
                    "heart_rate": hr,
                    "glucose": glucose
                }
                
                assessment = risk_engine.assess_risk(vitals)
                print(f"\n*** RISK ASSESSMENT: {assessment['risk_level']} ***")
                for warn in assessment["warnings"]:
                    print(f"- {warn}")
                
                # Store for report
                patient_data["risk_assessment"] = assessment
                
            except ValueError:
                print("Invalid input. Please enter numbers.")

        elif choice == "3":
            if "risk_assessment" not in patient_data:
                print("Warning: No risk assessment performed. defaulting to unknown.")
                patient_data["risk_assessment"] = {"risk_level": "None Performed", "warnings": []}
            
            filepath = report_gen.generate_report(patient_data, patient_data["risk_assessment"], conversation_log)
            print(f"\nReport generated at: {filepath}")
            print("Exiting...")
            sys.exit(0)

        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()
