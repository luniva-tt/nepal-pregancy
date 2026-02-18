import streamlit as st
import os
import sys

# --- CONFIG & PATH SETUP ---
st.set_page_config(page_title="Pregnancy Health AI (Simplified)", page_icon="🤰", layout="wide")

# Correctly point to the Medical_Chatbot-main src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "Medical_Chatbot-main")
sys.path.append(PROJECT_DIR)

try:
    from src.rag_pipeline import PregnancyRAG
    from src.risk_engine import RiskEvaluator
    from src.report_generator import PDFReportGenerator
    from src.ingest import IngestionPipeline
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# --- SESSION STATE ---
if "rag" not in st.session_state:
    st.session_state.rag = None
if "risk_engine" not in st.session_state:
    st.session_state.risk_engine = RiskEvaluator()
if "report_gen" not in st.session_state:
    st.session_state.report_gen = PDFReportGenerator()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "patient_data" not in st.session_state:
    st.session_state.patient_data = {"risk_assessment": None}

# --- SIDEBAR: SETUP & INGEST ---
with st.sidebar:
    st.title("⚙️ System Setup")
    
    st.info("Medical_Chatbot-main/data")
    uploaded_files = st.file_uploader("Upload Guidelines (PDF)", type="pdf", accept_multiple_files=True)
    
    if st.button("🔄 Ingest Documents"):
        if uploaded_files:
            save_dir = os.path.join(PROJECT_DIR, "data")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            for uploaded_file in uploaded_files:
                with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"Saved {len(uploaded_files)} files.")
        
        with st.spinner("Processing documents..."):
            pipeline = IngestionPipeline()
            pipeline.create_vector_store()
            st.session_state.rag = PregnancyRAG()
        st.success("Ingestion Complete! RAG Ready.")

    if st.session_state.rag is None:
        if st.button("▶️ Load Existing Knowledge"):
            try:
                st.session_state.rag = PregnancyRAG()
                st.success("Loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load: {e}. Try ingesting first.")

# --- TABS Interface ---
tab1, tab2, tab3 = st.tabs(["💬 AI Chat", "🩺 Vitals Check", "📄 Report"])

# --- TAB 1: CHAT ---
with tab1:
    st.header("Pregnancy Health Assistant")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about pregnancy health..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.rag:
                with st.spinner("Thinking..."):
                    result = st.session_state.rag.ask(prompt)
                    response = result["answer"]
                    st.markdown(response)
                    
                    if result["source_docs"]:
                        with st.expander("Sources"):
                            for doc in result["source_docs"]:
                                st.caption(f"- {doc.page_content[:150]}...")
            else:
                response = "⚠️ Please load the knowledge base from the sidebar first."
                st.warning(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

# --- TAB 2: VITALS ---
with tab2:
    st.header("Risk Assessment")
    
    col1, col2 = st.columns(2)
    with col1:
        bp_sys = st.number_input("Systolic BP", 80, 200, 120)
        bp_dia = st.number_input("Diastolic BP", 50, 130, 80)
        hr = st.number_input("Heart Rate", 40, 150, 75)
    with col2:
        glucose = st.number_input("Glucose Check (mg/dL)", 50, 300, 90)
        week = st.number_input("Gestational Week", 1, 42, 20)
    
    if st.button("Assess Risk"):
        vitals = {
            "bp_systolic": bp_sys, 
            "bp_diastolic": bp_dia, 
            "heart_rate": hr, 
            "glucose": glucose
        }
        assessment = st.session_state.risk_engine.assess_risk(vitals)
        
        st.session_state.patient_data["vitals"] = vitals
        st.session_state.patient_data["risk_assessment"] = assessment
        st.session_state.patient_data["week"] = week
        
        lvl = assessment['risk_level']
        if lvl == "High": st.error(f"Risk Level: {lvl}")
        elif lvl == "Medium": st.warning(f"Risk Level: {lvl}")
        else: st.success(f"Risk Level: {lvl}")
        
        for w in assessment['warnings']:
            st.write(f"- {w}")

# --- TAB 3: REPORT ---
with tab3:
    st.header("Generate Report")
    
    name = st.text_input("Patient Name", "Jane Doe")
    age = st.number_input("Age", 18, 50, 28)
    
    if st.button("Download PDF Report"):
        if not st.session_state.patient_data["risk_assessment"]:
            st.warning("Please run a risk assessment first.")
        else:
            st.session_state.patient_data["name"] = name
            st.session_state.patient_data["age"] = age
            
            # Extract chat history as tuples
            chat_log = [(m["content"], "") for m in st.session_state.messages if m["role"] == "user"]
            # Simplified log extraction (user queries only for now or pair them up properly)
            
            path = st.session_state.report_gen.generate_report(
                st.session_state.patient_data, 
                st.session_state.patient_data["risk_assessment"], 
                []
            )
            st.success(f"Report Generated: {path}")
            
            with open(path, "rb") as f:
                st.download_button("Download PDF", f, file_name="Pregnancy_Report.pdf")
