"""
Pregnancy Health RAG System - Streamlit Frontend
Production-ready web interface for patient management and health monitoring.
"""
import streamlit as st
import requests
import json
from datetime import date, datetime
import os

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Pregnancy Health Monitor",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "token" not in st.session_state:
    st.session_state.token = None
if "user_data" not in st.session_state:
    st.session_state.user_data = None
if "current_patient" not in st.session_state:
    st.session_state.current_patient = None

# Utility functions
def api_call(method, endpoint, data=None, auth_required=True):
    """Make API call with error handling."""
    url = f"{API_BASE_URL}{endpoint}"
    headers = {}
    
    if auth_required and st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=data)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        
        if response.status_code in [200, 201, 204]:
            return response.json() if response.content else None
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        status_code = response.status_code if 'response' in locals() else 'N/A'
        st.error(f"Connection Error ({status_code}): {str(e)}")
        if 'response' in locals() and response:
             st.write("Raw Response:", response.text)
        return None

# Authentication Pages
def login_page():
    """Login page."""
    st.title("🤰 Pregnancy Health Monitor")
    st.subheader("Login")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            data = {"email": email, "password": password}
            result = api_call("POST", "/api/auth/login", data, auth_required=False)
            
            if result:
                st.session_state.token = result["access_token"]
                st.session_state.user_data = {
                    "user_id": result["user_id"],
                    "email": result["email"],
                    "name": result["name"]
                }
                st.success(f"Welcome, {result['name']}!")
                st.rerun()

def register_page():
    """Registration page."""
    st.title("🤰 Pregnancy Health Monitor")
    st.subheader("Register")
    
    with st.form("register_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        age = st.number_input("Age", min_value=18, max_value=50, value=28)
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match!")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters!")
            else:
                data = {
                    "email": email,
                    "password": password,
                    "name": name,
                    "age": age
                }
                result = api_call("POST", "/api/auth/register", data, auth_required=False)
                
                if result:
                    st.session_state.token = result["access_token"]
                    st.session_state.user_data = {
                        "user_id": result["user_id"],
                        "email": result["email"],
                        "name": result["name"]
                    }
                    st.success(f"Account created! Welcome, {result['name']}!")
                    st.rerun()

# Main Application Pages
def dashboard_page():
    """Main dashboard."""
    st.title("Dashboard")
    st.write(f"Welcome, {st.session_state.user_data['name']}!")
    
    # Get patients
    patients = api_call("GET", f"/api/patients?user_token={st.session_state.token}")
    
    if patients:
        st.subheader("Your Patients")
        cols = st.columns(3)
        for idx, patient in enumerate(patients):
            with cols[idx % 3]:
                st.markdown(f"""
                <div style="padding: 20px; border: 2px solid #4CAF50; border-radius: 10px; margin-bottom: 10px;">
                    <h3>{patient['name']}</h3>
                    <p>📅 Week: {patient.get('gestational_week', 'N/A')}</p>
                    <p>🩸 Blood Group: {patient.get('blood_group', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Select {patient['name']}", key=f"select_{patient['patient_id']}"):
                    st.session_state.current_patient = patient
                    st.rerun()
    else:
        st.info("No patients yet. Create your first patient profile!")
    
    # Create new patient
    with st.expander("➕ Add New Patient"):
        with st.form("new_patient_form"):
            name = st.text_input("Patient Name")
            age = st.number_input("Age", min_value=18, max_value=50, value=28)
            blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])
            gestational_week = st.number_input("Gestational Week", min_value=1, max_value=42, value=20)
            due_date = st.date_input("Due Date")
            submit = st.form_submit_button("Create Patient")
            
            if submit:
                data = {
                    "name": name,
                    "age": age,
                    "blood_group": blood_group,
                    "gestational_week": gestational_week,
                    "due_date": str(due_date)
                }
                result = api_call("POST", f"/api/patients?user_token={st.session_state.token}", data)
                if result:
                    st.success(f"Patient {name} created successfully!")
                    st.rerun()

def chat_page():
    """Chat / RAG interface."""
    if not st.session_state.current_patient:
        st.warning("Please select a patient first!")
        return
    
    st.title(f"💬 Chat - {st.session_state.current_patient['name']}")
    
    # Chat history
    patient_id = st.session_state.current_patient['patient_id']
    history = api_call("GET", f"/api/chat/history/{patient_id}?user_token={st.session_state.token}")
    
    if history:
        st.subheader("Conversation History")
        for conv in reversed(history[:5]):  # Show last 5
            with st.expander(f"Q: {conv['question'][:60]}..."):
                st.write(f"**Q:** {conv['question']}")
                st.write(f"**A:** {conv['answer']}")
                if conv['sources']:
                    st.write("**Sources:**")
                    for source in conv['sources']:
                        st.caption(f"- {source.get('source', 'Unknown')} (Page {source.get('page', '?')})")
    
    # Ask question
    st.subheader("Ask a Question")
    question = st.text_area("Your question about pregnancy health:")
    
    if st.button("Get Answer", type="primary"):
        if question:
            with st.spinner("Thinking..."):
                data = {
                    "patient_id": patient_id,
                    "question": question
                }
                result = api_call("POST", f"/api/chat/ask?user_token={st.session_state.token}", data)
                
                if result:
                    st.success("Answer received!")
                    st.write(f"**Answer:** {result['answer']}")
                    
                    if result['sources']:
                        st.write("**Sources:**")
                        for source in result['sources']:
                            st.caption(f"- {source.get('source', 'Unknown')} (Page {source.get('page', '?')})")
                    
                    st.rerun()
        else:
            st.warning("Please enter a question!")

def vitals_page():
    """Vitals input and risk assessment."""
    if not st.session_state.current_patient:
        st.warning("Please select a patient first!")
        return
    
    st.title(f"🩺 Vitals & Risk Assessment - {st.session_state.current_patient['name']}")
    
    # Input vitals
    st.subheader("Record Vitals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bp_systolic = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=200, value=120)
        bp_diastolic = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=130, value=80)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=75)
    
    with col2:
        glucose = st.number_input("Blood Glucose (mg/dL)", min_value=50.0, max_value=300.0, value=90.0)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=12.0, step=0.1)
        weight = st.number_input("Weight (kg)", min_value=40.0, max_value=150.0, value=65.0, step=0.5)
    
    if st.button("Submit Vitals & Assess Risk", type="primary"):
        data = {
            "patient_id": st.session_state.current_patient['patient_id'],
            "bp_systolic": bp_systolic,
            "bp_diastolic": bp_diastolic,
            "heart_rate": heart_rate,
            "glucose": glucose,
            "hemoglobin": hemoglobin,
            "weight": weight
        }
        
        with st.spinner("Analyzing..."):
            result = api_call("POST", f"/api/vitals?user_token={st.session_state.token}", data)
            
            if result:
                st.success("Vitals recorded and risk assessment completed!")
                
                # Display risk level
                risk_level = result['risk_level']
                if risk_level == "High":
                    st.error(f"⚠️ Risk Level: {risk_level}")
                elif risk_level == "Medium":
                    st.warning(f"⚠️ Risk Level: {risk_level}")
                else:
                    st.success(f"✅ Risk Level: {risk_level}")
                
                # Warnings
                if result['warnings']:
                    st.subheader("Warnings")
                    for warning in result['warnings']:
                        st.warning(f"• {warning}")
                
                # Recommendations
                st.subheader("Recommendations")
                for rec in result['recommendations']:
                    st.info(f"• {rec}")
                
                st.write(f"**Clinical Interpretation:** {result['clinical_interpretation']}")

def reports_page():
    """Reports generation and download."""
    if not st.session_state.current_patient:
        st.warning("Please select a patient first!")
        return
    
    st.title(f"📄 Reports - {st.session_state.current_patient['name']}")
    
    # Generate new report
    if st.button("Generate New Report", type="primary"):
        data = {"patient_id": st.session_state.current_patient['patient_id']}
        
        with st.spinner("Generating report..."):
            result = api_call("POST", f"/api/reports/generate?user_token={st.session_state.token}", data)
            
            if result:
                st.success(f"Report generated successfully!")
                st.rerun()
    
    # List existing reports
    patient_id = st.session_state.current_patient['patient_id']
    reports = api_call("GET", f"/api/reports/{patient_id}?user_token={st.session_state.token}")
    
    if reports:
        st.subheader("Previous Reports")
        for report in reports:
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.write(f"📄 {report['report_type']}")
            with col2:
                st.caption(f"Generated: {report['generated_at'][:10]}")
            with col3:
                download_url = f"{API_BASE_URL}/api/reports/download/{report['report_id']}?user_token={st.session_state.token}"
                st.markdown(f"[Download]({download_url})")

# Main Application Flow
def main():
    """Main application logic."""
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        
        if st.session_state.token:
            st.success(f"Logged in as: {st.session_state.user_data['name']}")
            
            # Patient selector
            if st.session_state.current_patient:
                st.info(f"Current Patient: {st.session_state.current_patient['name']}")
                if st.button("Change Patient"):
                    st.session_state.current_patient = None
                    st.rerun()
            
            # Navigation
            page = st.radio(
                "Go to",
                ["Dashboard", "Chat", "Vitals", "Reports"],
                key="navigation"
            )
            
            if st.button("Logout"):
                st.session_state.token = None
                st.session_state.user_data = None
                st.session_state.current_patient = None
                st.rerun()
        else:
            page = st.radio(
                "Choose",
                ["Login", "Register"],
                key="auth_page"
            )
    
    # Main content
    if not st.session_state.token:
        if page == "Login":
            login_page()
        else:
            register_page()
    else:
        if page == "Dashboard":
            dashboard_page()
        elif page == "Chat":
            chat_page()
        elif page == "Vitals":
            vitals_page()
        elif page == "Reports":
            reports_page()

if __name__ == "__main__":
    main()
