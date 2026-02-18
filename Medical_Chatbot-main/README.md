# Pregnancy Health RAG System - Web Application

## 🎯 Overview

A production-ready, full-stack web application for pregnancy health monitoring using RAG (Retrieval-Augmented Generation), risk assessment, and comprehensive patient management.

### Features

✅ **User Authentication** - Secure JWT-based authentication  
✅ **Patient Management** - Multiple patient profiles per user  
✅ **RAG Chat Interface** - Ask questions powered by medical guidelines  
✅ **Vitals Tracking** - Record and monitor patient vitals over time  
✅ **Risk Assessment** - Automatic risk analysis based on clinical thresholds  
✅ **PDF Reports** - Generate professional medical reports  
✅ **Database Integration** - Full persistence with SQLite/PostgreSQL  
✅ **Modern UI** - Clean, responsive Streamlit interface  

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Git

### Installation

```powershell
# 1. Clone the repository
cd c:\Users\Admin.DESKTOP-A1ODH33\Desktop\Medical_Chatbot

# 2. Install backend dependencies
pip install -r backend/requirements.txt

# 3. Install frontend dependencies
pip install -r frontend/requirements.txt

# 4. Copy environment template
copy .env.example .env

# 5. Ensure Ollama is running
ollama serve

# 6. Pull required model (if not already done)
ollama pull mistral
```

### Running the Application

**Terminal 1 - Backend API:**
```powershell
# Start FastAPI backend
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend UI:**
```powershell
# Start Streamlit frontend
streamlit run frontend/app.py --server.port 8501
```

**Access the application:**
- Frontend UI: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

---

## 📁 Project Structure

```
Medical_Chatbot/
│
├── backend/                    # FastAPI Backend
│   ├── app/
│   │   ├── models/            # SQLAlchemy database models
│   │   │   ├── user.py
│   │   │   ├── patient.py
│   │   │   ├── conversation.py
│   │   │   ├── vitals.py
│   │   │   ├── risk_assessment.py
│   │   │   └── report.py
│   │   │
│   │   ├── api/               # REST API endpoints
│   │   │   ├── auth.py        # Authentication
│   │   │   ├── patients.py    # Patient management
│   │   │   ├── chat.py        # RAG chat
│   │   │   ├── vitals.py      # Vitals & risk
│   │   │   └── reports.py     # PDF reports
│   │   │
│   │   ├── services/          # Business logic layer
│   │   │   ├── rag_service.py
│   │   │   ├── risk_service.py
│   │   │   └── report_service.py
│   │   │
│   │   ├── utils/             # Utilities
│   │   │   └── auth.py        # JWT & password hashing
│   │   │
│   │   ├── database.py        # Database configuration
│   │   └── main.py            # FastAPI application
│   │
│   └── requirements.txt
│
├── frontend/                   # Streamlit Frontend
│   ├── app.py                 # Main Streamlit application
│   └── requirements.txt
│
├── src/                        $ Existing RAG modules (integrated)
│   ├── rag_pipeline.py
│   ├── risk_engine.py
│   ├── report_generator.py
│   ├── ingest.py
│   └── config.py
│
├── data/                       # PDF medical guidelines
├── vectorstore/                # ChromaDB vector database
├── pdf_reports/                # Generated reports
├── .env                        # Environment variables
├── .gitignore
└── README.md

```

---

## 🔑 API Endpoints

### Authentication
- `POST /api/auth/register` - Create new user account
- `POST /api/auth/login` - Login and get JWT token
- `GET /api/auth/me` - Get current user info

### Patients
- `POST /api/patients` - Create patient profile
- `GET /api/patients` - List all patients
- `GET /api/patients/{id}` - Get patient details
- `PUT /api/patients/{id}` - Update patient
- `DELETE /api/patients/{id}` - Delete patient (soft)

### Chat/RAG
- `POST /api/chat/ask` - Ask a question
- `GET /api/chat/history/{patient_id}` - Get conversation history
- `DELETE /api/chat/history/{conversation_id}` - Delete conversation

### Vitals & Risk
- `POST /api/vitals` - Record vitals and assess risk
- `GET /api/vitals/{patient_id}` - Get vitals history
- `GET /api/vitals/{patient_id}/latest` - Get latest vitals

### Reports
- `POST /api/reports/generate` - Generate PDF report
- `GET /api/reports/{patient_id}` - List patient reports
- `GET /api/reports/download/{report_id}` - Download PDF

---

## 🗄️ Database Schema

The system uses 6 main tables:
- **users** - User authentication
- **patients** - Patient profiles  
- **conversations** - RAG chat history
- **vitals** - Vital signs measurements
- **risk_assessments** - Risk evaluation results
- **reports** - Generated PDF metadata

---

## 🎨 Using the Application

### 1. Register/Login
- Open http://localhost:8501
- Register a new account or login

### 2. Create Patient Profile
- Navigate to Dashboard
- Click "Add New Patient"
- Fill in patient details

### 3. Chat with RAG
- Select a patient
- Go to "Chat" page
- Ask pregnancy-related questions
- Get evidence-based answers with sources

### 4. Record Vitals
- Go to "Vitals" page
- Enter patient measurements
- Get automatic risk assessment
- View recommendations

### 5. Generate Reports
- Go to "Reports" page
- Click "Generate New Report"
- Download professional PDF report

---

## 🔒 Security Features

- JWT token-based authentication
- Password hashing with bcrypt
- SQL injection protection (SQLAlchemy ORM)
- Input validation (Pydantic)
- CORS configuration
- User authorization checks

---

## 🧪 Testing

```powershell
# Test backend API
# Visit http://localhost:8000/docs for interactive API testing

# Test demo account (after starting the app)
Email: demo@example.com
Password: demo123
```

---

## 📝 Environment Variables

Create a `.env` file with:

```
DATABASE_URL=sqlite:///./pregnancy_health.db
SECRET_KEY=your-secret-random-key-here
CORS_ORIGINS=http://localhost:8501
```

---

## 🐛 Troubleshooting

**Issue: RAG system not initialized**
- Solution: Make sure you have PDF files in the `data/` folder and run ingestion

**Issue: Ollama connection error**
- Solution: Ensure Ollama is running with `ollama serve`

**Issue: Database errors**
- Solution: Delete `pregnancy_health.db` and restart backend to recreate

**Issue: Import errors**
- Solution: Make sure you're running from the project root directory

---

## 🎓 For Semester Project Demo

This is a **production-ready** system with:
- Full authentication and authorization
- Database persistence
- RESTful API design
- Modern UI/UX
- Medical-grade risk assessment
- PDF report generation
- Evidence-based RAG responses

Perfect for demonstrating:
- Full-stack development skills
- API design
- Database modeling
- ML/AI integration
- Security best practices

---

## 📚 Technologies Used

**Backend:**
- FastAPI - Web framework
- SQLAlchemy - ORM
- JWT - Authentication
- Bcrypt - Password hashing

**Frontend:**
- Streamlit - UI framework

**RAG System:**
- LangChain - RAG framework
- ChromaDB - Vector database
- Ollama - LLM (Mistral)
- Sentence Transformers - Embeddings

**Other:**
- ReportLab - PDF generation
- Pydantic - Data validation

---

## 📄 License

For educational and research purposes.

## ⚠️ Medical Disclaimer

This system is for educational and research purposes only and does not replace professional medical consultation.
