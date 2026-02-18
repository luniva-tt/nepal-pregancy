from fastapi import FastAPI, HTTPException, Depends, status, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text, create_engine, DateTime
from datetime import datetime
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base
import hashlib
import base64
import bcrypt
import os
import joblib
import pandas as pd
import numpy as np
import re
from typing import Optional, Dict, Any

# --- Configuration ---
DATABASE_URL = "sqlite:///./mamucare.db"
SECRET_KEY = "MamuCare_Super_Secret_Key_123"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Artifact Paths (must match what the notebook saves)
MODEL_PATH = os.path.join(MODELS_DIR, 'best_maternal_risk_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'ml_scaler.pkl')
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, 'feature_names.pkl')

# --- Database Setup ---
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- ML Helpers ---
# These encoding maps MUST match what the notebook uses during training.
ORDINAL_MAP = {'1st': 1, '2nd': 2, '3rd': 3}
SEVERITY_MAP = {'No': 1, 'Medium': 2, 'Higher': 3}
BINARY_YN = {'Yes': 1, 'No': 0}
BINARY_PN = {'Positive': 1, 'Negative': 0}
NORMAL_MAP = {'Normal': 0, 'Abnormal': 1}

def _feet_to_cm(val: Any) -> float:
    """Convert '5.3' style feet.inches string to centimetres (matches notebook)."""
    try:
        val = str(val).replace("'", "").strip()
        parts = val.split('.')
        feet = int(parts[0])
        inches = int(parts[1]) if len(parts) > 1 else 0
        return round((feet * 12 + inches) * 2.54, 1)
    except Exception:
        return np.nan

def _extract_number(val: Any) -> float:
    """Pull the first number out of a string like '38 week' or '140m'."""
    try:
        matches = re.findall(r'[\d.]+', str(val))
        return float(matches[0]) if matches else np.nan
    except Exception:
        return np.nan

def predict_risk_ml(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run prediction using the same preprocessing as the notebook."""
    try:
        if not os.path.exists(MODEL_PATH):
            return {"error": "Model not trained yet. Run the notebook first!"}

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)

        # --- Parse numeric fields exactly like the notebook ---
        age = float(input_data.get('age', 25))
        gestational_age_weeks = _extract_number(input_data.get('gestational_age_weeks', '0'))
        weight_kg = _extract_number(input_data.get('weight_kg', '60'))
        height_cm = _feet_to_cm(input_data.get('height_cm', '5.3'))
        fetal_heart_rate = _extract_number(input_data.get('fetal_heart_rate', '0'))
        systolic_bp = float(input_data.get('systolic_bp', 120))
        diastolic_bp = float(input_data.get('diastolic_bp', 80))

        # BMI (derived)
        bmi = round(weight_kg / ((height_cm / 100) ** 2), 2) if height_cm and height_cm > 0 else np.nan

        # --- Apply the same categorical encodings as the notebook ---
        previous_pregnancies = ORDINAL_MAP.get(str(input_data.get('previous_pregnancies', '1st')), 1)
        tt_vaccine = ORDINAL_MAP.get(str(input_data.get('tt_vaccine', '1st')), 1)
        anemia = SEVERITY_MAP.get(str(input_data.get('anemia', 'No')), 1)
        jaundice = SEVERITY_MAP.get(str(input_data.get('jaundice', 'No')), 1)
        urine_albumin = SEVERITY_MAP.get(str(input_data.get('urine_albumin', 'No')), 1)
        urine_sugar = BINARY_YN.get(str(input_data.get('urine_sugar', 'No')), 0)
        vdrl_test = BINARY_PN.get(str(input_data.get('vdrl_test', 'Negative')), 0)
        hbsag_test = BINARY_PN.get(str(input_data.get('hbsag_test', 'Negative')), 0)
        fetal_position = NORMAL_MAP.get(str(input_data.get('fetal_position', 'Normal')), 0)
        fetal_movement = NORMAL_MAP.get(str(input_data.get('fetal_movement', 'Normal')), 0)

        # --- Build feature vector in the EXACT same column order as notebook training ---
        feature_dict = {
            'age': age,
            'previous_pregnancies': previous_pregnancies,
            'tt_vaccine': tt_vaccine,
            'gestational_age_weeks': gestational_age_weeks,
            'weight_kg': weight_kg,
            'height_cm': height_cm,
            'anemia': anemia,
            'jaundice': jaundice,
            'fetal_position': fetal_position,
            'fetal_movement': fetal_movement,
            'fetal_heart_rate': fetal_heart_rate,
            'urine_albumin': urine_albumin,
            'urine_sugar': urine_sugar,
            'vdrl_test': vdrl_test,
            'hbsag_test': hbsag_test,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'bmi': bmi,
        }

        # Re-order to match training feature order
        X_row = pd.DataFrame([{fname: feature_dict.get(fname, 0) for fname in feature_names}])
        X_row.fillna(X_row.median(), inplace=True)

        # Scale and predict
        X_scaled = scaler.transform(X_row)
        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        confidence = float(max(proba))

        risk_label = 'Risk' if prediction == 1 else 'No Risk'

        # --- Clinical safety-net: override to Risk for medically dangerous vitals ---
        clinical_flags = []
        if systolic_bp >= 160 or systolic_bp < 90:
            clinical_flags.append('abnormal systolic BP')
        if diastolic_bp >= 100 or diastolic_bp < 60:
            clinical_flags.append('abnormal diastolic BP')
        if bmi and not np.isnan(bmi) and (bmi < 16 or bmi > 40):
            clinical_flags.append('abnormal BMI')
        if fetal_heart_rate and not np.isnan(fetal_heart_rate) and (fetal_heart_rate > 180 or fetal_heart_rate < 100):
            clinical_flags.append('abnormal fetal heart rate')
        if weight_kg and weight_kg < 35:
            clinical_flags.append('very low weight')

        if clinical_flags and risk_label == 'No Risk':
            risk_label = 'Risk'
            confidence = max(confidence, 0.75)

        return {
            'risk_level': risk_label,
            'probability': round(confidence, 4),
            'details': {
                'bmi': bmi if bmi and not np.isnan(bmi) else None
            }
        }
    except Exception as e:
        return {'error': str(e)}

# --- Models ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(String)  # 'mother' or 'doctor'
    full_name = Column(String)
    health_records = relationship("HealthRecord", back_populates="user")

class HealthRecord(Base):
    __tablename__ = "health_records"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    age = Column(Integer)
    systolic_bp = Column(Integer)
    diastolic_bp = Column(Integer)
    weight_kg = Column(Float)
    height_cm = Column(String)
    gestational_age_weeks = Column(String)
    previous_pregnancies = Column(Integer)
    fetal_heart_rate = Column(String)
    tt_vaccine = Column(String)
    anemia = Column(String)
    jaundice = Column(String)
    fetal_position = Column(String)
    urine_albumin = Column(String)
    urine_sugar = Column(String)
    hbsag_test = Column(String)
    vdrl_test = Column(String)
    fetal_movement = Column(String)
    body_temp = Column(Float)
    bmi = Column(Float)
    risk_level = Column(String)
    prediction_date = Column(DateTime, default=datetime.now)
    
    user = relationship("User", back_populates="health_records")

Base.metadata.create_all(bind=engine)

# --- App Setup ---
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Security Helpers ---
def hash_password(password: str) -> str:
    # Pre-hash with SHA-256 and Base64 encode to bypass bcrypt 72-byte limit
    sha256_hash = hashlib.sha256(password.encode()).digest()
    b64_hash = base64.b64encode(sha256_hash)
    return bcrypt.hashpw(b64_hash, bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    sha256_hash = hashlib.sha256(password.encode()).digest()
    b64_hash = base64.b64encode(sha256_hash)
    return bcrypt.checkpw(b64_hash, hashed.encode())

# --- Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Template Helper ---
def render_template(request: Request, name: str, context: Optional[dict] = None):
    try:
        if context is None:
            context = {}
        user_id = request.session.get("user_id")
        db = SessionLocal()
        current_user = None
        if user_id:
            current_user = db.query(User).filter(User.id == user_id).first()
        db.close()
        
        full_context = {
            "request": request,
            "current_user": current_user,
            **context
        }
        return templates.TemplateResponse(name, full_context)
    except Exception as e:
        import traceback
        error_msg = f"Error rendering template {name}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        with open("error_log.txt", "a") as f:
            f.write(error_msg + "\n")
        raise HTTPException(status_code=500, detail="Internal Server Error during rendering")

# --- Routes ---

@app.get("/", response_class=HTMLResponse, name="index")
async def index(request: Request):
    return render_template(request, "index.html")

@app.get("/about", response_class=HTMLResponse, name="about")
async def about(request: Request):
    return render_template(request, "about.html")

@app.get("/contact", response_class=HTMLResponse, name="contact")
async def contact(request: Request):
    return render_template(request, "contact.html")

@app.get("/blog", response_class=HTMLResponse, name="blog")
async def blog(request: Request):
    return render_template(request, "blog.html")

@app.get("/developer", response_class=HTMLResponse, name="developer")
async def developer(request: Request):
    return render_template(request, "developer.html")

@app.get("/signup", response_class=HTMLResponse, name="signup")
async def signup(request: Request):
    return render_template(request, "signup.html")

@app.post("/signup", name="signup_post")
async def signup_post(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    full_name: str = Form(...),
    role: str = Form(...),
    db: Session = Depends(get_db)
):
    if db.query(User).filter(User.username == username).first():
        return render_template(request, "signup.html", {"error": "Username already taken"})
    
    # Remove specific name validation as per user request (allow numbers etc)
    
    new_user = User(
        username=username,
        email=email,
        password_hash=hash_password(password),
        full_name=full_name,
        role=role
    )
    db.add(new_user)
    db.commit()
    return RedirectResponse(url="/login", status_code=303)

@app.get("/login", response_class=HTMLResponse, name="login")
async def login(request: Request):
    return render_template(request, "login.html")

@app.post("/login", name="login_post")
async def login_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == username).first()
    if user and verify_password(password, user.password_hash):
        request.session["user_id"] = user.id
        request.session["role"] = user.role
        if user.role == "doctor":
            return RedirectResponse(url="/doctor-dashboard", status_code=303)
        return RedirectResponse(url="/dashboard", status_code=303)
    return render_template(request, "login.html", {"error": "Invalid credentials"})

@app.get("/logout", name="logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/")

@app.get("/dashboard", response_class=HTMLResponse, name="dashboard")
async def dashboard(request: Request, db: Session = Depends(get_db)):
    user_id = request.session.get("user_id")
    if not user_id or request.session.get("role") != "mother":
        return RedirectResponse(url="/login")
    
    user = db.query(User).filter(User.id == user_id).first()
    records = db.query(HealthRecord).filter(HealthRecord.user_id == user_id).order_by(HealthRecord.prediction_date.desc()).all()
    # If using attachments expectations, result might be the latest risk level
    latest_result = records[0].risk_level if records else None
    return render_template(request, "dashboard.html", {"records": records, "user": user, "result": latest_result})

@app.post("/predict", name="predict")
async def predict(
    request: Request,
    age: int = Form(...),
    systolic_bp: int = Form(...),
    diastolic_bp: int = Form(...),
    weight_kg: float = Form(...),
    height_cm: str = Form(...),
    gestational_age_weeks: str = Form(...),
    previous_pregnancies: str = Form(...),
    fetal_heart_rate: str = Form(...),
    tt_vaccine: str = Form(...),
    anemia: str = Form(...),
    jaundice: str = Form(...),
    urine_sugar: str = Form("No"),
    hbsag_test: str = Form("Negative"),
    vdrl_test: str = Form("Negative"),
    fetal_position: str = Form(...),
    urine_albumin: str = Form(...),
    fetal_movement: str = Form(...),
    db: Session = Depends(get_db)
):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login")
    
    input_data = {
        "age": age,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "gestational_age_weeks": gestational_age_weeks,
        "previous_pregnancies": previous_pregnancies,
        "fetal_heart_rate": fetal_heart_rate,
        "tt_vaccine": tt_vaccine,
        "anemia": anemia,
        "jaundice": jaundice,
        "urine_sugar": urine_sugar,
        "hbsag_test": hbsag_test,
        "vdrl_test": vdrl_test,
        "fetal_position": fetal_position,
        "urine_albumin": urine_albumin,
        "fetal_movement": fetal_movement
    }
    
    prediction = predict_risk_ml(input_data)
    
    if "error" in prediction:
        records = db.query(HealthRecord).filter(HealthRecord.user_id == user_id).all()
        return render_template(request, "dashboard.html", {
            "records": records,
            "error": f"Prediction failed: {prediction['error']}"
        })

    new_record = HealthRecord(
        user_id=user_id,
        age=age,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        weight_kg=weight_kg,
        height_cm=height_cm,
        gestational_age_weeks=gestational_age_weeks,
        previous_pregnancies=previous_pregnancies,
        fetal_heart_rate=fetal_heart_rate,
        tt_vaccine=tt_vaccine,
        anemia=anemia,
        jaundice=jaundice,
        urine_sugar=urine_sugar,
        hbsag_test=hbsag_test,
        vdrl_test=vdrl_test,
        fetal_position=fetal_position,
        urine_albumin=urine_albumin,
        fetal_movement=fetal_movement,
        bmi=prediction.get("details", {}).get("bmi"),
        risk_level=prediction.get("risk_level")
    )
    db.add(new_record)
    db.commit()
    
    return RedirectResponse(url="/dashboard", status_code=303)

@app.get("/doctor-dashboard", response_class=HTMLResponse, name="doctor_dashboard")
async def doctor_dashboard(request: Request, db: Session = Depends(get_db)):
    if request.session.get("role") != "doctor":
        return RedirectResponse(url="/login")
    
    # Get mothers and their latest records
    mothers = db.query(User).filter(User.role == "mother").all()
    # For each mother, find latest record
    patient_list = []
    for m in mothers:
        latest = db.query(HealthRecord).filter(HealthRecord.user_id == m.id).order_by(HealthRecord.id.desc()).first()
        patient_list.append({"user": m, "latest": latest})
        
    high_risk_count = sum(1 for p in patient_list if p['latest'] and p['latest'].risk_level == 'Risk')
    low_risk_count = sum(1 for p in patient_list if not p['latest'] or p['latest'].risk_level != 'Risk')
    medium_risk_count = 0 # Not currently used in logic but placeholder for template
        
    return render_template(request, "doctor_dashboard.html", {
        "mothers": mothers, 
        "patient_list": patient_list,
        "user": db.query(User).filter(User.id == request.session.get("user_id")).first(),
        "high_risk_count": high_risk_count,
        "low_risk_count": low_risk_count,
        "medium_risk_count": medium_risk_count
    })

@app.get("/patient/{patient_id}", response_class=HTMLResponse, name="patient_detail")
async def patient_detail(request: Request, patient_id: int, db: Session = Depends(get_db)):
    if request.session.get("role") != "doctor":
        return RedirectResponse(url="/login")
    
    patient = db.query(User).filter(User.id == patient_id).first()
    records = db.query(HealthRecord).filter(HealthRecord.user_id == patient_id).order_by(HealthRecord.prediction_date.desc()).all()
    return render_template(request, "patient_detail.html", {"patient": patient, "records": records, "user": patient})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_fastapi:app", host="127.0.0.1", port=8000, reload=True)
