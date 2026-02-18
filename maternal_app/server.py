"""
Unified Maternal Pregnancy Assistant — FastAPI Server
Integrates Risk Engine, MedChat RAG, and PregnancySafe Route.
"""
import os
import sys
import json
from datetime import date, datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

# ── Path Setup ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(BASE_DIR)

# Add sibling projects to path for imports
sys.path.insert(0, os.path.join(WORKSPACE, "Medical_Chatbot-main"))
sys.path.insert(0, os.path.join(WORKSPACE, "pregnancysafe-main"))

from database import init_db, get_db, User, SymptomLog, RiskHistory, ChatHistory
from auth import hash_password, verify_password, create_token, decode_token

# ── Lazy-loaded engines ─────────────────────────────────────────────
risk_engine = None
rag_engine = None
rag_failed = False      # Set True after first RAG init failure to avoid retrying
route_engine = None


def get_risk_engine():
    global risk_engine
    if risk_engine is None:
        from src.risk_engine import RiskEvaluator
        risk_engine = RiskEvaluator()
    return risk_engine


def get_route_engine():
    global route_engine
    if route_engine is None:
        from route_engine import SafeRouter
        road = os.path.join(WORKSPACE, "pregnancysafe-main", "dataset", "nepal_roads_full.gpkg")
        clinic = os.path.join(WORKSPACE, "pregnancysafe-main", "dataset", "nepal_hospitals_full.geojson")
        if os.path.exists(road) and os.path.exists(clinic):
            route_engine = SafeRouter(road, clinic)
        else:
            print(f"WARNING: Route data files not found")
    return route_engine


# ── Articles Data ───────────────────────────────────────────────────
ARTICLES_PATH = os.path.join(BASE_DIR, "data", "articles.json")
with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
    ARTICLES = json.load(f)


# ── FastAPI App ─────────────────────────────────────────────────────
app = FastAPI(title="Maternal Health Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
async def startup():
    init_db()
    print("Database initialized.")
    # Pre-load risk engine (lightweight)
    get_risk_engine()
    print("Risk engine ready.")


@app.get("/")
async def index():
    return FileResponse(os.path.join(static_dir, "index.html"))


# ══════════════════════════════════════════════════════════════════
#  AUTH API
# ══════════════════════════════════════════════════════════════════

class RegisterReq(BaseModel):
    name: str
    email: str
    password: str
    lmp_date: Optional[str] = None      # YYYY-MM-DD
    due_date: Optional[str] = None
    emergency_contact: Optional[str] = None

class LoginReq(BaseModel):
    email: str
    password: str


@app.post("/api/auth/register")
def register(req: RegisterReq, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(400, "Email already registered")

    user = User(
        name=req.name,
        email=req.email,
        password_hash=hash_password(req.password),
        lmp_date=date.fromisoformat(req.lmp_date) if req.lmp_date else None,
        due_date=date.fromisoformat(req.due_date) if req.due_date else None,
        emergency_contact=req.emergency_contact,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "token": create_token(user.user_id),
        "user": _user_dict(user),
    }


@app.post("/api/auth/login")
def login(req: LoginReq, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return {
        "token": create_token(user.user_id),
        "user": _user_dict(user),
    }


@app.get("/api/auth/me")
def me(token: str = Query(...), db: Session = Depends(get_db)):
    user = _get_user(token, db)
    return _user_dict(user)


def _get_user(token: str, db: Session) -> User:
    uid = decode_token(token)
    if not uid:
        raise HTTPException(401, "Invalid token")
    user = db.query(User).filter(User.user_id == uid).first()
    if not user:
        raise HTTPException(401, "User not found")
    return user


def _user_dict(u: User) -> dict:
    return {
        "user_id": u.user_id,
        "name": u.name,
        "email": u.email,
        "lmp_date": str(u.lmp_date) if u.lmp_date else None,
        "due_date": str(u.due_date) if u.due_date else None,
        "emergency_contact": u.emergency_contact,
        "pregnancy_week": u.pregnancy_week(),
    }


# ══════════════════════════════════════════════════════════════════
#  ARTICLES API
# ══════════════════════════════════════════════════════════════════

@app.get("/api/articles")
def get_articles(week: int = Query(None)):
    """Return article for the given week (closest match) or all."""
    if week is not None:
        # Find the closest article <= week
        best = ARTICLES[0]
        for art in ARTICLES:
            if art["week"] <= week:
                best = art
        return best
    return ARTICLES


# ══════════════════════════════════════════════════════════════════
#  RISK PREDICTION API
# ══════════════════════════════════════════════════════════════════

class RiskReq(BaseModel):
    token: str
    bp_systolic: int
    bp_diastolic: int
    heart_rate: int
    glucose: int


@app.post("/api/risk/assess")
def assess_risk(req: RiskReq, db: Session = Depends(get_db)):
    user = _get_user(req.token, db)
    engine = get_risk_engine()

    vitals = {
        "bp_systolic": req.bp_systolic,
        "bp_diastolic": req.bp_diastolic,
        "heart_rate": req.heart_rate,
        "glucose": req.glucose,
    }
    result = engine.assess_risk(vitals)

    # Save to history
    record = RiskHistory(
        user_id=user.user_id,
        bp_systolic=req.bp_systolic,
        bp_diastolic=req.bp_diastolic,
        heart_rate=req.heart_rate,
        glucose=req.glucose,
        risk_level=result["risk_level"],
        warnings=json.dumps(result["warnings"]),
    )
    db.add(record)
    db.commit()

    return result


@app.get("/api/risk/history")
def risk_history(token: str = Query(...), db: Session = Depends(get_db)):
    user = _get_user(token, db)
    rows = (
        db.query(RiskHistory)
        .filter(RiskHistory.user_id == user.user_id)
        .order_by(RiskHistory.assessed_at.desc())
        .limit(20)
        .all()
    )
    return [
        {
            "record_id": r.record_id,
            "bp_systolic": r.bp_systolic,
            "bp_diastolic": r.bp_diastolic,
            "heart_rate": r.heart_rate,
            "glucose": r.glucose,
            "risk_level": r.risk_level,
            "warnings": json.loads(r.warnings) if r.warnings else [],
            "assessed_at": str(r.assessed_at),
        }
        for r in rows
    ]


# ══════════════════════════════════════════════════════════════════
#  MEDCHAT API (RAG)
# ══════════════════════════════════════════════════════════════════

class ChatReq(BaseModel):
    token: str
    question: str


@app.post("/api/chat/ask")
def chat_ask(req: ChatReq, db: Session = Depends(get_db)):
    user = _get_user(req.token, db)

    global rag_engine, rag_failed
    answer = ""
    week = user.pregnancy_week()

    # Only try RAG if it hasn't already failed
    if not rag_failed:
        try:
            if rag_engine is None:
                from src.rag_pipeline import PregnancyRAG
                rag_engine = PregnancyRAG()

            context_q = f"[Patient is at pregnancy week {week}] {req.question}" if week else req.question
            result = rag_engine.ask(context_q)
            answer = result["answer"]
        except Exception as e:
            print(f"RAG unavailable (will use fallback from now on): {e}")
            rag_failed = True
            rag_engine = None
            answer = _simple_response(req.question, week)
    else:
        answer = _simple_response(req.question, week)

    # Save
    chat = ChatHistory(user_id=user.user_id, question=req.question, answer=answer)
    db.add(chat)
    db.commit()

    return {"answer": answer}


@app.get("/api/chat/history")
def chat_history(token: str = Query(...), db: Session = Depends(get_db)):
    user = _get_user(token, db)
    rows = (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == user.user_id)
        .order_by(ChatHistory.created_at.desc())
        .limit(30)
        .all()
    )
    return [
        {"chat_id": r.chat_id, "question": r.question, "answer": r.answer, "created_at": str(r.created_at)}
        for r in rows
    ]


def _simple_response(question: str, week: Optional[int]) -> str:
    """Basic fallback when RAG is unavailable."""
    q = question.lower()
    wk = f"At week {week}, " if week else ""

    if any(w in q for w in ["nausea", "morning sickness", "vomit"]):
        return f"{wk}nausea is common, especially in the first trimester. Try eating small, frequent meals and ginger tea. If you cannot keep any food or liquids down, please see your doctor."
    if any(w in q for w in ["headache", "head"]):
        return f"{wk}occasional headaches can be normal. Stay hydrated and rest. Severe or persistent headaches with vision changes should be reported to your doctor immediately as they may indicate pre-eclampsia."
    if any(w in q for w in ["blood pressure", "bp", "preeclampsia", "pre-eclampsia"]):
        return f"{wk}monitoring blood pressure is crucial during pregnancy. Normal is below 120/80. Readings above 140/90 may indicate pre-eclampsia — contact your healthcare provider."
    if any(w in q for w in ["diet", "food", "eat", "nutrition"]):
        return f"{wk}a balanced diet with iron (spinach, beans), calcium (milk, yogurt), folic acid (leafy greens), and protein (lean meat, eggs) is recommended. Avoid raw fish, unpasteurized dairy, and excess caffeine."
    if any(w in q for w in ["exercise", "walk", "workout"]):
        return f"{wk}moderate exercise like walking, swimming, and prenatal yoga is generally safe and beneficial. Avoid high-impact activities and always consult your doctor before starting a new routine."
    if any(w in q for w in ["kick", "movement", "fetal"]):
        return f"{wk}you should feel regular movements after week 20. Count kicks daily — aim for at least 10 movements in 2 hours. Reduced movement should be reported to your doctor immediately."

    return f"{wk}that's a great question. For accurate medical advice, please consult your healthcare provider. You can also use the Risk Check feature for a quick vitals assessment."


# ══════════════════════════════════════════════════════════════════
#  ROUTE / HOSPITAL API (proxying to PregnancySafe)
# ══════════════════════════════════════════════════════════════════

@app.get("/api/route/find")
def find_hospital(
    lat: float = Query(...),
    lon: float = Query(...),
    week: int = Query(20),
    mode: str = Query("routine"),
):
    router = get_route_engine()
    if router is None:
        raise HTTPException(503, "Route engine not available — data files missing")

    results = router.get_safest_route(lat, lon, week=week, mode=mode)
    if not results:
        raise HTTPException(404, "No hospitals found near your location")

    # Return top 3
    out = []
    for r in results[:3]:
        clinic = router.clinics.iloc[r["clinic_idx"]]
        out.append({
            "name": clinic.get("name", "Unknown Facility"),
            "lat": r["lat"],
            "lon": r["lon"],
            "score": r["score"],
            "distance_km": round(r["distance_meters"] / 1000, 2),
            "time_minutes": round(r["time_minutes"], 1),
            "route_segments": r["route_segments"],
        })
    return out


# ══════════════════════════════════════════════════════════════════
#  SYMPTOM TRACKER API
# ══════════════════════════════════════════════════════════════════

class SymptomReq(BaseModel):
    token: str
    symptom: str
    severity: str = "mild"   # mild / moderate / severe
    notes: str = ""


@app.post("/api/symptoms/log")
def log_symptom(req: SymptomReq, db: Session = Depends(get_db)):
    user = _get_user(req.token, db)
    entry = SymptomLog(
        user_id=user.user_id,
        symptom=req.symptom,
        severity=req.severity,
        notes=req.notes,
    )
    db.add(entry)
    db.commit()
    return {"status": "ok", "log_id": entry.log_id}


@app.get("/api/symptoms/history")
def symptom_history(token: str = Query(...), db: Session = Depends(get_db)):
    user = _get_user(token, db)
    rows = (
        db.query(SymptomLog)
        .filter(SymptomLog.user_id == user.user_id)
        .order_by(SymptomLog.logged_at.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "log_id": r.log_id,
            "symptom": r.symptom,
            "severity": r.severity,
            "notes": r.notes,
            "logged_at": str(r.logged_at),
        }
        for r in rows
    ]


# ══════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
