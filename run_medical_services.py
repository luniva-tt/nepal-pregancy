import subprocess
import os
import sys
import time

def main():
    # Base directory of the workspace
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.join(BASE_DIR, "Medical_Chatbot-main")
    
    print(f"🚀 Launching Medical Chatbot System from: {PROJECT_DIR}")
    
    # Check if data directory exists
    data_dir = os.path.join(PROJECT_DIR, "data")
    if not os.path.exists(data_dir):
        print(f"📁 Creating missing directory: {data_dir}")
        os.makedirs(data_dir)
        print("⚠️  PLEASE ADD PDF GUIDELINES TO 'Medical_Chatbot-main/data' FOLDER!")

    # 1. Start Backend (Uvicorn)
    print("\nStarting Backend API (Port 8000)...")
    
    # Use sys.executable to ensure we use the same Python environment
    backend_env = os.environ.copy()
    backend_env["PYTHONPATH"] = PROJECT_DIR + os.pathsep + backend_env.get("PYTHONPATH", "")

    backend_cmd = [
        sys.executable, "-m", "uvicorn", 
        "backend.app.main:app", 
        "--reload", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ]
    
    backend_process = subprocess.Popen(
        backend_cmd, 
        cwd=PROJECT_DIR,
        env=backend_env
    )
    
    # Wait a bit for backend to initialize
    time.sleep(5)
    
    # 2. Start Frontend (Streamlit)
    print("\nStarting Frontend UI (Port 8501)...")
    frontend_cmd = [
        sys.executable, "-m", "streamlit", "run",
        "frontend/app.py",
        "--server.port", "8501"
    ]
    
    frontend_process = subprocess.Popen(
        frontend_cmd,
        cwd=PROJECT_DIR,
        env=backend_env
    )
    
    print("\n✅ System Running!")
    print("Backend Docs: http://localhost:8000/docs")
    print("Frontend App: http://localhost:8501")
    print("Press Ctrl+C to stop all services.")
    
    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        backend_process.terminate()
        frontend_process.terminate()
        print("Done.")

if __name__ == "__main__":
    main()
