@echo off
echo ========================================
echo   Pregnancy Health RAG System Launcher
echo ========================================
echo.

rem Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

rem Check if Ollama is running
echo Checking Ollama...
curl -s http://localhost:11434 > nul 2>&1
if errorlevel 1 (
    echo WARNING: Ollama may not be running
    echo Starting Ollama...
    start "" ollama serve
    timeout /t 3 /nobreak > nul
)

echo.
echo Starting Backend API...
start "Backend API" cmd /k "python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo.
echo Starting Frontend UI...
start "Frontend UI" cmd /k "streamlit run frontend/app.py --server.port 8501"

echo.
echo ========================================
echo   Application Started Successfully!
echo ========================================
echo.
echo   Frontend UI: http://localhost:8501
echo   Backend API: http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo.
echo Press any key to open the application in your browser...
pause > nul

start http://localhost:8501
