@echo off
echo Starting OralAI Backend Server...
echo Access the application at http://localhost:8000

:: Open browser after 3 seconds
start "" cmd /c "timeout /t 3 >nul & start http://localhost:8000"

call venv\Scripts\activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
pause