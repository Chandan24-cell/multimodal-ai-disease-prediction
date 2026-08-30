@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  py -m venv venv
)

call "venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt

cd frontend
call npm install
call npm run build
cd ..

where docker >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  docker compose -f docker-compose.yml -f docker-compose.host.yml up -d mongodb
)

set "PYTHONPATH=%CD%\backend;%PYTHONPATH%"
echo Application: http://localhost:8000
echo Health:      http://localhost:8000/api/health
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
