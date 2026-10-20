@echo off
echo ===================================================
echo Starting Pakistani ANPR Live Prediction Studio
echo URL: http://127.0.0.1:8000
echo ===================================================
call .\yolo_env\Scripts\activate.bat
.\yolo_env\Scripts\python.exe -m uvicorn web_app.app:app --host 127.0.0.1 --port 8000
pause
