@echo off
cd /d "C:\Users\andre\OneDrive\Documents\TrafficManagementSystem"
start "" "C:\Users\andre\AppData\Local\Programs\Python\Python312\python.exe" traffic_detection.py
timeout /t 5 /nobreak >nul
start http://127.0.0.1:5000/
exit
