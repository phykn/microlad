@echo off
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python gen_fake.py %*
pause
