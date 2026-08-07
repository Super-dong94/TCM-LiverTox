@echo off
cd /d "%~dp0"

set "TOXHERB_USER_DIR=%~dp0.toxherb_user"
set "TOXHERB_OUTPUT_DIR=%~dp0prediction_outputs"
set "TOXHERB_JOB_DB=%~dp0.toxherb_user\jobs.sqlite"
set "TOXHERB_JOB_LOG_DIR=%~dp0.toxherb_user\job_logs"

echo.
echo [ToxHERB] Starting local hepatotoxicity prediction system...
echo [ToxHERB] The browser will open automatically after the server starts.
echo.

set PYTHON_CMD=python
if exist ".venv\Scripts\python.exe" set PYTHON_CMD=.venv\Scripts\python.exe

"%PYTHON_CMD%" main.py

echo.
echo [ToxHERB] Server stopped.
pause
