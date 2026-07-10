@echo off
REM Chạy GUI bằng python của venv (có torch cu124), KHÔNG dùng python global.
"%~dp0..\.venv\Scripts\python.exe" "%~dp0app.py" %*
