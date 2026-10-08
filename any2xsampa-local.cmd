@echo off
setlocal
set "ANY2XSAMPA_HOME=%~dp0"
set "PYTHONPATH=%ANY2XSAMPA_HOME%src"
"%ANY2XSAMPA_HOME%.venv\Scripts\python.exe" -X utf8 -m any2xsampa.cli %*
exit /b %errorlevel%
