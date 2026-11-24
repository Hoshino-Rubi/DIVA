@echo off
setlocal
set "ANY2XSAMPA_HOME=%~dp0"
set "ANY2XSAMPA_MODELS=%~dp0local-models.json"
set "ANY2XSAMPA_NOTES="
set "ANY2XSAMPA_GAME_SNAPSHOT="
set "PYTHONHOME="
if /i "%~1"=="cantonese-vsqx" goto singing
"%~dp0runtime\core\python.exe" -B -X utf8 -m any2xsampa.cli %*
exit /b %ERRORLEVEL%

:singing
"%~dp0runtime\phonetic\python.exe" -B -X utf8 -m any2xsampa.cli %*
exit /b %ERRORLEVEL%
