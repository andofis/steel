@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10+ was not found in PATH.
  pause
  exit /b 1
)
title Hikrobot Four Camera Capture
python "%~dp0server.py"
if errorlevel 1 pause
