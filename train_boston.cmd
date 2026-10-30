@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_boston.ps1" %*
exit /b %ERRORLEVEL%
