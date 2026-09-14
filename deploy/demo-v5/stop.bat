@echo off
cd /d "%~dp0"
docker compose stop
if errorlevel 1 pause
