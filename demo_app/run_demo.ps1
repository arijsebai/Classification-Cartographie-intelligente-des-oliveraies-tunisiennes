$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
.\.venv\Scripts\python.exe -m uvicorn demo_app.main:app --host 127.0.0.1 --port 8000
