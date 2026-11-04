# Script de arranque para la Estación Científica PINN-MRI (FastAPI GPU + React PACS)
Write-Host "Iniciando Backend FastAPI (Puerto 8000) en GPU RTX 4060 Ti..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; python -m uvicorn backend.main:app --reload --port 8000"

Write-Host "Iniciando Frontend React PACS (Puerto 5173)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\frontend'; npm run dev"
