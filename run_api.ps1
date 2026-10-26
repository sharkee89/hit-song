# Pokreće Hit Song Predictor API za Flutter aplikaciju (http://localhost:8000)
$env:PYTHONIOENCODING = "utf-8"
$python = "C:\Users\Sharkee\PycharmProjects\suggesteat-automation\venv\Scripts\python.exe"
Set-Location $PSScriptRoot
& $python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
