# KI Girl Manager - Start Script

Write-Host "🎭 KI Girl Manager wird gestartet..." -ForegroundColor Cyan

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python gefunden: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python nicht gefunden! Bitte Python installieren." -ForegroundColor Red
    exit 1
}

# Check if requirements are installed
Write-Host "`n📦 Prüfe Dependencies..." -ForegroundColor Yellow
$pipList = pip list 2>&1

if ($pipList -notmatch "Flask") {
    Write-Host "⚠️  Flask nicht gefunden. Installiere Dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
} else {
    Write-Host "✅ Dependencies vorhanden" -ForegroundColor Green
}

# Start manager
Write-Host "`n🚀 Starte Manager auf http://localhost:8000" -ForegroundColor Cyan
Write-Host "   Drücke Strg+C zum Beenden`n" -ForegroundColor Gray

python app.py
