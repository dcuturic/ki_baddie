# ==========================
# Ollama "M4000 Perf" Starter
# ==========================

# --- Einstellungen ---
$BASE_MODEL = "llama3.1:8b-instruct-q4_K_M"   # <--- falls dein Tag anders heißt, hier ändern
$PERF_MODEL = "deeliar-m4000-perf"
$NUM_CTX    = 4096                      # 4096 oder 8192 testen
$THREADS    = 8                         # bei dir: 12 Threads vorhanden -> 8 oft sweet spot

Write-Host "Stopping Ollama (if running)..." -ForegroundColor Cyan
taskkill /F /IM ollama.exe 2>$null | Out-Null
Start-Sleep -Seconds 1

# --- GPU Auswahl (bei dir sehr wahrscheinlich Device 0 = Quadro M4000) ---
# Hinweis: Auf manchen Setups ignoriert Ollama CUDA_VISIBLE_DEVICES. Mit nur 1 GPU ist es egal. :contentReference[oaicite:0]{index=0}
$env:CUDA_VISIBLE_DEVICES = "0"

# --- Performance / Stabilität ---
$env:OLLAMA_NUM_THREADS = "$THREADS"

# Modell im VRAM halten (kein ständiges Laden/Entladen) :contentReference[oaicite:1]{index=1}
$env:OLLAMA_KEEP_ALIVE = "-1"

# Nur ein Modell gleichzeitig laden (damit VRAM nicht fragmentiert/geteilt wird) :contentReference[oaicite:2]{index=2}
$env:OLLAMA_MAX_LOADED_MODELS = "1"

# Längeres Timeout beim Laden, damit "discovery/load timeout" weniger nervt :contentReference[oaicite:3]{index=3}
$env:OLLAMA_LOAD_TIMEOUT = "15m"

# --- Modelfile automatisch erzeugen ---
$modelfile = @"
FROM $BASE_MODEL

# Größerer Kontext (mehr VRAM, kann aber Flow verbessern) :contentReference[oaicite:4]{index=4}
PARAMETER num_ctx $NUM_CTX

# Weniger Sampling-Overhead (minimal schneller / stabiler)
PARAMETER temperature 0.6
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER repeat_last_n 64

# Optional: Standard-Limit, damit Requests nicht ewig laufen
PARAMETER num_predict 512
"@

$modelfilePath = Join-Path $PSScriptRoot "Modelfile.$PERF_MODEL"
Set-Content -Path $modelfilePath -Value $modelfile -Encoding UTF8

# --- Perf-Modell erstellen/überschreiben ---
Write-Host "Creating/Updating model $PERF_MODEL ..." -ForegroundColor Cyan
& ollama create $PERF_MODEL -f $modelfilePath | Out-Host

# --- Ollama starten ---
Write-Host "Starting ollama serve ..." -ForegroundColor Green
$proc = Start-Process -FilePath "ollama" -ArgumentList "serve" -PassThru -WindowStyle Minimized
Start-Sleep -Seconds 2

# Priorität hoch setzen (best-effort)
try {
  (Get-Process -Id $proc.Id).PriorityClass = "High"
  Write-Host "Process priority set to HIGH." -ForegroundColor Green
} catch {
  Write-Host "Could not set priority (try PowerShell as Admin)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "READY. Use model: $PERF_MODEL" -ForegroundColor Green
Write-Host "Tip: for 'GPU stays busy' you still need parallel requests (2-8 at once)." -ForegroundColor Cyan
