# ============================================================
# Ollama Starter - liest Config aus config.json im selben Ordner
# ============================================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigFile = Join-Path $ScriptDir "config.json"

if (-not (Test-Path $ConfigFile)) {
    Write-Host "[ERROR] config.json nicht gefunden: $ConfigFile" -ForegroundColor Red
    exit 1
}

# Config laden
$cfg = Get-Content $ConfigFile -Raw | ConvertFrom-Json

# Werte aus Config lesen (mit Defaults)
$HOST_VAL   = if ($cfg.server.host)   { $cfg.server.host }   else { "127.0.0.1" }
$PORT_VAL   = if ($cfg.server.port)   { $cfg.server.port }   else { 11434 }
$GPU_DEVICE = if ($null -ne $cfg.gpu.cuda_device) { $cfg.gpu.cuda_device } else { 1 }
$CTX_LEN    = if ($cfg.model.context_length) { $cfg.model.context_length } else { 4096 }
$THREADS    = if ($cfg.model.threads)        { $cfg.model.threads }        else { 8 }
$KEEP_ALIVE = if ($cfg.model.keep_alive)     { $cfg.model.keep_alive }     else { "-1" }
$MAX_MODELS = if ($cfg.model.max_loaded_models) { $cfg.model.max_loaded_models } else { 1 }
$LOAD_TIMEOUT = if ($cfg.model.load_timeout) { $cfg.model.load_timeout }   else { "15m" }

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Ollama Config geladen:" -ForegroundColor Cyan
Write-Host "    Host:         ${HOST_VAL}:${PORT_VAL}" -ForegroundColor White
Write-Host "    GPU Device:   $GPU_DEVICE" -ForegroundColor White
Write-Host "    Threads:      $THREADS" -ForegroundColor White
Write-Host "    Context:      $CTX_LEN" -ForegroundColor White
Write-Host "    Keep Alive:   $KEEP_ALIVE" -ForegroundColor White
Write-Host "    Max Models:   $MAX_MODELS" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan

# Ollama beenden (falls es laeuft)
Write-Host "Stoppe laufende Ollama Instanz..." -ForegroundColor Yellow
taskkill /F /IM ollama.exe 2>$null

# Environment-Variablen setzen
$env:OLLAMA_HOST = "${HOST_VAL}:${PORT_VAL}"
$env:OLLAMA_CONTEXT_LENGTH = "$CTX_LEN"
$env:CUDA_VISIBLE_DEVICES = "$GPU_DEVICE"
$env:OLLAMA_NUM_THREADS = "$THREADS"
$env:OLLAMA_KEEP_ALIVE = "$KEEP_ALIVE"
$env:OLLAMA_MAX_LOADED_MODELS = "$MAX_MODELS"
$env:OLLAMA_LOAD_TIMEOUT = "$LOAD_TIMEOUT"

Write-Host "Starte Ollama auf ${HOST_VAL}:${PORT_VAL} (GPU $GPU_DEVICE, Threads=$THREADS, CTX=$CTX_LEN)" -ForegroundColor Green

# Ollama starten
ollama serve
