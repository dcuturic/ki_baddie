# Ollama beenden (falls es läuft)
taskkill /F /IM ollama.exe 2>$null

# Konfiguration
$THREADS    = 8
$NUM_CTX    = 4096                      # 4096 oder 8192 testen
$OLLAMA_HOST = "127.0.0.1:11434"       # Host:Port auf dem Ollama lauscht

# Environment-Variablen setzen
$env:OLLAMA_HOST = $OLLAMA_HOST
$env:OLLAMA_CONTEXT_LENGTH = "$NUM_CTX" 
$env:CUDA_VISIBLE_DEVICES = "1" 
$env:OLLAMA_NUM_THREADS = "$THREADS" 
$env:OLLAMA_KEEP_ALIVE = "-1" 
$env:OLLAMA_MAX_LOADED_MODELS = "1" 
$env:OLLAMA_LOAD_TIMEOUT = "15m"

Write-Host "Ollama startet auf $OLLAMA_HOST (GPU 1, Threads=$THREADS, CTX=$NUM_CTX)"

# Ollama starten
ollama serve
