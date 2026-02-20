# Ollama beenden (falls es läuft)
taskkill /F /IM ollama.exe 2>$null

# Konfiguration
$THREADS    = 8
$NUM_CTX    = 4096                      # 4096 oder 8192 testen
$OLLAMA_HOST = "127.0.0.1:11434"       # Host:Port auf dem Ollama lauscht

# Environment-Variablen setzen
$env:OLLAMA_HOST = $OLLAMA_HOST
$env:OLLAMA_CONTEXT_LENGTH = "$NUM_CTX" 
$env:CUDA_VISIBLE_DEVICES = "0"             # RTX 5060 Ti ist GPU 0!
$env:OLLAMA_NUM_THREADS = "$THREADS" 
$env:OLLAMA_KEEP_ALIVE = "-1" 
$env:OLLAMA_MAX_LOADED_MODELS = "1" 
$env:OLLAMA_LOAD_TIMEOUT = "15m"

# ===== GPU Performance Optimierungen =====
$env:OLLAMA_FLASH_ATTENTION = "1"         # Flash Attention = deutlich schneller + weniger VRAM
$env:OLLAMA_KV_CACHE_TYPE = "q4_0"        # Aggressiv komprimierter KV-Cache
$env:OLLAMA_NUM_GPU = "999"               # ALLE Layers auf GPU
$env:OLLAMA_GPU_OVERHEAD = "0"            # Kein VRAM-Reserve
$env:OLLAMA_NUM_PARALLEL = "1"            # Volle GPU-Power pro Request
$env:CUDA_LAUNCH_BLOCKING = "0"           # Async CUDA-Calls
$env:OLLAMA_LLM_LIBRARY = "cuda_v12"      # Explizit CUDA 12 (RTX 50xx)

Write-Host "Ollama startet auf $OLLAMA_HOST (GPU 0, Threads=$THREADS, CTX=$NUM_CTX)"

# Ollama starten
ollama serve
