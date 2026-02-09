# Ollama beenden (falls es läuft)
taskkill /F /IM ollama.exe 2>$null

# CUDA Device setzen
$env:CUDA_VISIBLE_DEVICES = "1"
$env:OLLAMA_CONTEXT_LENGTH = "4096" 
$env:CUDA_VISIBLE_DEVICES = "1" 
$env:OLLAMA_NUM_THREADS = "$THREADS" 
$env:OLLAMA_KEEP_ALIVE = "-1" 
$env:OLLAMA_MAX_LOADED_MODELS = "1" 
$env:OLLAMA_LOAD_TIMEOUT = "15m"
$NUM_CTX    = 4096                      # 4096 oder 8192 testen
$THREADS    = 8   
# Ollama starten
ollama serve
