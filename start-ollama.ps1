# Ollama beenden (falls es läuft)
taskkill /F /IM ollama.exe 2>$null

# CUDA Device setzen
$env:CUDA_VISIBLE_DEVICES = "1"

# Ollama starten
ollama serve
Speichern als z. B.