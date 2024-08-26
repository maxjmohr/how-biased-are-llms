import ollama

print("Downloading gemma2...")
ollama.pull("gemma2")
print("Downloading gemma2:27b...")
ollama.pull("gemma2:27b")
print("Downloading llama3.1...")
ollama.pull("llama3.1")
print("Downloading llama3.1:70b...")
ollama.pull("llama3.1:70b")
print("Downloading phi3.5...")
ollama.pull("phi3.5")
print("Downloading phi3:medium...")
ollama.pull("phi3:medium")
