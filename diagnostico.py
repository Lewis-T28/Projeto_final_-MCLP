import os
from numba import cuda

print(f"CUDA disponível? {cuda.is_available()}")

try:
    cuda.detect()
except Exception as e:
    print(f"Erro ao detectar: {e}")

# Tenta listar as GPUs
if cuda.is_available():
    for gpu in cuda.gpus:
        print(f"GPU Encontrada: {gpu.name}")
else:
    print("\n--- PISTAS DO PROBLEMA ---")
    print("1. Verifique se a variável de ambiente CUDA_HOME está definida.")
    # Procura caminhos comuns
    caminhos_comuns = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\Windows\System32\nvcuda.dll"
    ]
    print("2. Verifique se o arquivo nvcuda.dll existe em System32.")