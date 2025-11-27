# cuda_kernels.py

from numba import cuda, float64
import numpy as np
import math
from typing import Dict, Set, Tuple

# Variável global (será definida no wrapper) para o tamanho N do problema
N_VERTICES_GLOBAL = 0

@cuda.jit(device=True)
def get_dist(dist_matrix_gpu, i, j):
    """
    Função device auxiliar para obter a distância na matriz 2D plana (1-based index).
    """
    # dist_matrix_gpu é um array 1D de (N * N) elementos
    idx = (i - 1) * N_VERTICES_GLOBAL + (j - 1)
    if 0 <= idx < N_VERTICES_GLOBAL * N_VERTICES_GLOBAL:
        return dist_matrix_gpu[idx]
    return float('inf')

@cuda.jit
def evaluate_kernel(
    dist_matrix_gpu, 
    demand_weights_gpu, 
    solution_set_gpu, 
    radius, 
    beta, 
    result_array, 
    coverage_count_gpu, 
    p_size # Tamanho do conjunto de facilidades (p)
):
    """
    Kernel CUDA: Cada thread processa um nó de demanda 'i' e calcula
    sua contribuição para a FO e sua contagem de cobertura.
    """
    # i é o índice do nó de demanda (1-based index)
    i = cuda.grid(1) + 1 
    
    if i <= N_VERTICES_GLOBAL:
        count = 0.0
        
        # Itera sobre o conjunto de facilidades abertas (solução)
        for k in range(p_size):
            j = solution_set_gpu[k] # j é o índice da facilidade (1-based index)
            
            # Obtém a distância
            dist_ij = get_dist(dist_matrix_gpu, i, j)
            
            if dist_ij <= radius:
                count += 1.0
        
        # Salva a contagem de cobertura (para a Busca Local/Análise)
        coverage_count_gpu[i - 1] = count
        
        # Cálculo da Contribuição para a FO
        y_i = 1.0 if count >= 1.0 else 0.0
        y2_i = 1.0 if count >= 2.0 else 0.0
        
        weight_i = demand_weights_gpu[i - 1] # 0-based index
        
        contribution = weight_i * (y_i + beta * y2_i)
        
        # Salva o resultado
        result_array[i - 1] = contribution


def evaluate_solution_cuda(
    solution_set: Set[int],
    dist_dict: Dict[int, Dict[int, float]],
    demand_weights: Dict[int, float],
    radius: float,
    beta: float,
) -> Tuple[float, Dict[int, int]]:
    """
    Função wrapper que prepara os dados, executa o kernel CUDA e coleta resultados.
    Esta função substitui a avaliação sequencial.
    """
    global N_VERTICES_GLOBAL
    N_VERTICES_GLOBAL = len(demand_weights)
    N = N_VERTICES_GLOBAL
    
    # --- 1. Preparação de Dados (CPU/NumPy) ---
    
    # Matriz de distâncias (1D plana)
    dist_np = np.array(
        [dist_dict[i][j] for i in range(1, N + 1) for j in range(1, N + 1)], 
        dtype=np.float64
    )
    
    # Pesos de demanda (1D)
    weights_np = np.array(
        [demand_weights[i] for i in range(1, N + 1)], 
        dtype=np.float64
    )
    
    # Solução (Facilidades abertas, 1D)
    solution_list = list(solution_set)
    solution_np = np.array(solution_list, dtype=np.int32)
    p_size = len(solution_np)
    
    # Arrays de resultados na CPU (para cópia de volta)
    result_np = np.zeros(N, dtype=np.float64)
    coverage_count_np = np.zeros(N, dtype=np.float64)

    # --- 2. Transferência para a GPU (Device Memory) ---
    dist_gpu = cuda.to_device(dist_np)
    weights_gpu = cuda.to_device(weights_np)
    solution_gpu = cuda.to_device(solution_np)
    result_gpu = cuda.to_device(result_np)
    coverage_count_gpu = cuda.to_device(coverage_count_np)

    # --- 3. Execução do Kernel ---
    threads_per_block = 256
    blocks_per_grid = math.ceil(N / threads_per_block)
    
    evaluate_kernel[blocks_per_grid, threads_per_block](
        dist_gpu, weights_gpu, solution_gpu, radius, beta, 
        result_gpu, coverage_count_gpu, p_size
    )
    cuda.synchronize()

    # --- 4. Transferência de volta para a CPU ---
    result_gpu.copy_to_host(result_np)
    coverage_count_gpu.copy_to_host(coverage_count_np)

    # --- 5. Processamento dos Resultados ---
    total_objective = np.sum(result_np)
    
    # Converte o array de contagens de volta para o dict {i: count}
    coverage_count = {i: int(coverage_count_np[i-1]) for i in range(1, N + 1)}

    return total_objective, coverage_count