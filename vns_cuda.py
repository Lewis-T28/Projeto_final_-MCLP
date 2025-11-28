import random
import numpy as np
import os
import time
import sys
import math
from datetime import datetime

# =============================================================================
# 0. VERIFICAÇÃO ESTRITA DE GPU
# =============================================================================
try:
    from numba import cuda, float32, int32
    if not cuda.is_available():
        print("\n" + "!"*60)
        print("ERRO CRÍTICO: GPU NVIDIA com CUDA não detectada.")
        print("Este código está configurado para encerrar se não houver GPU.")
        print("!"*60 + "\n")
        sys.exit(1)
    has_gpu = True
except ImportError:
    print("\n" + "!"*60)
    print("ERRO CRÍTICO: Biblioteca Numba não encontrada.")
    print("Instale usando: pip install numba (ou conda install numba cudatoolkit)")
    print("!"*60 + "\n")
    sys.exit(1)

# =============================================================================
# 1. CARREGAMENTO DE DADOS E MOCKS
# =============================================================================

def load_pmed_instance_mock(filepath):
    # Mock para testes rápidos sem arquivo
    regioes = ['Centro', 'Norte', 'Leste', 'Sudeste', 'Sul', 'Oeste', 'Rural', 'SFX']
    populacao_dict = {
        'Centro': 72401, 'Norte': 61940, 'Leste': 181463, 'Sudeste': 62541,
        'Sul': 237572, 'Oeste': 64482, 'Rural': 15212, 'SFX': 1443
    }
    dist = np.array([
        [0, 6, 8, 7, 10, 9, 25, 35], [6, 0, 12, 10, 14, 8, 20, 40], 
        [8, 12, 0, 5, 15, 18, 30, 45], [7, 10, 5, 0, 12, 16, 28, 42], 
        [10, 14, 15, 12, 0, 20, 22, 38], [9, 8, 18, 16, 20, 0, 15, 50], 
        [25, 20, 30, 28, 22, 15, 0, 60], [35, 40, 45, 42, 38, 50, 60, 0]
    ], dtype=np.float32)
    pop_arr = np.array([populacao_dict[r] for r in regioes], dtype=np.float32)
    return 4, regioes, pop_arr, dist

try:
    from src.pmed_loader_to_vns_alpha_npmp import load_pmed_instance
    USING_MOCK = False
except ImportError:
    USING_MOCK = True

# =============================================================================
# 2. KERNELS CUDA (GPU)
# =============================================================================

@cuda.jit
def kernel_eval_moves(moves_out, moves_in, current_sol, dist_matrix, pop_arr, alpha, costs_result):
    tid = cuda.grid(1)
    num_moves = moves_out.shape[0]
    if tid >= num_moves:
        return

    out_node = moves_out[tid]
    in_node = moves_in[tid]
    
    num_clientes = dist_matrix.shape[0]
    p_val = current_sol.shape[0]
    total_cost = 0.0
    
    # Loop sobre clientes
    for j in range(num_clientes):
        current_pop = pop_arr[j]
        sum_dist_alpha = 0.0
        
        # Array local simulado (máscara de usados)
        # Suporte até P=500 facilidades
        used_mask = cuda.local.array(500, dtype=int32) 
        for k_reset in range(p_val):
            used_mask[k_reset] = 0
            
        # Encontra os 'alpha' menores valores iterativamente
        for a in range(alpha):
            min_dist = 1e20 # Infinito
            best_k = -1
            
            for k in range(p_val):
                if used_mask[k] == 1:
                    continue
                
                fac_idx = current_sol[k]
                # Troca virtual
                if fac_idx == out_node:
                    real_fac = in_node
                else:
                    real_fac = fac_idx
                
                d = dist_matrix[real_fac, j]
                
                if d < min_dist:
                    min_dist = d
                    best_k = k
            
            if best_k != -1:
                sum_dist_alpha += min_dist
                used_mask[best_k] = 1
        
        total_cost += sum_dist_alpha * current_pop

    costs_result[tid] = total_cost

# =============================================================================
# 3. FUNÇÕES (CPU e Lógica)
# =============================================================================

def evaluate_solution_cpu(solution_indices, dist_matrix, populacao_arr, alpha):
    """Versão Otimizada (Vetorizada) para CPU - Apenas para validação inicial."""
    if not isinstance(solution_indices, np.ndarray):
        solution_indices = np.array(solution_indices, dtype=np.int32)
    
    # Seleciona apenas as linhas das facilidades abertas
    dists_submatrix = dist_matrix[solution_indices, :]
    # Ordena colunas (axis=0) para pegar os menores custos para cada cliente
    dists_sorted = np.sort(dists_submatrix, axis=0)
    # Soma os alpha menores
    top_alpha_dists = dists_sorted[:alpha, :]
    custos_por_cliente = np.sum(top_alpha_dists, axis=0)
    # Custo total ponderado
    total_cost = np.dot(custos_por_cliente, populacao_arr)
    
    return total_cost

def shaking(solution, num_nodes, k):
    """Perturbação k-swap."""
    s_prime = list(solution)
    abertas = list(s_prime)
    fechadas = [i for i in range(num_nodes) if i not in abertas]
    
    for _ in range(k):
        if not abertas or not fechadas: break
        i_fechar = random.choice(abertas)
        abertas.remove(i_fechar)
        s_prime.remove(i_fechar)
        
        i_abrir = random.choice(fechadas)
        fechadas.remove(i_abrir)
        s_prime.append(i_abrir)
    
    return np.array(s_prime, dtype=np.int32)

def local_search_gpu(solution, d_dist_matrix, d_pop_arr, alpha, num_nodes):
    """Busca Local na GPU."""
    set_sol = set(solution)
    abertas = list(solution)
    fechadas = [i for i in range(num_nodes) if i not in set_sol]
    
    moves_out = []
    moves_in = []
    
    for out_node in abertas:
        for in_node in fechadas:
            moves_out.append(out_node)
            moves_in.append(in_node)
            
    num_moves = len(moves_out)
    if num_moves == 0:
        return solution, 0.0
        
    # Prepara arrays
    np_moves_out = np.array(moves_out, dtype=np.int32)
    np_moves_in = np.array(moves_in, dtype=np.int32)
    np_solution = np.array(solution, dtype=np.int32)
    
    # Aloca e transfere para GPU
    d_costs = cuda.device_array(num_moves, dtype=np.float32)
    d_moves_out = cuda.to_device(np_moves_out)
    d_moves_in = cuda.to_device(np_moves_in)
    d_current_sol = cuda.to_device(np_solution)
    
    # Lança Kernel
    threads_per_block = 128
    blocks_per_grid = (num_moves + (threads_per_block - 1)) // threads_per_block
    
    kernel_eval_moves[blocks_per_grid, threads_per_block](
        d_moves_out, d_moves_in, d_current_sol, 
        d_dist_matrix, d_pop_arr, alpha, d_costs
    )
    
    # Recupera
    costs_cpu = d_costs.copy_to_host()
    
    min_idx = np.argmin(costs_cpu)
    min_cost = costs_cpu[min_idx]
    
    best_out = moves_out[min_idx]
    best_in = moves_in[min_idx]
    
    best_neighbor = [x for x in solution if x != best_out] + [best_in]
    return np.array(best_neighbor, dtype=np.int32), min_cost

def vns_solve(p_val, alpha_val, dist_matrix_cpu, populacao_arr_cpu, regiao_names, max_iter=50, k_max=4):
    num_nodes = len(populacao_arr_cpu)
    
    print(">> Inicializando memória GPU...")
    # Transferência Única: Envia as matrizes pesadas para a VRAM apenas uma vez
    d_dist_matrix = cuda.to_device(dist_matrix_cpu)
    d_pop_arr = cuda.to_device(populacao_arr_cpu)
    
    # Solução Inicial
    s = np.array(random.sample(range(num_nodes), p_val), dtype=np.int32)
    cost = evaluate_solution_cpu(s, dist_matrix_cpu, populacao_arr_cpu, alpha_val)
    s_best = np.copy(s)
    cost_best = cost
    
    print(f"Custo Inicial: {cost_best:,.2f}")
    
    # Relatório
    data_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("relatorios", exist_ok=True)
    filename = f"relatorios/execucao_gpu_{data_hora}.txt"
    
    print(f">> Iniciando VNS (Modo GPU Estrito)")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Modo: GPU ONLY | P={p_val} | Alpha={alpha_val}\n")
        
        start_global = time.time()
        
        for iter_count in range(max_iter):
            iter_start = time.time()
            k = 1
            
            while k <= k_max:
                # 1. Shaking
                s_prime = shaking(s, num_nodes, k)
                cost_prime = evaluate_solution_cpu(s_prime, dist_matrix_cpu, populacao_arr_cpu, alpha_val)
                
                # 2. Busca Local (GPU)
                curr_s = s_prime
                curr_cost = cost_prime
                
                while True:
                    # Chamada direta para GPU
                    best_n, best_c = local_search_gpu(curr_s, d_dist_matrix, d_pop_arr, alpha_val, num_nodes)
                    
                    if best_c < curr_cost:
                        curr_cost = best_c
                        curr_s = best_n
                    else:
                        break # Ótimo local encontrado
                
                # 3. Move
                if curr_cost < cost:
                    s = curr_s
                    cost = curr_cost
                    k = 1
                    if cost < cost_best:
                        s_best = np.copy(s)
                        cost_best = cost
                        print(f"Iter {iter_count+1}: Novo Melhor: {cost_best:,.2f}")
                else:
                    k += 1
            
            iter_time = time.time() - iter_start
            f.write(f"Iter {iter_count+1}: {iter_time:.4f}s | Cost={cost:.2f}\n")
            
            if (iter_count+1) % 10 == 0:
                print(f"Concluído {iter_count+1}/{max_iter}...")

    total_time = time.time() - start_global
    print(f"Tempo Total: {total_time:.2f}s")
    return s_best, cost_best

# =============================================================================
# 4. MAIN
# =============================================================================

if __name__ == '__main__':
    INSTANCIA_PATH = "instancias/pmed1.txt"
    ALPHA = 1
    P_DEFAULT = 5
    
    # Load
    if USING_MOCK:
        print("--- Mock Data (Test Mode) ---")
        P, REGIOES, POPULACAO_ARR, DISTANCIA = load_pmed_instance_mock("")
        P = 4
    else:
        print(f"--- Lendo {INSTANCIA_PATH} ---")
        try:
            P_loaded, REGIOES, POPULACAO_DICT, DISTANCIA = load_pmed_instance(INSTANCIA_PATH)
            P = P_loaded if P_loaded else P_DEFAULT
            POPULACAO_ARR = np.array(list(POPULACAO_DICT.values()), dtype=np.float32)
            DISTANCIA = DISTANCIA.astype(np.float32)
        except Exception as e:
            print(f"Erro ao ler arquivo ({e}). Usando Mock.")
            P, REGIOES, POPULACAO_ARR, DISTANCIA = load_pmed_instance_mock("")
            
    print(f"Config: P={P}, Alpha={ALPHA}, Nodes={len(REGIOES)}")
    
    # Executa VNS
    best_sol, best_z = vns_solve(P, ALPHA, DISTANCIA, POPULACAO_ARR, REGIOES, max_iter=50, k_max=P)
    
    print("="*50)
    print(f"MELHOR CUSTO: {best_z:,.2f}")
    names = [REGIOES[i] for i in best_sol]
    print(f"FACILIDADES: {names}")