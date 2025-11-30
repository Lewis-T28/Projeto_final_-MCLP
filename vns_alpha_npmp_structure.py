import random
import numpy as np
import os
from datetime import datetime
import time

# Tenta importar o loader, senão usa mock (para garantir que o código rode)
try:
    from src.pmed_loader_to_vns_alpha_npmp import load_pmed_instance
    USING_MOCK = False
except ImportError:
    USING_MOCK = True

# --- 1. DADOS DE ENTRADA (MOCK / DEFAULT) ---
REGIOES = ['Centro', 'Norte', 'Leste', 'Sudeste', 'Sul', 'Oeste', 'Rural', 'SFX']
POPULACAO = {
    'Centro': 72401, 'Norte': 61940, 'Leste': 181463, 'Sudeste': 62541,
    'Sul': 237572, 'Oeste': 64482, 'Rural': 15212, 'SFX': 1443
}
# Matriz de Distâncias (d_ij em km)
DISTANCIA = np.array([
    [0, 6, 8, 7, 10, 9, 25, 35], [6, 0, 12, 10, 14, 8, 20, 40], 
    [8, 12, 0, 5, 15, 18, 30, 45], [7, 10, 5, 0, 12, 16, 28, 42], 
    [10, 14, 15, 12, 0, 20, 22, 38], [9, 8, 18, 16, 20, 0, 15, 50], 
    [25, 20, 30, 28, 22, 15, 0, 60], [35, 40, 45, 42, 38, 50, 60, 0]
], dtype=np.float32)

P = 4
ALPHA = 1 
INSTANCIA = "instancias/pmed40.txt"

# --- 2. FUNÇÃO OTIMIZADA (O PULO DO GATO) ---

def evaluate_solution_optimized(solution_indices, dist_matrix, populacao_arr, alpha):
    """
    Substitui loops explícitos e 'append' por operações vetoriais do NumPy.
    Acelera o cálculo em 50x-100x dependendo do tamanho da instância.
    """
    # 1. Indexação Avançada (Fancy Indexing)
    # Pega TODAS as distâncias das facilidades escolhidas para TODOS os clientes de uma vez.
    # Shape resultante: (P, Num_Clientes)
    dists_submatrix = dist_matrix[solution_indices, :]
    
    # --- OTIMIZAÇÃO CRÍTICA (ALPHA = 1) ---
    if alpha == 1:
        # Se Alpha=1, queremos apenas a menor distância (facilidade mais próxima).
        # np.min é O(P), enquanto np.sort é O(P log P).
        min_dists = np.min(dists_submatrix, axis=0)
        return np.dot(min_dists, populacao_arr)

    # --- OTIMIZAÇÃO CRÍTICA (ALPHA > 1) ---
    # Se Alpha > 1, não precisamos ordenar TUDO. Usamos partition para achar os menores.
    if alpha < dists_submatrix.shape[0]:
        # np.partition move os 'alpha' menores elementos para o topo (sem ordenar entre si)
        partitioned = np.partition(dists_submatrix, alpha-1, axis=0)
        top_alpha = partitioned[:alpha, :]
    else:
        # Se P <= Alpha, pega tudo
        top_alpha = dists_submatrix

    # 4. Soma das Distâncias (Colapso Vertical)
    # Soma as distâncias selecionadas para cada cliente.
    # Shape resultante: (Num_Clientes,)
    sum_dists_per_client = np.sum(top_alpha, axis=0)
    
    # 5. Custo Ponderado (Produto Escalar)
    # Multiplica a soma das distâncias pela população de cada cliente e soma tudo.
    # Isso substitui o loop 'total_cost += ...'
    total_cost = np.dot(sum_dists_per_client, populacao_arr)
            
    return total_cost

def generate_initial_solution(p_val, num_regioes):
    return random.sample(range(num_regioes), p_val)

# --- 3. ESTRUTURA DO VNS ---

def shaking(solution, k, num_regioes):
    s_prime = list(solution)
    abertas = set(s_prime)
    fechadas = [i for i in range(num_regioes) if i not in abertas]
    
    for _ in range(k):
        if not abertas or not fechadas: break
        
        # Escolha aleatória mais eficiente
        i_fechar = random.choice(list(abertas))
        abertas.remove(i_fechar)
        s_prime.remove(i_fechar)
        
        i_abrir = random.choice(fechadas)
        fechadas.remove(i_abrir)
        # Nota: fechadas não precisa ser atualizada se vamos reconstruir o set na proxima iteração
        s_prime.append(i_abrir)
        
    return s_prime

def local_search(solution, dist_matrix, populacao_arr, alpha, num_regioes):
    """
    Busca local usando a função de avaliação otimizada.
    """
    s_best = list(solution)
    # Importante: converter para array numpy para indexação funcionar
    s_best_arr = np.array(s_best, dtype=int)
    cost_best = evaluate_solution_optimized(s_best_arr, dist_matrix, populacao_arr, alpha)
    
    while True:
        melhoria = False
        abertas = list(s_best)
        set_abertas = set(s_best)
        fechadas = [i for i in range(num_regioes) if i not in set_abertas]
        
        # Estrutura First Improvement (pode ser trocada por Best Improvement)
        for i_fechar in abertas:
            for i_abrir in fechadas:
                # Cria vizinho
                neighbor = [x for x in s_best if x != i_fechar] + [i_abrir]
                neighbor_arr = np.array(neighbor, dtype=int)
                
                # Avalia (Rápido agora!)
                neighbor_cost = evaluate_solution_optimized(neighbor_arr, dist_matrix, populacao_arr, alpha)
                
                if neighbor_cost < cost_best:
                    cost_best = neighbor_cost
                    s_best = neighbor
                    melhoria = True
                    break # First improvement (sai e reinicia vizinhança)
            if melhoria: break
            
        if not melhoria:
            break
            
    return s_best, cost_best

def vns_solve(p_val, alpha_val, dist_matrix, populacao_arr, regioes, max_iter=50, k_max=4):
    num_nodes = len(regioes)
    
    # Solução Inicial
    s = generate_initial_solution(p_val, num_nodes)
    s_arr = np.array(s, dtype=int)
    cost = evaluate_solution_optimized(s_arr, dist_matrix, populacao_arr, alpha_val)
    
    s_best = list(s)
    cost_best = cost
    
    print(f"Custo Inicial: {cost_best:,.2f}")
    
    data_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("relatorios", exist_ok=True)
    filename = f"relatorios/cpu_opt_{data_hora}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"VNS CPU OPTIMIZED | P={p_val} | Alpha={alpha_val}\n")
        start_time = time.time()
        
        for iter_count in range(1, max_iter):
            iter_start = time.time()
            k = 1
            
            while k <= k_max:
                # 1. Shaking
                s_prime = shaking(s, k, num_nodes)
                
                # 2. Busca Local
                s_double, cost_double = local_search(s_prime, dist_matrix, populacao_arr, alpha_val, num_nodes)
                
                # 3. Move
                if cost_double < cost:
                    s = s_double
                    cost = cost_double
                    k = 1
                    if cost < cost_best:
                        s_best = list(s)
                        cost_best = cost
                        print(f"Iter {iter_count}: Novo Melhor: {cost_best:,.2f}")
                else:
                    k += 1
            
            duration = time.time() - iter_start
            f.write(f"Iter {iter_count}: {duration:.4f}s | Cost={cost:.2f} | S={s_best}\n")
            
            if iter_count % 10 == 0:
                print(f"Iter {iter_count}/{max_iter}...")
                
    total_time = time.time() - start_time
    print(f"Tempo Total: {total_time:.2f}s")
    return s_best, cost_best

# --- 4. EXECUÇÃO ---

if __name__ == '__main__':
    # Carregar Dados Reais se possível
    if not USING_MOCK:
        print(f"--- Lendo {INSTANCIA} ---")
        try:
            P_loaded, REGIOES, POP_DICT, DISTANCIA = load_pmed_instance(INSTANCIA)
            if P_loaded: P = P_loaded
            # CONVERSÃO CRÍTICA: Dict -> Array (alinhado com a ordem de REGIOES)
            # A função otimizada precisa de um array de população, não dicionário.
            POPULACAO_ARR = np.array([POP_DICT[r] for r in REGIOES], dtype=float)
        except Exception as e:
            print(f"Erro no load: {e}. Usando dados padrão.")
            POPULACAO_ARR = np.array([POPULACAO[r] for r in REGIOES], dtype=float)
    else:
        print("--- Usando Mock Data ---")
        POPULACAO_ARR = np.array([POPULACAO[r] for r in REGIOES], dtype=float)

    print(f"Config: P={P}, Alpha={ALPHA}, Nodes={len(REGIOES)}")
    
    best_idx, best_val = vns_solve(P, ALPHA, DISTANCIA, POPULACAO_ARR, REGIOES, max_iter=50, k_max=P)
    
    print("\n" + "="*50)
    print(f"MELHOR CUSTO: {best_val:,.2f}")
    names = [REGIOES[i] for i in best_idx]
    print(f"FACILIDADES: {names}")