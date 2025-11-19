# vns_core.py

import random
from typing import Dict, Iterable, Set, Tuple, List, Optional
# Importa a função de avaliação paralela
from .cuda_kernels import evaluate_solution_cuda

# Renomeia para clareza dentro deste módulo
evaluate_solution = evaluate_solution_cuda


# =====================================================================
#   Funções de Vizinhança (Neighborhoods)
# =====================================================================

def neighborhood_k_exchange_random(solution_set: Set[int], candidates: Iterable[int], k: int) -> Set[int]:
    """
    Aplica uma operação aleatória de k-Exchange (Shaking).
    Remove k abertos e adiciona k fechados.
    """
    p = len(solution_set)
    open_facilities = list(solution_set)
    closed_facilities = [j for j in candidates if j not in solution_set]

    if len(open_facilities) < k or len(closed_facilities) < k:
        return solution_set.copy() 

    remove_j_list = random.sample(open_facilities, k)
    add_j_list = random.sample(closed_facilities, k)

    new_solution = solution_set.copy()
    for j in remove_j_list:
        new_solution.remove(j)
    for j in add_j_list:
        new_solution.add(j)
    
    return new_solution

# =====================================================================
#   Busca Local (Local Search - 1-Exchange Best Improvement)
# =====================================================================

def local_search_1_exchange(
    solution_set: Set[int],
    dist_dict: Dict[int, Dict[int, float]],
    demand_weights: Dict[int, float],
    radius: float,
    beta: float,
    candidates: Iterable[int],
) -> Tuple[Set[int], float]:
    """
    Aplica Busca Local (Best Improvement) usando a vizinhança 1-Exchange.
    Usa evaluate_solution_cuda para paralelizar a avaliação de cada vizinho.
    """
    best_solution = solution_set.copy()
    best_objective, _ = evaluate_solution(
        best_solution, dist_dict, demand_weights, radius, beta
    )
    
    improved = True
    while improved:
        improved = False
        
        open_facilities = list(best_solution)
        closed_facilities = [j for j in candidates if j not in best_solution]
        
        local_best_objective = best_objective
        local_best_solution = best_solution.copy()
        
        # Itera por todos os p * (N-p) movimentos 1-Exchange
        for remove_j in open_facilities:
            for add_j in closed_facilities:
                # Gera o novo conjunto de facilidades
                new_solution = (best_solution - {remove_j}) | {add_j}
                
                # Avalia o novo conjunto usando o kernel CUDA
                new_objective, _ = evaluate_solution(
                    new_solution, dist_dict, demand_weights, radius, beta
                )
                
                if new_objective > local_best_objective:
                    local_best_objective = new_objective
                    local_best_solution = new_solution
                    improved = True

        # Aplica o melhor movimento da iteração
        if improved:
            best_objective = local_best_objective
            best_solution = local_best_solution
        
    return best_solution, best_objective

# =====================================================================
#   VNS Principal
# =====================================================================

def solve_mclp_backup_vns(
    dist_dict: Dict[int, Dict[int, float]],
    demand_weights: Dict[int, float],
    p: int,
    radius: float,
    beta: float = 0.5,
    max_iter: int = 100,
    k_max: int = 3,
) -> Tuple[Set[int], float]:
    """
    Loop principal do Variable Neighborhood Search (VNS).
    """
    candidate_sites = sorted(dist_dict.keys())
    
    # 1. Solução Inicial: Escolha aleatória de p facilidades
    initial_solution_list = random.sample(candidate_sites, p)
    S_current: Set[int] = set(initial_solution_list)
    
    # Avaliação inicial (usando CUDA)
    FO_current, _ = evaluate_solution(S_current, dist_dict, demand_weights, radius, beta)
    
    S_best = S_current.copy()
    FO_best = FO_current
    
    print(f"\n--- VNS INICIADO (p={p}, R={radius}, beta={beta}) ---")
    print(f"VNS Inicial: FO={FO_current:.4f}, facilidades={S_current}")

    iteration = 0
    while iteration < max_iter:
        
        k = 1
        while k <= k_max:
            # a. Shaking (k-Exchange aleatório)
            S_prime = neighborhood_k_exchange_random(S_current, candidate_sites, k)
            
            # b. Busca Local (Refinamento usando 1-Exchange e avaliação CUDA)
            S_double_prime, FO_new = local_search_1_exchange(
                S_prime, dist_dict, demand_weights, radius, beta, candidate_sites
            )

            # c. Move or Not
            if FO_new > FO_current:
                # Melhora - Move, Atualiza Melhor Global, Reinicia k=1
                S_current = S_double_prime
                FO_current = FO_new
                
                if FO_new > FO_best:
                    FO_best = FO_new
                    S_best = S_double_prime.copy()
                
                print(f"Iter {iteration:03d}, FO={FO_current:.4f} (Melhora), k={k} -> Reinicia k=1")
                k = 1 
            else:
                # Não melhora - Incrementa k
                k += 1

        iteration += 1
        
    return S_best, FO_best