# vns_pmedian_redundancy.py

import random
import time
import math
import argparse
from typing import Dict, Iterable, Set, Tuple, List, Optional, Any

# =====================================================================
#   IMPORTAÇÕES DAS FUNÇÕES DE UTILIDADE (src/)
# =====================================================================

from src.reader_pmed import read_pmed_file, build_cost_matrix 
from src.floyd_marshall_algo import floyd_marshall 
from src.build_dist_matrix import matrix_to_dict
# Usamos a função de plotagem original do p-median (sem backup) ou uma adaptada
# Vamos assumir que você tem uma plotagem que só precisa do assignment final.
from src.print_nodes import plot_solution 


# =====================================================================
#   1. FUNÇÃO DE AVALIAÇÃO PARA p-MEDIAN COM REDUNDÂNCIA (K)
# =====================================================================

def evaluate_pmedian_redundancy(
    solution_set: Set[int],
    dist_dict: Dict[int, Dict[int, float]],
    demand_weights: Dict[int, float],
    K_redundancy: int = 2,  # Parâmetro K para redundância
) -> Tuple[float, Dict[int, int]]:
    """
    Calcula o valor da Função Objetivo (FO) para o problema p-Median com redundância K.

    FO = min sum_i w_i * (soma das K menores distâncias d_ij)
    
    Retorna o valor da FO e um dicionário de atribuição (apenas para a 1ª facilidade mais próxima).
    """
    total_objective = 0.0
    demand_nodes = demand_weights.keys()
    
    # Usado para armazenar a atribuição da demanda i para a 1ª facilidade mais próxima (para plotagem)
    assignment: Dict[int, int] = {}
    
    # ----------------------------------------------------
    # ATENÇÃO: Se K for maior que p, este cálculo falhará ou
    # usará menos de K facilidades para a soma.
    # Assumimos que K <= p.
    # ----------------------------------------------------

    for i in demand_nodes:
        # 1. Coleta todas as distâncias da demanda i para as facilidades abertas j
        distances_to_centers = []
        for j in solution_set:
            distances_to_centers.append((dist_dict[i].get(j, float('inf')), j))
        
        # 2. Ordena as distâncias (do menor para o maior)
        distances_to_centers.sort(key=lambda x: x[0])
        
        # 3. Calcula a soma das K menores distâncias
        
        # O número real de facilidades abertas que podemos usar é min(K, len(distances_to_centers))
        num_to_sum = min(K_redundancy, len(distances_to_centers))

        redundancy_cost = 0.0
        
        # Soma as distâncias até a k-ésima facilidade mais próxima
        for k in range(num_to_sum):
            distance, center_j = distances_to_centers[k]
            redundancy_cost += distance
            
            # Armazena apenas a 1ª facilidade (k=0) para plotagem
            if k == 0:
                assignment[i] = center_j

        # Adiciona a contribuição ponderada à FO total (Minimização)
        total_objective += demand_weights[i] * redundancy_cost

    return total_objective, assignment


# =====================================================================
#   2. Funções de Vizinhança (SHAKING)
# =====================================================================

# Reutilizamos a função neighborhood_k_exchange_random do MCLP-Backup (sem alteração)
def neighborhood_k_exchange_random(solution_set: Set[int], candidates: Iterable[int], k: int) -> Set[int]:
    """
    Aplica uma operação aleatória de k-Exchange (Shaking).
    Remove k facilidades abertas e adiciona k fechadas. (Mesma lógica do MCLP-Backup)
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
    
    assert len(new_solution) == p
    return new_solution


# =====================================================================
#   3. Busca Local (1-Exchange Best Improvement)
# =====================================================================

# Reutilizamos a função local_search_1_exchange, substituindo a chamada de avaliação.
def local_search_1_exchange_redundancy(
    solution_set: Set[int],
    dist_dict: Dict[int, Dict[int, float]],
    demand_weights: Dict[int, float],
    K_redundancy: int,
    candidates: Iterable[int],
) -> Tuple[Set[int], float]:
    """
    Aplica Busca Local (Best Improvement) usando a vizinhança 1-Exchange,
    usando a FO de p-median com redundância.
    """
    # A lógica de Best Improvement é a mesma, mas FO é minimizada (Minimization)
    
    best_solution = solution_set.copy()
    # Usa a nova função de avaliação
    best_objective, _ = evaluate_pmedian_redundancy(
        best_solution, dist_dict, demand_weights, K_redundancy
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
                new_solution = (best_solution - {remove_j}) | {add_j}
                
                # Avalia o novo conjunto
                new_objective, _ = evaluate_pmedian_redundancy(
                    new_solution, dist_dict, demand_weights, K_redundancy
                )
                
                # Critério de MELHORIA para MINIMIZAÇÃO
                if new_objective < local_best_objective:
                    local_best_objective = new_objective
                    local_best_solution = new_solution
                    improved = True

        # Aplica o melhor movimento da iteração
        if improved:
            best_objective = local_best_objective
            best_solution = local_best_solution
        
    return best_solution, best_objective


# =====================================================================
#   4. VNS Principal para Redundância
# =====================================================================

def solve_pmedian_redundancy_vns(
    dist_dict: Dict[int, Dict[int, float]],
    demand_weights: Dict[int, float],
    p: int,
    K_redundancy: int = 2,
    max_iter: int = 100,
    k_max: int = 3, 
):
    """
    Loop principal do VNS para p-Median com Redundância K.
    """
    candidate_sites = sorted(dist_dict.keys())
    
    # 1. Solução Inicial
    initial_solution_list = random.sample(candidate_sites, p)
    S_current: Set[int] = set(initial_solution_list)
    
    FO_current, _ = evaluate_pmedian_redundancy(S_current, dist_dict, demand_weights, K_redundancy)
    
    S_best = S_current.copy()
    FO_best = FO_current
    
    print(f"\n--- VNS p-MEDIAN REDUNDÂNCIA (p={p}, K={K_redundancy}) ---")
    print(f"VNS Inicial: FO={FO_current:.4f}, facilidades={S_current}")

    iteration = 0
    while iteration < max_iter:
        
        k = 1
        while k <= k_max:
            # a. Shaking (k-Exchange)
            S_prime = neighborhood_k_exchange_random(S_current, candidate_sites, k)
            
            # b. Busca Local (1-Exchange)
            S_double_prime, FO_new = local_search_1_exchange_redundancy(
                S_prime, dist_dict, demand_weights, K_redundancy, candidate_sites
            )

            # c. Move or Not (Critério de MINIMIZAÇÃO)
            if FO_new < FO_current:
                # Melhora - Move, Atualiza Melhor Global, Reinicia k=1
                S_current = S_double_prime
                FO_current = FO_new
                
                if FO_new < FO_best:
                    FO_best = FO_new
                    S_best = S_double_prime.copy()
                
                print(f"Iter {iteration:03d}, FO={FO_current:.4f} (Melhora), k={k} -> Reinicia k=1")
                k = 1 
            else:
                # Não melhora - Incrementa k
                k += 1

        iteration += 1
        
    return S_best, FO_best

# =====================================================================
#   5. main_redundancy()
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Resolve p-median com redundância K usando a heurística VNS."
    )
    parser.add_argument("arquivo", help="Arquivo pmed* (formato OR-Library).")
    parser.add_argument(
        "--k-redundancy",
        "-K",
        type=int,
        default=2,
        help="Número de facilidades mais próximas a serem somadas na FO. Default: 2.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        help='Número máximo de iterações do loop principal do VNS. Default: 50.',
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=3,
        help='Número máximo de vizinhanças (k). Default: 3 (1-Exc, 2-Exc, 3-Exc).',
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Se usado, gera imagem da solução (solution_pmedian_red.png).",
    )

    args = parser.parse_args()

    # 1-5) Leitura, Floyd-Warshall, Matrizes
    n_vertices, n_edges, p, edges = read_pmed_file(args.arquivo)
    print(f"Instância lida: n={n_vertices}, m={n_edges}, p={p}")

    cost_matrix = build_cost_matrix(n_vertices, edges)
    dist_all_pairs = floyd_marshall(cost_matrix)
    dist_dict = matrix_to_dict(dist_all_pairs)
    demand_weights = {i: 1.0 for i in range(1, n_vertices + 1)}

    # 6) Resolve p-median com redundância K usando VNS
    start_time = time.time()
    S_best, FO_best = solve_pmedian_redundancy_vns(
        dist_dict=dist_dict,
        demand_weights=demand_weights,
        p=p,
        K_redundancy=args.k_redundancy,
        max_iter=args.max_iter,
        k_max=args.k_max
    )
    end_time = time.time()

    print("\n" + "="*50)
    print("✨ Solução VNS p-Median Redundância (K) ✨")
    print(f"Valor objetivo (FO) final (Soma das K Distâncias): {FO_best:.4f}")
    print(f"Facilidades abertas (sites): {S_best}")
    print(f"Tempo de execução total: {end_time - start_time:.2f} segundos")
    print("="*50)
    
    # 7) Plotagem (Opcional)
    if args.print:
        # Reavalia a solução final para obter a atribuição primária (k=1) para plotagem
        _, assignment = evaluate_pmedian_redundancy(S_best, dist_dict, demand_weights, K_redundancy=1)
        
        plot_solution(
            n_vertices=n_vertices,
            edges=edges,            
            dist_dict=dist_dict,    
            centers=list(S_best),
            assignment=assignment,
            filename="solution_pmedian_redundancy.png",
        )


if __name__ == "__main__":
    main()