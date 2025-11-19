# vns_b_mclp_pulp.py (Nome do Arquivo)

import random
import time
import math
import argparse
from typing import Dict, Iterable, Set, Tuple, List, Optional, Any

# =====================================================================
#   IMPORTAÇÕES DAS FUNÇÕES DE UTILIDADE (Seus arquivos anexos)
# =====================================================================

# Funções de leitura e matriz de custos
from src.reader_pmed import read_pmed_file, build_cost_matrix 

# Algoritmo de menor caminho (distâncias de todos os pares)
from src.floyd_marshall_algo import floyd_marshall 

# Conversão da matriz de distâncias para dict-of-dicts
from src.build_dist_matrix import matrix_to_dict

# Função de plotagem (grava o gráfico em arquivo)
from src.print_nodes_backup import plot_solution_backup 


# =====================================================================
#   1. Funções de Avaliação e Estrutura MCLP
# =====================================================================

def evaluate_solution(
    solution_set: Set[int],
    dist_dict: Dict[int, Dict[int, float]],
    demand_weights: Dict[int, float],
    radius: float,
    beta: float,
) -> Tuple[float, Dict[int, int]]:
    """
    Calcula o valor da Função Objetivo (FO) para um dado conjunto de facilidades abertas.
    
    FO = sum_i w_i * (y_i + beta * y2_i)
    
    Retorna o valor da FO e a contagem de cobertura para cada nó de demanda.
    """
    total_objective = 0.0
    demand_nodes = demand_weights.keys()
    
    # Armazena a contagem de facilidades que cobrem cada demanda
    coverage_count: Dict[int, int] = {}
    
    for i in demand_nodes:
        # Conta quantas facilidades em 'solution_set' cobrem a demanda 'i'
        count = sum(1 for j in solution_set if dist_dict[i].get(j, float('inf')) <= radius)
        coverage_count[i] = count

        # y_i = 1 se count >= 1 (cobertura primária)
        y_i = 1 if count >= 1 else 0
        # y2_i = 1 se count >= 2 (cobertura de backup)
        y2_i = 1 if count >= 2 else 0

        total_objective += demand_weights[i] * (y_i + beta * y2_i)

    return total_objective, coverage_count


# =====================================================================
#   2. Funções de Vizinhança (Neighborhoods)
# =====================================================================

def neighborhood_k_exchange_random(solution_set: Set[int], candidates: Iterable[int], k: int) -> Set[int]:
    """
    Aplica uma operação aleatória de k-Exchange (Shaking).
    Remove k facilidades abertas e adiciona k fechadas.
    """
    p = len(solution_set)
    open_facilities = list(solution_set)
    # Identifica as facilidades fechadas disponíveis
    closed_facilities = [j for j in candidates if j not in solution_set]

    # Verifica se é possível realizar a troca (precisa de k abertos e k fechados)
    if len(open_facilities) < k or len(closed_facilities) < k:
        return solution_set.copy() 

    # Escolhe aleatoriamente k para remover e k para adicionar
    remove_j_list = random.sample(open_facilities, k)
    add_j_list = random.sample(closed_facilities, k)

    new_solution = solution_set.copy()
    for j in remove_j_list:
        new_solution.remove(j)
    for j in add_j_list:
        new_solution.add(j)
    
    # O tamanho deve permanecer p
    assert len(new_solution) == p
    return new_solution

# =====================================================================
#   3. Busca Local (Local Search - 1-Exchange Best Improvement)
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
    Itera até que nenhum movimento 1-Exchange melhore a solução atual.
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
        
        # Armazena o melhor movimento encontrado na iteração atual
        local_best_objective = best_objective
        local_best_solution = best_solution.copy()
        
        # Itera por todos os p * (N-p) movimentos 1-Exchange
        for remove_j in open_facilities:
            for add_j in closed_facilities:
                # Gera o novo conjunto de facilidades
                new_solution = (best_solution - {remove_j}) | {add_j}
                
                # Avalia o novo conjunto
                new_objective, _ = evaluate_solution(
                    new_solution, dist_dict, demand_weights, radius, beta
                )
                
                if new_objective > local_best_objective:
                    local_best_objective = new_objective
                    local_best_solution = new_solution
                    improved = True

        # Aplica o melhor movimento da iteração, se houver
        if improved:
            best_objective = local_best_objective
            best_solution = local_best_solution
        
    return best_solution, best_objective

# =====================================================================
#   4. VNS Principal
# =====================================================================

def solve_mclp_backup_vns(
    dist_dict: Dict[int, Dict[int, float]],
    demand_weights: Dict[int, float],
    p: int,
    radius: float,
    beta: float = 0.5,
    max_iter: int = 100,
    k_max: int = 3, # Vizinhanças: N1 (k=1), N2 (k=2), N3 (k=3)
):
    """
    Resolve o MCLP com backup coverage usando a heurística VNS.
    """
    candidate_sites = sorted(dist_dict.keys())
    
    # 1. Solução Inicial: Escolha aleatória de p facilidades
    initial_solution_list = random.sample(candidate_sites, p)
    S_current: Set[int] = set(initial_solution_list)
    
    FO_current, _ = evaluate_solution(S_current, dist_dict, demand_weights, radius, beta)
    
    S_best = S_current.copy()
    FO_best = FO_current
    
    print(f"\n--- VNS INICIADO (p={p}, R={radius}, beta={beta}) ---")
    print(f"VNS Inicial: FO={FO_current:.4f}, facilidades={S_current}")

    iteration = 0
    while iteration < max_iter:
        
        k = 1
        while k <= k_max:
            # a. Shaking (Mudança de Vizinhança)
            S_prime = neighborhood_k_exchange_random(S_current, candidate_sites, k)
            
            # b. Busca Local (Refinamento usando 1-Exchange)
            S_double_prime, FO_new = local_search_1_exchange(
                S_prime, dist_dict, demand_weights, radius, beta, candidate_sites
            )

            # c. Move or Not
            if FO_new > FO_current:
                # Melhora - Move, Atualiza Melhor Global, Reinicia k=1 (Mudança de Vizinhança)
                S_current = S_double_prime
                FO_current = FO_new
                
                if FO_new > FO_best:
                    FO_best = FO_new
                    S_best = S_double_prime.copy()
                
                print(f"Iter {iteration:03d}, FO={FO_current:.4f} (Melhora), k={k} -> Reinicia k=1")
                k = 1 # Reinicia o ciclo de vizinhanças
            else:
                # Não melhora - Incrementa k (Explora próxima vizinhança)
                k += 1

        iteration += 1
        
    return S_best, FO_best

# =====================================================================
#   5. main_vns() com argparse e Plotagem
# =====================================================================

def main_vns():
    """Fluxo principal do VNS com leitura real e plotagem."""
    parser = argparse.ArgumentParser(
        description="Resolve MCLP com backup coverage usando a heurística VNS."
    )
    parser.add_argument("arquivo", help="Arquivo pmed* (formato OR-Library).")
    parser.add_argument(
        "--radius",
        "-R",
        type=float,
        required=True,
        help="Raio de cobertura para o MCLP.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help='Peso da cobertura de backup (beta) na FO. Default: 0.3.',
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
        help="Se usado, gera imagem da solução (solution_mclp_backup_*.png) no diretório grafo_prints.",
    )

    args = parser.parse_args()

    # 1) Leitura dos dados da instância
    n_vertices, n_edges, p, edges = read_pmed_file(args.arquivo)
    print(f"Instância lida: n={n_vertices}, m={n_edges}, p={p}")

    # 2) Matriz de custos original
    cost_matrix = build_cost_matrix(n_vertices, edges)

    # 3) Floyd–Warshall (Distâncias do menor caminho)
    dist_all_pairs = floyd_marshall(cost_matrix)

    # 4) Dict-of-dicts
    dist_dict = matrix_to_dict(dist_all_pairs)

    # 5) Pesos de demanda (Uniforme)
    # No VNS, geralmente assumimos w_i=1.0 se não especificado (maximizar cobertura)
    demand_weights = {i: 1.0 for i in range(1, n_vertices + 1)}

    # 6) Resolve MCLP com backup usando VNS
    start_time = time.time()
    S_best, FO_best = solve_mclp_backup_vns(
        dist_dict=dist_dict,
        demand_weights=demand_weights,
        p=p,
        radius=args.radius,
        beta=args.beta,
        max_iter=args.max_iter,
        k_max=args.k_max
    )
    end_time = time.time()

    print("\n" + "="*50)
    print("✨ Solução VNS Final ✨")
    print(f"Valor objetivo (FO) final: {FO_best:.4f}")
    print(f"Facilidades abertas (sites): {S_best}")
    print(f"Tempo de execução total: {end_time - start_time:.2f} segundos")
    print("="*50)
    
    # 7) Plotagem (Opcional)
    if args.print:
        
        # Avalia a solução final para obter as contagens e atribuições para a plotagem
        _, coverage_counts = evaluate_solution(S_best, dist_dict, demand_weights, args.radius, args.beta)
        
        centers = list(S_best)
        assign_primary = {}
        assign_backup = {}
        
        # Determina a atribuição Primária e de Backup (para fins de visualização)
        for i in range(1, n_vertices + 1):
            
            # Encontra todos os centros que cobrem a demanda i
            feasible_centers = sorted(
                [j for j in centers if dist_dict[i].get(j, float('inf')) <= args.radius],
                key=lambda j: dist_dict[i][j]
            )
            
            # Atribuição Primária (o centro mais próximo)
            if coverage_counts.get(i, 0) >= 1 and feasible_centers:
                assign_primary[i] = feasible_centers[0]
                
                # Atribuição de Backup (o segundo centro mais próximo)
                if coverage_counts.get(i, 0) >= 2:
                    if len(feasible_centers) >= 2:
                        assign_backup[i] = feasible_centers[1]


        plot_solution_backup(
            n_vertices=n_vertices,
            edges=edges,
            dist_dict=dist_dict,
            centers=centers,
            assign_primary=assign_primary,
            assign_backup=assign_backup,
            instance_name=args.arquivo,
            filename="solution_mclp_backup",
        )


if __name__ == "__main__":
    main_vns()