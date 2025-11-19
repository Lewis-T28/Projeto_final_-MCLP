# main_vns_cuda.py

import time
import argparse
from typing import Dict, List, Tuple
from .vns_core import solve_mclp_backup_vns, evaluate_solution_cuda # Importa o VNS core e o avaliador CUDA

# Importações das funções de utilidade (seus arquivos)
from src.reader_pmed import read_pmed_file, build_cost_matrix 
from src.floyd_marshall_algo import floyd_marshall 
from src.build_dist_matrix import matrix_to_dict
from src.print_nodes_backup import plot_solution_backup 

"""
Exemplo de execução
python -m vns_mclp.main_vns_cuda instancias/pmed1.txt -R 50.0 --beta 0.3 --max-iter 3 --k-max 3 --print
"""


def main():
    parser = argparse.ArgumentParser(
        description="Resolve MCLP com backup coverage usando VNS (Paralelizado em CUDA)."
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
        help="Se usado, gera imagem da solução (solution_mclp_backup_*.png).",
    )

    args = parser.parse_args()

    # --- 1) Leitura e Processamento dos Dados ---
    n_vertices, n_edges, p, edges = read_pmed_file(args.arquivo)
    print(f"Instância lida: n={n_vertices}, m={n_edges}, p={p}")

    cost_matrix = build_cost_matrix(n_vertices, edges)
    dist_all_pairs = floyd_marshall(cost_matrix)
    dist_dict = matrix_to_dict(dist_all_pairs)
    demand_weights = {i: 1.0 for i in range(1, n_vertices + 1)}

    # --- 2) Execução do VNS (com avaliação CUDA) ---
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
    print("✨ Solução VNS Final (CUDA) ✨")
    print(f"Valor objetivo (FO) final: {FO_best:.4f}")
    print(f"Facilidades abertas (sites): {S_best}")
    print(f"Tempo de execução total: {end_time - start_time:.2f} segundos")
    print("="*50)
    
    # --- 3) Plotagem (Opcional) ---
    if args.print:
        # Reavalia a solução final (usando CUDA) para obter as contagens e atribuições
        _, coverage_counts = evaluate_solution_cuda(S_best, dist_dict, demand_weights, args.radius, args.beta)
        
        centers = list(S_best)
        assign_primary = {}
        assign_backup = {}
        
        # Determina a atribuição Primária e de Backup (para fins de visualização)
        for i in range(1, n_vertices + 1):
            
            feasible_centers = sorted(
                [j for j in centers if dist_dict[i].get(j, float('inf')) <= args.radius],
                key=lambda j: dist_dict[i][j]
            )
            
            if coverage_counts.get(i, 0) >= 1 and feasible_centers:
                assign_primary[i] = feasible_centers[0]
                
                if coverage_counts.get(i, 0) >= 2 and len(feasible_centers) >= 2:
                    assign_backup[i] = feasible_centers[1]


        plot_solution_backup(
            n_vertices=n_vertices,
            edges=edges,
            dist_dict=dist_dict,
            centers=centers,
            assign_primary=assign_primary,
            assign_backup=assign_backup,
            instance_name=args.arquivo,
            filename="solution_mclp_vns_cuda",
        )


if __name__ == "__main__":
    main()