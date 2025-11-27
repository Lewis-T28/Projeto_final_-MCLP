# mclp_pulp.py  -> agora implementa o problema p-median

from typing import Dict, Iterable, Optional
import argparse
import pulp

from src.reader_pmed import read_pmed_file, build_cost_matrix
from src.floyd_marshall_algo import floyd_marshall
from src.build_dist_matrix import matrix_to_dict
from src.print_nodes import plot_solution

"""
Example of use (p-median)
python -m pmed_pulp instancias/pmed1.txt --uniform-weight
"""


def solve_pmedian(
    dist_matrix: Dict[int, Dict[int, float]],
    demand_weights: Dict[int, float],
    p: int,
    candidate_sites: Optional[Iterable[int]] = None,
    demand_nodes: Optional[Iterable[int]] = None,
):
    """
    Resolve o problema p-median clássico usando PuLP.

    dist_matrix[i][j] = distância da demanda i ao candidato j.
    demand_weights[i] = demanda em i.
    p = número de facilidades a instalar (medianas).

    Retorna:
        model, x_solution, z_solution
    """

    if candidate_sites is None:
        candidate_sites = sorted(dist_matrix.keys())
    else:
        candidate_sites = list(candidate_sites)

    if demand_nodes is None:
        demand_nodes = sorted(demand_weights.keys())
    else:
        demand_nodes = list(demand_nodes)

    # Modelo
    model = pulp.LpProblem("P_Median", sense=pulp.LpMinimize)

    # x_j = 1 se instala em j
    x = pulp.LpVariable.dicts("x", candidate_sites, 0, 1, cat="Binary")

    # z_ij = 1 se demanda i é atendida pelo site j
    z = pulp.LpVariable.dicts(
        "z",
        (demand_nodes, candidate_sites),
        lowBound=0,
        upBound=1,
        cat="Binary",
    )

    # Objetivo: minimizar soma_i,j w_i * d_ij * z_ij
    model += pulp.lpSum(
        demand_weights[i] * dist_matrix[i][j] * z[i][j]
        for i in demand_nodes
        for j in candidate_sites
    )

    # Exatamente p facilidades abertas
    model += pulp.lpSum(x[j] for j in candidate_sites) == p, "Num_facilities"

    # Cada demanda é atendida por exatamente um site
    for i in demand_nodes:
        model += pulp.lpSum(z[i][j] for j in candidate_sites) == 1, f"Assign_{i}"

    # Só pode atribuir a sites abertos: z_ij <= x_j
    for i in demand_nodes:
        for j in candidate_sites:
            model += z[i][j] <= x[j], f"Link_{i}_{j}"

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    x_solution = {j: int(pulp.value(x[j]) > 0.5) for j in candidate_sites}
    z_solution = {
        (i, j): int(pulp.value(z[i][j]) > 0.5)
        for i in demand_nodes
        for j in candidate_sites
    }

    return model, x_solution, z_solution


def main():
    parser = argparse.ArgumentParser(description="Resolve p-median usando instância pmed (OR-Library).")
    parser.add_argument("arquivo", help="Arquivo pmed* (formato OR-Library).")
    parser.add_argument(
        "--uniform-weight",
        action="store_true",
        help="Usa peso 1 para todas as demandas (default).",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="If set, generate a solution graph image (solution_mclp.png).",
    )


    args = parser.parse_args()

    # 1) Recupera dados do arquivo
    n_vertices, n_edges, p, edges = read_pmed_file(args.arquivo)
    print(f"Instância lida: n={n_vertices}, m={n_edges}, p={p}")

    # 2) Matriz de custos inicial
    cost_matrix = build_cost_matrix(n_vertices, edges)

    # 3) Floyd–Warshall para obter todas as distâncias
    dist_all_pairs = floyd_marshall(cost_matrix)

    # 4) Dict-of-dicts para o solver
    dist_dict = matrix_to_dict(dist_all_pairs)

    # 5) Pesos de demanda
    if args.uniform_weight:
        demand_weights = {i: 1.0 for i in range(1, n_vertices + 1)}
    else:
        # por enquanto, default também é 1.0; aqui você pode plugar dados reais
        demand_weights = {i: 1.0 for i in range(1, n_vertices + 1)}

    # 6) Resolve p-median
    model, x_sol, z_sol = solve_pmedian(
        dist_matrix=dist_dict,
        demand_weights=demand_weights,
        p=p,
    )

    print("Status:", pulp.LpStatus[model.status])
    print("Custo total (função objetivo):", pulp.value(model.objective))

    print("\nMedianas abertas (x_j = 1):")
    centers = [j for j in sorted(x_sol) if x_sol[j] == 1]
    for j in centers:
        print(f"  - site {j}")

    # opcional: mostrar para onde cada demanda foi atribuída
    print("\nAtribuição das demandas (i -> j):")
    assignment = {}
    for (i, j), val in z_sol.items():
        if val == 1:
            assignment[i] = j
            print(f"  demanda {i} -> site {j}")
    
    # --- NEW: if --print, plot solution ---
    if args.print:
        plot_solution(
            n_vertices=n_vertices,
            edges=edges,            # original edges from reader_pmed
            dist_dict=dist_dict,    # from matrix_to_dict(Floyd)
            centers=centers,
            assignment=assignment,
            filename="solution_pmedian.png",
        )


if __name__ == "__main__":
    main()
