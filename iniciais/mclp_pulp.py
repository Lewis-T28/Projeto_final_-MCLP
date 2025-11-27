# mclp_pulp.py
from typing import Dict, Iterable, Optional
import argparse
import pulp

from src.reader_pmed import read_pmed_file, build_cost_matrix
from src.floyd_marshall_algo import floyd_marshall
from src.build_dist_matrix import matrix_to_dict
from src.print_nodes import plot_solution

"""
Exemple of use
python -m mclp_pulp instancias/pmed12.txt -R 50 --uniform-weight
"""


def solve_mclp(
    dist_matrix: Dict[int, Dict[int, float]],
    demand_weights: Dict[int, float],
    p: int,
    radius: float,
    candidate_sites: Optional[Iterable[int]] = None,
    demand_nodes: Optional[Iterable[int]] = None,
):
    """
    Resolve o Maximum Coverage Location Problem (MCLP) usando PuLP.

    dist_matrix[i][j] = distância da demanda i ao candidato j.
    demand_weights[i] = demanda em i.
    p = número máximo de facilidades.
    radius = raio de cobertura (d_ij <= radius => j cobre i).
    """

    if candidate_sites is None:
        candidate_sites = sorted(dist_matrix.keys())
    else:
        candidate_sites = list(candidate_sites)

    if demand_nodes is None:
        demand_nodes = sorted(demand_weights.keys())
    else:
        demand_nodes = list(demand_nodes)

    # Conjunto de candidatos que cobrem cada demanda
    cover_sets = {
        i: [j for j in candidate_sites if dist_matrix[i][j] <= radius]
        for i in demand_nodes
    }

    model = pulp.LpProblem("MCLP", sense=pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", candidate_sites, 0, 1, cat="Binary")
    y = pulp.LpVariable.dicts("y", demand_nodes, 0, 1, cat="Binary")

    # Objetivo
    model += pulp.lpSum(demand_weights[i] * y[i] for i in demand_nodes)

    # Limite no número de facilidades
    model += pulp.lpSum(x[j] for j in candidate_sites) <= p, "Limit_number_of_facilities"

    # Cobertura
    for i in demand_nodes:
        if cover_sets[i]:
            model += pulp.lpSum(x[j] for j in cover_sets[i]) >= y[i], f"Coverage_{i}"
        else:
            model += y[i] == 0, f"NoCoverage_{i}"

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    x_solution = {j: int(pulp.value(x[j]) > 0.5) for j in candidate_sites}
    y_solution = {i: int(pulp.value(y[i]) > 0.5) for i in demand_nodes}

    return model, x_solution, y_solution


def main():
    parser = argparse.ArgumentParser(description="Resolve MCLP usando instância pmed (OR-Library).")
    parser.add_argument("arquivo", help="Arquivo pmed* (formato OR-Library).")
    parser.add_argument(
        "--radius",
        "-R",
        type=float,
        required=True,
        help="Raio de cobertura para o MCLP.",
    )
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

    # 2) Monta matriz de custos inicial
    cost_matrix = build_cost_matrix(n_vertices, edges)

    # 3) Executa Floyd–Warshall
    dist_all_pairs = floyd_marshall(cost_matrix)

    # 4) Converte para dict-of-dicts (dist_matrix para o MCLP)
    dist_dict = matrix_to_dict(dist_all_pairs)

    # 5) Define pesos de demanda
    if args.uniform_weight:
        demand_weights = {i: 1.0 for i in range(1, n_vertices + 1)}
    else:
        # por enquanto, default também é 1.0; aqui dá pra plugar leitura de
        # pesos reais se você tiver.
        demand_weights = {i: 1.0 for i in range(1, n_vertices + 1)}

    # 6) Resolve MCLP
    model, x_sol, y_sol = solve_mclp(
        dist_matrix=dist_dict,
        demand_weights=demand_weights,
        p=p,
        radius=args.radius,
    )

    print("Status:", pulp.LpStatus[model.status])
    print("Valor objetivo (demanda coberta):", pulp.value(model.objective))

    print("\nFacilidades abertas (x_j = 1):")
    for j in sorted(x_sol):
        if x_sol[j] == 1:
            print(f"  - site {j}")

    covered = [i for i, val in y_sol.items() if val == 1]
    print(f"\nDemandas cobertas ({len(covered)} de {n_vertices}):")
    # imprime só alguns se for muito grande
    print(covered[:50], "..." if len(covered) > 50 else "")

    # --- NEW: if --print, build assignment and plot solution ---
    if args.print:
        # Simple rule: each covered demand is assigned to the closest open center
        assignment = {}
        centers = [j for j, val in x_sol.items() if val == 1]
        for i in range(1, n_vertices + 1):
            if y_sol.get(i, 0) == 1:
                feasible_centers = [
                    j for j in centers if dist_dict[i][j] <= args.radius
                ]
                if feasible_centers:
                    j_star = min(feasible_centers, key=lambda j: dist_dict[i][j])
                    assignment[i] = j_star
            # uncovered nodes just won't appear in assignment

        plot_solution(
            n_vertices=n_vertices,
            edges=edges,
            dist_dict=dist_dict,
            centers=centers,
            assignment=assignment,
            filename="solution_mclp.png",
        )


if __name__ == "__main__":
    main()
