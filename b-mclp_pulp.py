"""
MCLP com Backup Coverage usando instâncias pmed (OR-Library)

FO:
    max sum_i w_i (y_i + beta * y2_i)

onde:
    y_i   = 1 se demanda i tem >= 1 facilidade dentro do raio R
    y2_i  = 1 se demanda i tem >= 2 facilidades dentro do raio R
"""

from typing import Dict, Iterable, Optional
import argparse
import pulp

from reader_pmed import read_pmed_file, build_cost_matrix
from floyd_marshall_algo import floyd_marshall
from build_dist_matrix import matrix_to_dict
from print_nodes_backup import plot_solution_backup


"""
# MCLP com backup, beta = 0.3, raio 50, pesos uniformes, com gráfico
python b-mclp_pulp.py pmed1.txt -R 50 --beta 0.3 --uniform-weight --print
"""


# ---------------------------------------------------------------------
#   Modelo MCLP com backup coverage
# ---------------------------------------------------------------------
def solve_mclp_backup(
    dist_matrix: Dict[int, Dict[int, float]],
    demand_weights: Dict[int, float],
    p: int,
    radius: float,
    beta: float = 0.5,  # peso da cobertura de backup
    candidate_sites: Optional[Iterable[int]] = None,
    demand_nodes: Optional[Iterable[int]] = None,
):
    """
    Resolve o MCLP com backup coverage.

    dist_matrix[i][j] = distância entre i e j (menor caminho).
    demand_weights[i] = demanda em i.
    p  = número máximo de facilidades.
    radius = raio de cobertura R.
    beta   = peso da cobertura backup.

    Variáveis:
        x_j   = 1 se instala facilidade em j
        y_i   = 1 se i tem >= 1 facilidade em N(i)
        y2_i  = 1 se i tem >= 2 facilidades em N(i)
    """

    if candidate_sites is None:
        candidate_sites = sorted(dist_matrix.keys())
    else:
        candidate_sites = list(candidate_sites)

    if demand_nodes is None:
        demand_nodes = sorted(demand_weights.keys())
    else:
        demand_nodes = list(demand_nodes)

    # N(i): candidatos que cobrem i (d_ij <= R)
    cover_sets = {
        i: [j for j in candidate_sites if dist_matrix[i][j] <= radius]
        for i in demand_nodes
    }

    model = pulp.LpProblem("MCLP_Backup", sense=pulp.LpMaximize)

    # x_j: instala ou não
    x = pulp.LpVariable.dicts("x", candidate_sites, 0, 1, cat="Binary")

    # y_i: coberto por pelo menos uma facilidade
    y = pulp.LpVariable.dicts("y", demand_nodes, 0, 1, cat="Binary")

    # y2_i: coberto por pelo menos duas facilidades (backup)
    y2 = pulp.LpVariable.dicts("y2", demand_nodes, 0, 1, cat="Binary")

    # FO: max sum_i w_i (y_i + beta * y2_i)
    model += pulp.lpSum(
        demand_weights[i] * (y[i] + beta * y2[i]) for i in demand_nodes
    )

    # Número máximo de facilidades
    model += pulp.lpSum(x[j] for j in candidate_sites) <= p, "Limit_number_of_facilities"

    # Restrições de cobertura e backup
    for i in demand_nodes:
        if cover_sets[i]:
            # cobertura simples (MCLP padrão)
            model += (
                pulp.lpSum(x[j] for j in cover_sets[i]) >= y[i],
                f"Coverage_{i}",
            )

            # cobertura de backup: precisa de pelo menos 2 facilidades
            # sum_{j in N(i)} x_j >= 2 * y2_i
            model += (
                pulp.lpSum(x[j] for j in cover_sets[i]) >= 2 * y2[i],
                f"BackupCoverage_{i}",
            )

            # faz sentido exigir que quem tem backup esteja coberto:
            # y2_i <= y_i
            model += y2[i] <= y[i], f"BackupImpliesCoverage_{i}"
        else:
            # ninguém cobre i dentro de R -> não pode ser coberto nem ter backup
            model += y[i] == 0, f"NoCoverage_{i}"
            model += y2[i] == 0, f"NoBackup_{i}"

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    x_solution = {j: int(pulp.value(x[j]) > 0.5) for j in candidate_sites}
    y_solution = {i: int(pulp.value(y[i]) > 0.5) for i in demand_nodes}
    y2_solution = {i: int(pulp.value(y2[i]) > 0.5) for i in demand_nodes}

    return model, x_solution, y_solution, y2_solution


# ---------------------------------------------------------------------
#   main() com argparse
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Resolve MCLP com backup coverage usando instância pmed (OR-Library)."
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
        default=0.5,
        help='Peso da cobertura de backup (beta) na FO. Default: 0.5.',
    )
    parser.add_argument(
        "--uniform-weight",
        action="store_true",
        help="Usa peso 1 para todas as demandas (default).",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Se usado, gera imagem da solução (solution_mclp.png).",
    )

    args = parser.parse_args()

    # 1) Dados da instância
    n_vertices, n_edges, p, edges = read_pmed_file(args.arquivo)
    print(f"Instância lida: n={n_vertices}, m={n_edges}, p={p}")

    # 2) Matriz de custos original
    cost_matrix = build_cost_matrix(n_vertices, edges)

    # 3) Floyd–Warshall
    dist_all_pairs = floyd_marshall(cost_matrix)

    # 4) Dict-of-dicts
    dist_dict = matrix_to_dict(dist_all_pairs)

    # 5) Pesos de demanda
    if args.uniform_weight:
        demand_weights = {i: 1.0 for i in range(1, n_vertices + 1)}
    else:
        demand_weights = {i: 1.0 for i in range(1, n_vertices + 1)}

    # 6) Resolve MCLP com backup
    model, x_sol, y_sol, y2_sol = solve_mclp_backup(
        dist_matrix=dist_dict,
        demand_weights=demand_weights,
        p=p,
        radius=args.radius,
        beta=args.beta,
    )

    print("Status:", pulp.LpStatus[model.status])
    print("Valor objetivo:", pulp.value(model.objective))

    # Centros abertos
    print("\nFacilidades abertas (x_j = 1):")
    centers = [j for j, val in x_sol.items() if val == 1]
    for j in centers:
        print(f"  - site {j}")

    # Cobertura simples e backup
    covered = [i for i, val in y_sol.items() if val == 1]
    backup_covered = [i for i, val in y2_sol.items() if val == 1]

    print(f"\nDemandas cobertas (>=1 facilidade): {len(covered)} de {n_vertices}")
    print(covered[:50], "..." if len(covered) > 50 else "")

    print(
        f"\nDemandas com backup (>=2 facilidades dentro de R): "
        f"{len(backup_covered)} de {n_vertices}"
    )
    print(backup_covered[:50], "..." if len(backup_covered) > 50 else "")

    # 7) Plot (opcional)
    if args.print:
        # 1st coverage assignment
        assign_primary = {}
        for i in range(1, n_vertices + 1):
            if y_sol[i] == 1:
                feasible = [j for j in centers if dist_dict[i][j] <= args.radius]
                if feasible:
                    assign_primary[i] = min(feasible, key=lambda j: dist_dict[i][j])
                else:
                    assign_primary[i] = None

        # 2nd coverage assignment (backup)
        assign_backup = {}
        for i in range(1, n_vertices + 1):
            if y2_sol[i] == 1:
                feasible = [j for j in centers if dist_dict[i][j] <= args.radius]
                # remove o centro da 1ª cobertura
                if i in assign_primary and assign_primary[i] in feasible:
                    feasible.remove(assign_primary[i])
                if feasible:
                    assign_backup[i] = min(feasible, key=lambda j: dist_dict[i][j])
                else:
                    assign_backup[i] = None

        plot_solution_backup(
            n_vertices=n_vertices,
            edges=edges,
            dist_dict=dist_dict,
            centers=centers,
            assign_primary=assign_primary,
            assign_backup=assign_backup,
            instance_name=args.arquivo,   # ← AQUI!
            filename="solution_mclp_backup",
        )




if __name__ == "__main__":
    main()
