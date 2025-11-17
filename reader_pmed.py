# reader_pmed.py
from typing import List, Tuple


def read_pmed_file(path: str):
    """
    Lê o arquivo no formato OR-Library (pmed*).

    Formato:
        n_vertices n_arestas p
        u v custo
        ...

    Retorna:
        n_vertices, n_arestas, p, edges
        onde edges é uma lista de tuplas (u, v, custo)
    """
    edges: List[Tuple[int, int, float]] = []

    with open(path, "r") as f:
        first = f.readline().strip()
        if not first:
            raise ValueError("Arquivo vazio ou formato inválido.")

        n_vertices, n_edges, p = map(int, first.split())

        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v, c = map(int, line.split())
            edges.append((u, v, float(c)))

    return n_vertices, n_edges, p, edges


def build_cost_matrix(n_vertices: int, edges: List[Tuple[int, int, float]]):
    """
    Monta a matriz de custos inicial (adjacência) a partir da lista de arestas.

    Retorna:
        cost[i][j] com i,j de 1..n_vertices.
        Usa float('inf') para pares sem aresta direta e 0 na diagonal.
    """
    INF = float("inf")

    # matriz (n+1)x(n+1) usando índices 1..n
    cost = [[INF] * (n_vertices + 1) for _ in range(n_vertices + 1)]

    for i in range(1, n_vertices + 1):
        cost[i][i] = 0.0

    for u, v, c in edges:
        # grafo não-direcionado
        if c < cost[u][v]:
            cost[u][v] = c
            cost[v][u] = c

    return cost
