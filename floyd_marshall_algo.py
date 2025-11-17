# floyd_marshall_algo.py
from typing import List


def floyd_marshall(cost: List[List[float]]):
    """
    Executa o algoritmo de Floyd–Warshall para todos os pares.

    Parâmetro:
        cost: matriz de custos inicial, cost[i][j].

    Retorna:
        dist: matriz de menores distâncias entre todos os pares.
    """
    n = len(cost) - 1  # assumindo índices 1..n
    INF = float("inf")

    # cópia profunda
    dist = [row[:] for row in cost]

    for k in range(1, n + 1):
        for i in range(1, n + 1):
            ik = dist[i][k]
            if ik == INF:
                continue
            for j in range(1, n + 1):
                alt = ik + dist[k][j]
                if alt < dist[i][j]:
                    dist[i][j] = alt

    return dist
