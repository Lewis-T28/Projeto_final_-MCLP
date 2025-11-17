# build_dist_matrix.py
from typing import Dict, List


def matrix_to_dict(dist_matrix: List[List[float]]) -> Dict[int, Dict[int, float]]:
    """
    Converte uma matriz dist[i][j] (1..n) em um dict-of-dicts:
        {i: {j: dist_ij}}

    Ideal para ser usado pelo solve_mclp().
    """
    n = len(dist_matrix) - 1
    dist_dict: Dict[int, Dict[int, float]] = {}

    for i in range(1, n + 1):
        dist_dict[i] = {}
        for j in range(1, n + 1):
            dist_dict[i][j] = dist_matrix[i][j]

    return dist_dict
