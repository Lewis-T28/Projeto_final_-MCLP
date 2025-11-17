# print_nodes.py

from typing import Dict, Iterable, Tuple, List, Optional
import matplotlib.pyplot as plt
import networkx as nx
import os
from datetime import datetime


def plot_solution(
    n_vertices: int,
    edges: List[Tuple[int, int, float]],
    dist_dict: Dict[int, Dict[int, float]],
    centers: Iterable[int],
    assignment: Dict[int, Optional[int]],
    filename: str = "solution",
):
    """
    Gera um grafo da solução com timestamp e salva em /grafo_prints/.

    - Instalações (centers) em amarelo.
    - Demais nós em azul.
    - Arestas verde-claro para conexões center -> node coberto.
    - Para nós não cobertos, uma aresta cinza até o centro mais próximo.
    """

    # ----------------------------------------------------------------------
    # 1) Criar pasta grafo_prints/ caso não exista
    # ----------------------------------------------------------------------
    output_dir = "grafo_prints"
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------------
    # 2) Criar nome com timestamp
    # ----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, f"{filename}_{timestamp}.png")

    centers = set(centers)

    # Grafo apenas para montar o layout
    G = nx.Graph()
    G.add_nodes_from(range(1, n_vertices + 1))
    for u, v, c in edges:
        G.add_edge(u, v, weight=c)

    pos = nx.kamada_kawai_layout(G)

    plt.figure(figsize=(12, 12))

    # Cores dos nós
    node_colors = []
    for node in G.nodes():
        if node in centers:
            node_colors.append("#ffcc66")  # amarelo claro
        else:
            node_colors.append("#1f78b4")  # azul

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=130,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.4,
    )

    # ----------------------------------------------------------------------
    # 3) Edges da solução
    # ----------------------------------------------------------------------

    # 3.1 Arestas verdes (demanda -> centro)
    green_edges = []
    green_labels = {}

    for i, j in assignment.items():
        if j is None:
            continue
        green_edges.append((i, j))
        green_labels[(i, j)] = f"{dist_dict[i][j]:.1f}"

    if green_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=green_edges,
            edge_color="lightgreen",
            width=1.8,
            alpha=0.9,
        )
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=green_labels,
            font_size=6,
        )

    # 3.2 Para nós não cobertos: aresta cinza para o centro mais próximo
    all_nodes = set(range(1, n_vertices + 1))
    assigned_nodes = {i for i in assignment.keys() if assignment[i] is not None}
    uncovered_nodes = sorted(all_nodes - assigned_nodes)

    gray_edges = []
    gray_labels = {}

    if centers:
        for i in uncovered_nodes:
            best_center = None
            best_dist = float("inf")
            for j in centers:
                d = dist_dict[i][j]
                if d < best_dist:
                    best_dist = d
                    best_center = j

            if best_center is not None and best_dist < float("inf"):
                gray_edges.append((i, best_center))
                gray_labels[(i, best_center)] = f"{best_dist:.1f}"

    if gray_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=gray_edges,
            edge_color="gray",
            width=1.0,
            alpha=0.8,
        )
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=gray_labels,
            font_size=6,
        )

    # Rótulos dos nós
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=7,
        font_color="black",
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Solution graph saved to: {output_path}")
