# print_nodes_backup.py

from typing import Dict, Iterable, List, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx
import os
from datetime import datetime


def plot_solution_backup(
    n_vertices: int,
    edges: List[Tuple[int, int, float]],
    dist_dict: Dict[int, Dict[int, float]],
    centers: Iterable[int],
    assign_primary: Dict[int, Optional[int]],
    assign_backup: Dict[int, Optional[int]],
    instance_name: str,
    filename: str = "solution_mclp_backup"
):
    """
    Desenha solução do MCLP com backup coverage.

    Com legenda mostrando:
    - cores utilizadas
    - significados
    - nome do arquivo da instância lida (pmed1.txt)
    """

    centers = set(centers)

    # -------------------------------
    # Criar diretório de saída
    # -------------------------------
    out_dir = "grafo_prints"
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(out_dir, f"{filename}_{timestamp}.png")

    # -------------------------------
    # Layout base
    # -------------------------------
    G = nx.Graph()
    G.add_nodes_from(range(1, n_vertices + 1))
    for u, v, c in edges:
        G.add_edge(u, v)

    pos = nx.kamada_kawai_layout(G)

    plt.figure(figsize=(14, 14))

    # -------------------------------
    # Desenho dos nós
    # -------------------------------
    node_colors = []
    for node in G.nodes():
        if node in centers:
            node_colors.append("#ffcc66")  # amarelo (centro)
        else:
            node_colors.append("#1f78b4")  # azul (demanda)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=160,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.5,
    )

    # -------------------------------
    # Arestas de cobertura primária (verde)
    # -------------------------------
    green_edges = []
    green_labels = {}
    for i, j in assign_primary.items():
        if j is not None:
            green_edges.append((i, j))
            green_labels[(i, j)] = f"{dist_dict[i][j]:.1f}"

    if green_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=green_edges,
            edge_color="lightgreen",
            width=2.2,
            alpha=0.9,
        )
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=green_labels,
            font_size=6
        )

    # -------------------------------
    # Arestas de backup (lilás claro)
    # -------------------------------
    purple_edges = []
    purple_labels = {}
    for i, j in assign_backup.items():
        if j is not None:
            purple_edges.append((i, j))
            purple_labels[(i, j)] = f"{dist_dict[i][j]:.1f}"

    if purple_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=purple_edges,
            edge_color="#d8b7ff",
            width=2.2,
            alpha=0.95,
        )
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=purple_labels,
            font_size=6
        )

    # -------------------------------
    # Arestas cinzas → nós sem cobertura
    # -------------------------------
    all_nodes = set(range(1, n_vertices + 1))
    assigned_primary_nodes = {i for i, j in assign_primary.items() if j is not None}
    assigned_backup_nodes = {i for i, j in assign_backup.items() if j is not None}
    uncovered_nodes = sorted(all_nodes - assigned_primary_nodes - assigned_backup_nodes)

    gray_edges = []
    gray_labels = {}

    if centers:
        for i in uncovered_nodes:
            best_center = min(centers, key=lambda j: dist_dict[i][j])
            best_dist = dist_dict[i][best_center]
            gray_edges.append((i, best_center))
            gray_labels[(i, best_center)] = f"{best_dist:.1f}"

    if gray_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=gray_edges,
            edge_color="gray",
            width=1.4,
            alpha=0.7,
            style="dashed"
        )
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=gray_labels,
            font_size=6
        )

    # -------------------------------
    # Rótulos dos nós
    # -------------------------------
    nx.draw_networkx_labels(G, pos, font_size=7)

    # -------------------------------
    # LEGENDA
    # -------------------------------

    # Nome da instância no topo
    plt.text(
        0.5, 1.04,                # posição acima do gráfico
        f"Instância: {instance_name}",
        transform=plt.gca().transAxes,
        fontsize=12,
        fontweight="bold",
        ha="center",
    )


    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Backup graph saved to: {output_path}")
