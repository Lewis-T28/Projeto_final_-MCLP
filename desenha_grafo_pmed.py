import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt


def ler_instancia_pmed(caminho_arquivo):
    """
    Lê o arquivo no formato:
    n_vertices n_arestas p
    u v custo
    ...
    Retorna: grafo (networkx.Graph), p
    """
    G = nx.Graph()

    with open(caminho_arquivo, "r") as f:
        primeira_linha = f.readline().strip()
        if not primeira_linha:
            raise ValueError("Arquivo vazio ou formato inválido.")

        n_vertices, n_arestas, p = map(int, primeira_linha.split())
        G.add_nodes_from(range(1, n_vertices + 1))

        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            partes = linha.split()
            if len(partes) < 3:
                continue
            u, v, custo = map(int, partes)
            G.add_edge(u, v, weight=custo)

    return G, p


def desenhar_grafo(G, caminho_imagem="grafo.png", mostrar_valores=False):
    """
    Desenha o grafo G e salva em caminho_imagem.
    Use mostrar_valores=True para exibir custos das arestas + índices dos vértices.
    """

    # Layout recomendável para redes de 100 nós
    pos = nx.kamada_kawai_layout(G)

    plt.figure(figsize=(12, 12))

    nx.draw_networkx_nodes(
        G, pos,
        node_size=150,
        node_color="#1f78b4",
        edgecolors="black",
        linewidths=0.4,
    )

    nx.draw_networkx_edges(
        G, pos,
        width=0.6,
        alpha=0.4
    )

    # Rótulos dos vértices
    nx.draw_networkx_labels(
        G, pos,
        font_size=7,
        font_color="black"
    )

    # Rótulos das arestas (custos) — apenas quando solicitado
    if mostrar_valores:
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(
            G, pos, 
            edge_labels=edge_labels,
            font_size=5,
            label_pos=0.5
        )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(caminho_imagem, dpi=300)
    plt.close()
    print(f"Imagem salva em: {caminho_imagem}")


def main():
    parser = argparse.ArgumentParser(description="Desenha grafo PMED.")
    parser.add_argument("arquivo", help="Arquivo de entrada no formato PMED")
    parser.add_argument("saida", nargs="?", default="grafo.png",
                        help="Imagem de saída (default: grafo.png)")
    parser.add_argument("--values", "-v", action="store_true",
                        help="Mostra valores das arestas e rótulos dos vértices")

    args = parser.parse_args()

    G, p = ler_instancia_pmed(args.arquivo)
    print(f"Lido: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas, p = {p}")

    desenhar_grafo(G, caminho_imagem=args.saida, mostrar_valores=args.values)


if __name__ == "__main__":
    main()
