import numpy as np
from typing import List, Tuple, Dict

def read_pmed_file(path: str) -> Tuple[int, int, int, List[Tuple[int, int, float]]]:
    """
    Lê o arquivo no formato OR-Library (pmed*).
    
    Formato esperado:
        N_vertices N_arestas P
        u v custo
        ...
        
    Retorna:
        n_vertices, n_arestas, p, edges (lista de tuplas u, v, custo)
    """
    edges: List[Tuple[int, int, float]] = []

    with open(path, "r") as f:
        # Lê a primeira linha não vazia
        first = ""
        while not first:
            line = f.readline()
            if not line:
                raise ValueError(f"Arquivo {path} vazio ou inválido.")
            first = line.strip()

        # Parse do cabeçalho
        try:
            parts = first.split()
            n_vertices = int(parts[0])
            n_edges = int(parts[1])
            p = int(parts[2])
        except IndexError:
            raise ValueError("Cabeçalho do arquivo pmed inválido. Esperado: 'N M P'")

        # Parse das arestas
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = list(map(float, line.split()))
            # Formato: u v custo
            u, v, c = int(parts[0]), int(parts[1]), parts[2]
            edges.append((u, v, c))

    return n_vertices, n_edges, p, edges


def build_cost_matrix(n_vertices: int, edges: List[Tuple[int, int, float]]) -> List[List[float]]:
    """
    Monta a matriz de adjacência inicial (indices 1..N).
    Retorna matriz (N+1)x(N+1) com float('inf') onde não há conexão.
    """
    INF = float("inf")
    # Cria matriz (n+1)x(n+1) para facilitar o uso de índices baseados em 1 (comum na OR-Lib)
    cost = [[INF] * (n_vertices + 1) for _ in range(n_vertices + 1)]

    for i in range(1, n_vertices + 1):
        cost[i][i] = 0.0

    for u, v, c in edges:
        # Grafo não-direcionado: preenche simetricamente
        if c < cost[u][v]:
            cost[u][v] = c
            cost[v][u] = c

    return cost


def floyd_marshall(cost: List[List[float]]) -> List[List[float]]:
    """
    Executa o algoritmo de Floyd-Warshall para calcular o menor caminho entre todos os pares.
    Complexidade: O(N^3).
    """
    n = len(cost) - 1  # Desconsidera o índice 0
    INF = float("inf")

    # Cópia profunda da matriz de custos
    dist = [row[:] for row in cost]

    for k in range(1, n + 1):
        for i in range(1, n + 1):
            if dist[i][k] == INF:
                continue
            for j in range(1, n + 1):
                if dist[k][j] == INF:
                    continue
                
                alt = dist[i][k] + dist[k][j]
                if alt < dist[i][j]:
                    dist[i][j] = alt

    return dist


def load_pmed_instance(file_path: str) -> Tuple[int, List[str], Dict[str, float], np.ndarray]:
    """
    Função Adaptadora (Wrapper):
    1. Lê o arquivo pmed cru.
    2. Calcula matriz de distâncias completa via Floyd-Warshall.
    3. Converte os dados para o formato esperado pelo VNS (Numpy 0-based).
    
    Retorna:
        P (int): Número de facilidades a abrir.
        REGIOES (List[str]): Lista de IDs das regiões ['1', '2', ...].
        POPULACAO (Dict): Dicionário {'ID': peso}, assumindo peso 1.0.
        DISTANCIA (np.array): Matriz NxN numpy (índices 0 a N-1).
    """
    print(f"[Loader] Lendo arquivo: {file_path}...")
    n_vertices, n_edges, p, edges = read_pmed_file(file_path)
    
    print(f"[Loader] Grafo carregado: {n_vertices} nós, {n_edges} arestas. P={p}")
    print("[Loader] Calculando matriz de custos iniciais...")
    cost_matrix = build_cost_matrix(n_vertices, edges)
    
    print("[Loader] Executando Floyd-Warshall (Calculando todas as distâncias)...")
    dist_full = floyd_marshall(cost_matrix)
    
    # --- Pós-processamento para formato VNS ---
    
    # 1. Converter Matriz (N+1)x(N+1) para Numpy NxN (0-based)
    # Cortamos a linha 0 e a coluna 0 que eram placeholders
    dist_list = []
    for i in range(1, n_vertices + 1):
        row = dist_full[i][1:] 
        dist_list.append(row)
    
    DISTANCIA = np.array(dist_list)
    
    # 2. Criar lista de Regiões (Strings '1' a 'N')
    REGIOES = [str(i) for i in range(1, n_vertices + 1)]
    
    # 3. Criar População Dummy (Peso 1.0 para todos, padrão p-mediana clássico)
    POPULACAO = {r: 1.0 for r in REGIOES}
    
    print("[Loader] Instância carregada com sucesso.")
    return p, REGIOES, POPULACAO, DISTANCIA