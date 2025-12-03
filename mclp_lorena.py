import random
import numpy as np
import os
import time
import argparse
import sys
from datetime import datetime
import matplotlib.pyplot as plt

"""
python -m "mclp_lorena" --arquivo sjc_coords.txt --demanda sjc_demand.txt --p 10 --max-iter 100
"""


# =====================================================================
#   1. CARREGAMENTO DE DADOS (FORMATO ESPECÍFICO LORENA/SJC)
# =====================================================================

def load_lorena_instance(coord_file, demand_file):
    """
    Lê instâncias formato Lorena (SJC/Mcover) baseado nas imagens.
    
    Arquivo Coord:
      Linha 1: NNodes 9999 P Radius (Cabeçalho)
      Linhas seg: X Y (Indexação implícita pela ordem da linha)
      
    Arquivo Demanda:
      Linhas: Demanda (Indexação implícita pela ordem da linha)
    """
    print(f"[Loader] Lendo coordenadas: {coord_file}")
    
    coords = []  # Lista de tuplas (x, y)
    demands = [] # Lista de floats
    
    # --- 1. Leitura das Coordenadas ---
    try:
        with open(coord_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            
        if not lines:
            raise ValueError("Arquivo de coordenadas vazio.")

        # Parse do Cabeçalho (Linha 0)
        # Ex: "324 9999 10 500"
        header_parts = lines[0].split()
        expected_nodes = int(header_parts[0])
        print(f"[Loader] Cabeçalho diz: N={expected_nodes} nós.")

        # Parse do Corpo (Linhas 1 até o fim)
        for line_idx, line in enumerate(lines[1:], start=2):
            parts = line.split()
            # O formato é apenas "X Y"
            if len(parts) >= 2:
                coords.append((float(parts[0]), float(parts[1])))
            else:
                print(f"[Aviso] Linha {line_idx} ignorada (formato inválido): {line}")

    except Exception as e:
        raise ValueError(f"Erro fatal ao ler coordenadas: {e}")

    # --- 2. Leitura das Demandas ---
    print(f"[Loader] Lendo demandas: {demand_file}")
    try:
        with open(demand_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            
        for line in lines:
            parts = line.split()
            # O formato é apenas "Demanda" (um número por linha)
            if len(parts) >= 1:
                # Pega o último valor da linha para garantir (caso haja ID sujo)
                demands.append(float(parts[-1]))

    except Exception as e:
        raise ValueError(f"Erro fatal ao ler demandas: {e}")

    # --- 3. Validação e Sincronização ---
    num_nodes = len(coords)
    num_demands = len(demands)
    
    print(f"[Loader] Lidos: {num_nodes} Coordenadas, {num_demands} Demandas.")

    if num_nodes == 0:
        raise ValueError("Nenhuma coordenada válida encontrada.")

    # Ajusta tamanho das demandas se necessário
    if num_demands < num_nodes:
        print(f"[Aviso] Faltam demandas ({num_demands} vs {num_nodes}). Preenchendo com 1.0.")
        demands.extend([1.0] * (num_nodes - num_demands))
    elif num_demands > num_nodes:
        print(f"[Aviso] Sobram demandas. Truncando para {num_nodes}.")
        demands = demands[:num_nodes]

    # Conversão para Numpy
    coords_arr = np.array(coords, dtype=np.float32)
    pop_arr = np.array(demands, dtype=np.float32)

    print(f"[Loader] Calculando Matriz Euclidiana ({num_nodes}x{num_nodes})...")
    
    # 4. Matriz de Distância Euclidiana (Vetorizada)
    # dist[i,j] = sqrt((xi-xj)^2 + (yi-yj)^2)
    delta = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(delta**2, axis=-1))

    return num_nodes, pop_arr, dist_matrix, coords_arr

# =====================================================================
#   2. VISUALIZAÇÃO CARTESIANA
# =====================================================================

def plot_mapa_cartesiano(iter_count, current_s, dist_matrix, coords_arr, output_dir):
    """
    Plota as facilidades e alocações em um plano cartesiano 2D.
    """
    n_nodes = len(coords_arr)
    
    # Alocação (Vizinho mais próximo)
    assignments = []
    # Otimização: Fazer isso vetorizado seria melhor, mas loop simples é seguro aqui
    for i in range(n_nodes):
        dists = dist_matrix[i, current_s]
        idx_min = np.argmin(dists)
        best_facility = current_s[idx_min]
        assignments.append((i, best_facility))

    plt.figure(figsize=(10, 8))
    
    # 1. Nós de Demanda (Cinza)
    plt.scatter(coords_arr[:, 0], coords_arr[:, 1], c='lightgray', s=20, label='Clientes', edgecolors='none', alpha=0.8)
    
    # 2. Linhas de Conexão
    for cliente, facilidade in assignments:
        p1 = coords_arr[cliente]
        p2 = coords_arr[facilidade]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c='skyblue', alpha=0.2, linewidth=0.5, zorder=1)

    # 3. Facilidades Escolhidas (Estrelas Vermelhas)
    fac_coords = coords_arr[current_s]
    plt.scatter(fac_coords[:, 0], fac_coords[:, 1], c='red', s=120, marker='*', label='Mediana', zorder=10, edgecolors='black')

    plt.title(f"Iteração {iter_count:03d}")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.3)
    
    nome_arquivo = os.path.join(output_dir, f"mapa_iter_{iter_count:03d}.png")
    plt.savefig(nome_arquivo, dpi=100)
    plt.close()

def plot_convergencia(history_best, history_current, output_dir):
    plt.figure(figsize=(8, 4))
    iters = range(1, len(history_current) + 1)
    plt.plot(iters, history_current, label='Custo Atual', color='blue', alpha=0.3)
    plt.plot(iters, history_best, label='Melhor Global', color='green', linewidth=2)
    plt.xlabel('Iteração')
    plt.ylabel('Custo FO')
    plt.title('Convergência VNS')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "convergencia_vns.png"))
    plt.close()

# =====================================================================
#   3. LÓGICA VNS (CORE)
# =====================================================================

def evaluate_batch(solutions_batch, dist_matrix, populacao_arr, alpha):
    # solutions_batch shape: (Batch_Size, P)
    # Seleciona as colunas correspondentes às facilidades na matriz de distâncias
    # Result shape: (Batch_Size, P, N_Nodes)
    dists_batch = dist_matrix[solutions_batch, :]

    if alpha == 1:
        # P-Mediana Clássica: menor distância para qualquer facilidade
        min_dists = np.min(dists_batch, axis=1) # (Batch, N_Nodes)
        costs = np.dot(min_dists, populacao_arr) # (Batch,)
        return costs
    else:
        # Alpha-Neighbor: soma das distâncias aos alpha vizinhos mais próximos
        if alpha < dists_batch.shape[1]:
            partitioned = np.partition(dists_batch, alpha-1, axis=1)
            top_alpha = partitioned[:, :alpha, :]
            sum_dists = np.sum(top_alpha, axis=1)
        else:
            sum_dists = np.sum(dists_batch, axis=1)
        
        costs = np.dot(sum_dists, populacao_arr)
        return costs

def evaluate_single(solution, dist_matrix, populacao_arr, alpha):
    sol_batch = np.array([solution], dtype=int)
    return evaluate_batch(sol_batch, dist_matrix, populacao_arr, alpha)[0]

def neighborhood_k_exchange_random(solution, candidates_range, k):
    s_prime = list(solution)
    abertas = set(s_prime)
    fechadas = [i for i in candidates_range if i not in abertas]
    
    if len(abertas) < k or len(fechadas) < k: return s_prime

    for _ in range(k):
        if not abertas or not fechadas: break
        i_fechar = random.choice(list(abertas))
        abertas.remove(i_fechar)
        s_prime.remove(i_fechar)
        i_abrir = random.choice(fechadas)
        fechadas.remove(i_abrir)
        s_prime.append(i_abrir)
    return s_prime

def local_search_batch(solution, dist_matrix, populacao_arr, alpha, num_regioes):
    s_best = list(solution)
    cost_best = evaluate_single(s_best, dist_matrix, populacao_arr, alpha)
    BATCH_SIZE = 500
    
    while True:
        improved = False
        abertas = list(s_best)
        set_abertas = set(s_best)
        fechadas = [i for i in range(num_regioes) if i not in set_abertas]
        
        neighbors_buffer = []
        
        for i_fechar in abertas:
            for i_abrir in fechadas:
                neighbor = [x for x in s_best if x != i_fechar] + [i_abrir]
                neighbors_buffer.append(neighbor)
                
                if len(neighbors_buffer) >= BATCH_SIZE:
                    batch_arr = np.array(neighbors_buffer, dtype=int)
                    costs = evaluate_batch(batch_arr, dist_matrix, populacao_arr, alpha)
                    
                    min_idx = np.argmin(costs)
                    if costs[min_idx] < cost_best:
                        cost_best = costs[min_idx]
                        s_best = neighbors_buffer[min_idx]
                        improved = True
                        break
                    neighbors_buffer = []
            if improved: break
        
        # Resquício do buffer
        if not improved and neighbors_buffer:
            batch_arr = np.array(neighbors_buffer, dtype=int)
            costs = evaluate_batch(batch_arr, dist_matrix, populacao_arr, alpha)
            min_idx = np.argmin(costs)
            if costs[min_idx] < cost_best:
                cost_best = costs[min_idx]
                s_best = neighbors_buffer[min_idx]
                improved = True
        
        if not improved: break
            
    return s_best, cost_best

def vns_solve(dist_matrix, populacao_arr, coords_arr, p, alpha, max_iter, k_max, output_dir):
    num_nodes = len(populacao_arr)
    candidates_range = range(num_nodes)
    
    print(f"\n--- VNS INICIADO ---")
    print(f"Nodes: {num_nodes} | P: {p} | Alpha: {alpha}")
    
    # Validação Crítica
    if num_nodes < p:
        print(f"ERRO: Número de nós ({num_nodes}) menor que P ({p}).")
        return [], 0

    s_current = random.sample(candidates_range, p)
    cost_current = evaluate_single(s_current, dist_matrix, populacao_arr, alpha)
    
    s_best = list(s_current)
    cost_best = cost_current
    
    history_current = [cost_current]
    history_best = [cost_best]
    
    print(f"Custo Inicial: {cost_current:,.2f}")
    if coords_arr is not None:
        plot_mapa_cartesiano(0, s_current, dist_matrix, coords_arr, output_dir)
    
    filename = os.path.join(output_dir, "relatorio_execucao.txt")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"VNS Lorena | P={p} | Alpha={alpha}\n")
        start_time = time.time()
        
        for iter_count in range(1, max_iter + 1):
            iter_start = time.time()
            k = 1
            while k <= k_max:
                s_prime = neighborhood_k_exchange_random(s_current, candidates_range, k)
                s_double, cost_double = local_search_batch(s_prime, dist_matrix, populacao_arr, alpha, num_nodes)
                
                if cost_double < cost_current:
                    s_current = s_double
                    cost_current = cost_double
                    if cost_current < cost_best:
                        s_best = list(s_current)
                        cost_best = cost_current
                        print(f"Iter {iter_count} (k={k}): NOVO MELHOR: {cost_best:,.2f}")
                    k = 1
                    iter_count += 1
                else:
                    iter_count += 1
                    k += 1
            
            duration = time.time() - iter_start
            history_current.append(cost_current)
            history_best.append(cost_best)
            
            if iter_count % 10 == 0 or iter_count == 1:
                print(f"Iter {iter_count}/{max_iter} | Best: {cost_best:,.2f} | Time: {duration:.2f}s")
                if coords_arr is not None:
                    plot_mapa_cartesiano(iter_count, s_current, dist_matrix, coords_arr, output_dir)

    print(f"\nTempo Total: {time.time() - start_time:.2f}s")
    plot_convergencia(history_best, history_current, output_dir)
    return s_best, cost_best

# =====================================================================
#   4. MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="VNS para Instâncias Lorena (Cartesianas).")
    parser.add_argument("--arquivo", required=True, help="Arquivo de Coordenadas (N 9999 P S + X Y)")
    parser.add_argument("--demanda", required=True, help="Arquivo de Demanda (Apenas valores)")
    parser.add_argument("--p", type=int, default=10, help="Número de medianas")
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--k-max", type=int, default=3)
    
    args = parser.parse_args()
    
    # Cria diretório
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.normpath(os.path.join("relatorios_lorena", f"run_{timestamp}"))
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Carrega dados
        num_nodes, pop_arr, dist_matrix, coords_arr = load_lorena_instance(args.arquivo, args.demanda)
        
        # Executa
        vns_solve(
            dist_matrix=dist_matrix, 
            populacao_arr=pop_arr, 
            coords_arr=coords_arr, 
            p=args.p, 
            alpha=args.alpha, 
            max_iter=args.max_iter, 
            k_max=args.k_max, 
            output_dir=output_dir
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[ERRO FATAL] O programa parou: {e}")

if __name__ == "__main__":
    main()