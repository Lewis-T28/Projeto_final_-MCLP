import random
import numpy as np
import os
import time
import argparse
import sys
from datetime import datetime
import matplotlib.pyplot as plt

# =====================================================================
#   1. CARREGAMENTO (MCLP LORENA)
# =====================================================================

def load_lorena_instance(coord_file, demand_file):
    print(f"[Loader] Lendo coordenadas: {coord_file}")
    coords = [] 
    demands = []
    radius_s = 0.0
    
    # 1. Coordenadas
    try:
        with open(coord_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if not lines: raise ValueError("Arquivo de coordenadas vazio.")

        header_parts = lines[0].split()
        expected_nodes = int(header_parts[0])
        
        if len(header_parts) >= 4:
            radius_s = float(header_parts[3])
            print(f"[Loader] RAIO DE COBERTURA DETECTADO NO ARQUIVO: S = {radius_s}")
        
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 2:
                coords.append((float(parts[0]), float(parts[1])))
    except Exception as e:
        raise ValueError(f"Erro coords: {e}")

    # 2. Demandas
    print(f"[Loader] Lendo demandas: {demand_file}")
    try:
        with open(demand_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) >= 1:
                    demands.append(float(parts[-1]))
    except Exception as e:
        raise ValueError(f"Erro demandas: {e}")

    num_nodes = len(coords)
    if len(demands) > num_nodes: demands = demands[:num_nodes]
    elif len(demands) < num_nodes: demands.extend([1.0]*(num_nodes - len(demands)))

    coords_arr = np.array(coords, dtype=np.float32)
    pop_arr = np.array(demands, dtype=np.float32)

    print(f"[Loader] Calculando Matriz Euclidiana ({num_nodes}x{num_nodes})...")
    delta = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(delta**2, axis=-1))

    return num_nodes, pop_arr, dist_matrix, coords_arr, radius_s

# =====================================================================
#   2. VISUALIZAÇÃO
# =====================================================================

def plot_mapa_mclp(iter_count, current_s, dist_matrix, coords_arr, radius, output_dir):
    """
    Salva o mapa. Usa print() normal para NÃO sair no relatório de texto.
    """
    try:
        min_dists = np.min(dist_matrix[:, current_s], axis=1)
        covered_mask = min_dists <= radius
        
        plt.figure(figsize=(10, 8))
        
        plt.scatter(coords_arr[covered_mask, 0], coords_arr[covered_mask, 1], 
                    c='lightgreen', s=30, label='Coberto', alpha=0.7)
        plt.scatter(coords_arr[~covered_mask, 0], coords_arr[~covered_mask, 1], 
                    c='gray', s=15, label='Não Coberto', alpha=0.3, marker='x')

        fac_coords = coords_arr[current_s]
        plt.scatter(fac_coords[:, 0], fac_coords[:, 1], c='red', s=150, marker='*', 
                    label='Instalação', zorder=10, edgecolors='black')
        
        plt.title(f"Iter {iter_count:03d} | Raio={radius}")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.3)
        plt.axis('equal')
        
        nome_arquivo = os.path.join(output_dir, f"mclp_iter_{iter_count:03d}.png")
        plt.savefig(nome_arquivo, dpi=100)
        plt.close()
        # O usuário pediu para não salvar este log no txt, então usamos print normal
        print(f"[IO] Imagem salva: {os.path.basename(nome_arquivo)}")
    except Exception as e:
        print(f"[Aviso] Erro ao plotar imagem: {e}")

def plot_convergencia(history_best, output_dir):
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(history_best, label='Demanda NÃO Coberta (Minimizar)', color='red')
        plt.xlabel('Iteração')
        plt.ylabel('Custo')
        plt.title('Convergência MCLP')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "convergencia_mclp.png"))
        plt.close()
    except Exception as e:
        print(f"[Aviso] Erro ao plotar convergência: {e}")

# =====================================================================
#   3. VNS MCLP (COM LOGGING ATUALIZADO)
# =====================================================================

def evaluate_single_mclp(solution, dist_matrix, populacao_arr, alpha, radius):
    sol_batch = np.array([solution], dtype=int)
    dists_batch = dist_matrix[sol_batch, :]
    
    if alpha == 1:
        min_dists = np.min(dists_batch, axis=1)
        uncovered_mask = min_dists > radius
    else:
        if alpha < dists_batch.shape[1]:
            partitioned = np.partition(dists_batch, alpha-1, axis=1)
            alpha_dists = partitioned[:, alpha-1, :]
            uncovered_mask = alpha_dists > radius
        else:
            uncovered_mask = np.ones((1, dists_batch.shape[2]), dtype=bool)

    return np.dot(uncovered_mask, populacao_arr)[0]

def neighborhood_k_exchange_random(solution, candidates_range, k):
    s_prime = list(solution)
    abertas = set(s_prime)
    fechadas = [i for i in candidates_range if i not in abertas]
    effective_k = min(k, len(abertas), len(fechadas))
    if effective_k == 0: return s_prime

    for _ in range(effective_k):
        i_fechar = random.choice(list(abertas))
        abertas.remove(i_fechar); s_prime.remove(i_fechar)
        i_abrir = random.choice(fechadas)
        fechadas.remove(i_abrir); s_prime.append(i_abrir)
    return s_prime

def local_search_best_improvement(solution, dist_matrix, populacao_arr, alpha, num_regioes, radius):
    s_best = list(solution)
    cost_best = evaluate_single_mclp(s_best, dist_matrix, populacao_arr, alpha, radius)
    abertas = list(s_best)
    fechadas = [i for i in range(num_regioes) if i not in s_best]
    
    if len(fechadas) > 50: 
        fechadas = random.sample(fechadas, 50)
        
    for i_fechar in abertas:
        for i_abrir in fechadas:
            neighbor = [x for x in s_best if x != i_fechar] + [i_abrir]
            cost_neighbor = evaluate_single_mclp(neighbor, dist_matrix, populacao_arr, alpha, radius)
            if cost_neighbor < cost_best:
                return neighbor, cost_neighbor
    return s_best, cost_best

def vns_solve(dist_matrix, populacao_arr, coords_arr, p, alpha, max_iter, k_max, radius, output_dir):
    num_nodes = len(populacao_arr)
    candidates_range = range(num_nodes)
    total_demand = np.sum(populacao_arr)
    
    # ---------------------------------------------------------
    # CONFIGURAÇÃO DE LOGGING
    # ---------------------------------------------------------
    log_file = os.path.join(output_dir, "relatorio_detalhado.txt")
    
    # Função auxiliar para printar na tela E salvar no arquivo
    def log_dual(msg):
        print(msg) # Tela
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n") # Arquivo

    # Cabeçalho do Log
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"--- RELATÓRIO DE EXECUÇÃO MCLP ---\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parâmetros: P={p} | Raio={radius} | Nós={num_nodes} | Alpha={alpha}\n")
        f.write(f"Demanda Total: {total_demand:,.2f}\n")
        f.write("-" * 50 + "\n")

    log_dual(f"\n--- VNS MCLP INICIADO ---")
    log_dual(f"Nodes: {num_nodes} | P: {p} | Radius: {radius}")
    
    # Solução Inicial
    s_current = random.sample(candidates_range, p)
    cost_current = evaluate_single_mclp(s_current, dist_matrix, populacao_arr, alpha, radius)
    s_best = list(s_current)
    cost_best = cost_current
    
    history_best = [cost_best]
    start_time_global = time.time()
    
    # Iteração 0
    cob_percent = ((total_demand - cost_current)/total_demand)*100
    log_dual(f"Iter 000 | Uncovered: {cost_current:,.0f} | Coverage: {cob_percent:.1f}% | Inicial")
    
    if coords_arr is not None:
        plot_mapa_mclp(0, s_current, dist_matrix, coords_arr, radius, output_dir)
    
    # --- LOOP VNS ---
    for iter_count in range(1, max_iter + 1):
        iter_start_time = time.time() # Início do cronômetro da iteração
        k = 1
        
        while k <= k_max:
            s_prime = neighborhood_k_exchange_random(s_current, candidates_range, k)
            s_double, cost_double = local_search_best_improvement(s_prime, dist_matrix, populacao_arr, alpha, num_nodes, radius)
            
            if cost_double < cost_current:
                s_current = s_double
                cost_current = cost_double
                if cost_current < cost_best:
                    s_best = list(s_current)
                    cost_best = cost_current
                    
                    # LOG DE MELHORIA NO ARQUIVO
                    cob_abs = total_demand - cost_best
                    log_dual(f"  >> MELHORIA (k={k}): Cobrindo {cob_abs:,.0f} (Nodes: {s_best})")
                    
                k = 1 
            else:
                k += 1 
        
        history_best.append(cost_best)
        
        # Fim do cronômetro da iteração
        iter_duration = time.time() - iter_start_time
        
        # LOG PERIÓDICO NO ARQUIVO (com Tempo)
        if iter_count % 100 == 0 or iter_count == max_iter or iter_count == 1 or iter_count== 5 or iter_count == 20:
            cob_percent = ((total_demand - cost_best)/total_demand)*100
            # Adicionado campo Time
            log_dual(f"Iter {iter_count:03d}/{max_iter} | Uncovered: {cost_best:,.0f} | Cov: {cob_percent:.1f}% | Time: {iter_duration:.4f}s")
            
            if coords_arr is not None:
                plot_mapa_mclp(iter_count, s_current, dist_matrix, coords_arr, radius, output_dir)

    # --- FINALIZAÇÃO ---
    total_time = time.time() - start_time_global
    
    log_dual("-" * 50)
    log_dual(f"Tempo Total: {total_time:.2f}s")
    log_dual(f"Melhor Cobertura: {total_demand - cost_best:,.2f} ({((total_demand - cost_best)/total_demand)*100:.2f}%)")
    log_dual(f"Facilidades Finais: {s_best}")
    
    plot_convergencia(history_best, output_dir)
    print(f"[IO] Log completo salvo em: {log_file}")
    
    return s_best, cost_best

# =====================================================================
#   MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arquivo", required=True)
    parser.add_argument("--demanda", required=True)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--radius", type=float, default=0)
    parser.add_argument("--max-iter", type=int, default=30)
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.abspath(os.path.join("relatorios_mclp", f"run_{timestamp}"))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- INICIANDO ---")
    print(f"Pasta de Saída: {output_dir}")
    
    try:
        num_nodes, pop_arr, dist_matrix, coords_arr, file_radius = load_lorena_instance(args.arquivo, args.demanda)
        
        final_radius = args.radius if args.radius > 0 else file_radius
        if final_radius <= 0:
            print("AVISO: Raio = 0! Use --radius X para definir.")
        
        vns_solve(dist_matrix, pop_arr, coords_arr, args.p, 1, args.max_iter, 3, final_radius, output_dir)
        
    except Exception as e:
        print(f"ERRO FATAL: {e}")

if __name__ == "__main__":
    main()