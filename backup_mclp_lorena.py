import random
import numpy as np
import os
import time
import argparse
import sys
from datetime import datetime
import matplotlib.pyplot as plt

"""
python -m "backup_mclp_lorena" --arquivo "data_lorena/coord324.txt" --demanda "data_lorena/demanda324.txt" --p 10 --max-iter 30 --radius 260 --beta 0.3
"""

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
#   2. VISUALIZAÇÃO COM BACKUP
# =====================================================================

def plot_mapa_mclp_backup(iter_count, current_s, dist_matrix, coords_arr, radius, beta, output_dir):
    """
    Plota mapa distinguindo cobertura primária e secundária.
    """
    try:
        # Calcula quantas facilidades cobrem cada nó
        dists = dist_matrix[:, current_s] # (N, P)
        covered_counts = np.sum(dists <= radius, axis=1) # (N,) conta quantos <= S
        
        mask_0 = covered_counts == 0
        mask_1 = covered_counts == 1
        mask_2plus = covered_counts >= 2
        
        plt.figure(figsize=(10, 8))
        
        # Plot 0: Não Coberto (Cinza/X)
        plt.scatter(coords_arr[mask_0, 0], coords_arr[mask_0, 1], 
                    c='lightgray', s=20, label='Não Coberto', marker='x', alpha=0.5)
        
        # Plot 1: Cobertura Simples (Amarelo)
        plt.scatter(coords_arr[mask_1, 0], coords_arr[mask_1, 1], 
                    c='orange', s=35, label='Cobertura Simples (1)', alpha=0.8)
        
        # Plot 2+: Cobertura com Backup (Verde)
        label_backup = 'Cobertura Backup (2+)' if beta > 0 else 'Cobertura Múltipla'
        plt.scatter(coords_arr[mask_2plus, 0], coords_arr[mask_2plus, 1], 
                    c='green', s=35, label=label_backup, alpha=0.8)

        # Facilidades
        fac_coords = coords_arr[current_s]
        plt.scatter(fac_coords[:, 0], fac_coords[:, 1], c='red', s=150, marker='*', 
                    label='Instalação', zorder=10, edgecolors='black')
        
        plt.title(f"Iter {iter_count:03d} | Raio={radius} | Beta={beta}")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.3)
        plt.axis('equal')
        
        nome_arquivo = os.path.join(output_dir, f"mclp_beta_iter_{iter_count:03d}.png")
        plt.savefig(nome_arquivo, dpi=100)
        plt.close()
        print(f"[IO] Imagem salva: {os.path.basename(nome_arquivo)}")
    except Exception as e:
        print(f"[Aviso] Erro ao plotar imagem: {e}")

def plot_convergencia(history_best, output_dir):
    try:
        plt.figure(figsize=(8, 4))
        # Como estamos minimizando o Negativo, o gráfico mostra valores negativos
        # Vamos inverter para mostrar "Score Positivo"
        scores_positivos = [-h for h in history_best]
        plt.plot(scores_positivos, label='Score Cobertura (Max)', color='blue')
        plt.xlabel('Iteração')
        plt.ylabel('Score Ponderado (Pop + Beta*Pop)')
        plt.title('Evolução da Cobertura com Backup')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "convergencia_mclp.png"))
        plt.close()
    except Exception as e:
        print(f"[Aviso] Erro ao plotar convergência: {e}")

# =====================================================================
#   3. VNS MCLP - BACKUP (FO MODIFICADA)
# =====================================================================

def evaluate_backup_mclp(solution, dist_matrix, populacao_arr, radius, beta):
    """
    Calcula o score de cobertura considerando backup.
    Retorna NEGATIVO para minimização.
    """
    s_arr = np.array(solution)
    # Distâncias de todos os nós para as facilidades escolhidas
    # Shape: (P, N) -> Transpomos para (N, P) se necessário, mas numpy resolve bem
    dists_subset = dist_matrix[:, s_arr] # Shape (N, P)

    # Precisamos das 2 menores distâncias para cada nó
    # np.partition é mais rápido que sort. kth=1 garante que os índices 0 e 1 tenham os menores
    if dists_subset.shape[1] >= 2:
        part = np.partition(dists_subset, 1, axis=1)
        d1 = part[:, 0] # Menor distância
        d2 = part[:, 1] # Segunda menor distância
    else:
        # Se P=1, não existe backup
        d1 = dists_subset[:, 0]
        d2 = np.full_like(d1, np.inf)

    # Lógica de Pontuação
    # 1. Cobertura Primária (Peso 1)
    mask_primary = d1 <= radius
    score_primary = np.sum(populacao_arr[mask_primary])

    # 2. Cobertura Secundária (Peso Beta)
    score_backup = 0
    if beta > 0:
        mask_backup = d2 <= radius
        score_backup = np.sum(populacao_arr[mask_backup]) * beta

    total_score = score_primary + score_backup
    
    # Retorna negativo pois VNS minimiza
    return -total_score

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

def local_search_best_improvement(solution, dist_matrix, populacao_arr, num_regioes, radius, beta):
    s_best = list(solution)
    cost_best = evaluate_backup_mclp(s_best, dist_matrix, populacao_arr, radius, beta)
    
    abertas = list(s_best)
    fechadas = [i for i in range(num_regioes) if i not in s_best]
    
    # Amostragem para performance
    if len(fechadas) > 50: 
        fechadas = random.sample(fechadas, 50)
        
    for i_fechar in abertas:
        for i_abrir in fechadas:
            neighbor = [x for x in s_best if x != i_fechar] + [i_abrir]
            cost_neighbor = evaluate_backup_mclp(neighbor, dist_matrix, populacao_arr, radius, beta)
            if cost_neighbor < cost_best:
                return neighbor, cost_neighbor
    return s_best, cost_best

def vns_solve(dist_matrix, populacao_arr, coords_arr, p, beta, max_iter, k_max, radius, output_dir):
    num_nodes = len(populacao_arr)
    candidates_range = range(num_nodes)
    total_demand = np.sum(populacao_arr)
    
    # Score máximo teórico (se tudo for coberto 2 vezes)
    max_score_possible = total_demand * (1 + beta)
    
    # Logging Setup
    log_file = os.path.join(output_dir, "relatorio_detalhado.txt")
    def log_dual(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"--- RELATÓRIO MCLP COM BACKUP ---\n")
        f.write(f"Parâmetros: P={p} | Raio={radius} | Beta={beta}\n")
        f.write(f"Demanda Base: {total_demand:,.2f}\n")
        f.write(f"Score Máximo Teórico: {max_score_possible:,.2f}\n")
        f.write("-" * 50 + "\n")

    log_dual(f"\n--- VNS MCLP (BACKUP BETA={beta}) ---")
    
    # Solução Inicial
    s_current = random.sample(candidates_range, p)
    cost_current = evaluate_backup_mclp(s_current, dist_matrix, populacao_arr, radius, beta)
    s_best = list(s_current)
    cost_best = cost_current
    
    history_best = [cost_best]
    start_time_global = time.time()
    
    score_ini = -cost_current
    log_dual(f"Iter 000 | Score: {score_ini:,.2f} ({(score_ini/max_score_possible)*100:.1f}% do max teórico)")
    
    if coords_arr is not None:
        plot_mapa_mclp_backup(0, s_current, dist_matrix, coords_arr, radius, beta, output_dir)
    
    # Loop VNS
    for iter_count in range(1, max_iter + 1):
        iter_start_time = time.time()
        k = 1
        
        while k <= k_max:
            # 1. Shaking
            s_prime = neighborhood_k_exchange_random(s_current, candidates_range, k)
            
            # 2. Local Search
            s_double, cost_double = local_search_best_improvement(s_prime, dist_matrix, populacao_arr, num_nodes, radius, beta)
            
            # 3. Neighborhood Change
            if cost_double < cost_current:
                s_current = s_double
                cost_current = cost_double
                if cost_current < cost_best:
                    s_best = list(s_current)
                    cost_best = cost_current
                    
                    score_abs = -cost_best
                    log_dual(f"  >> MELHORIA (k={k}): Score {score_abs:,.2f}")
                    
                k = 1 
            else:
                k += 1 
        
        history_best.append(cost_best)
        iter_duration = time.time() - iter_start_time
        
        # Log Periódico
        if iter_count % 5 == 0 or iter_count == max_iter:
            score_abs = -cost_best
            log_dual(f"Iter {iter_count:03d}/{max_iter} | Score: {score_abs:,.2f} | Time: {iter_duration:.4f}s")
            
            if coords_arr is not None:
                plot_mapa_mclp_backup(iter_count, s_current, dist_matrix, coords_arr, radius, beta, output_dir)

    # Finalização
    total_time = time.time() - start_time_global
    final_score = -cost_best
    
    log_dual("-" * 50)
    log_dual(f"Tempo Total: {total_time:.2f}s")
    log_dual(f"Melhor Score Ponderado: {final_score:,.2f}")
    log_dual(f"Facilidades Finais: {s_best}")
    
    plot_convergencia(history_best, output_dir)
    print(f"[IO] Log salvo em: {log_file}")
    
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
    # Novo Argumento Beta
    parser.add_argument("--beta", type=float, default=0.0, help="Importância da segunda cobertura (0.0 a 1.0)")
    parser.add_argument("--max-iter", type=int, default=30)
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.abspath(os.path.join("relatorios_mclp_backup", f"run_{timestamp}"))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- INICIANDO MCLP COM BACKUP ---")
    print(f"Beta: {args.beta}")
    print(f"Pasta de Saída: {output_dir}")
    
    try:
        num_nodes, pop_arr, dist_matrix, coords_arr, file_radius = load_lorena_instance(args.arquivo, args.demanda)
        
        final_radius = args.radius if args.radius > 0 else file_radius
        if final_radius <= 0:
            print("AVISO: Raio = 0! Use --radius X para definir.")
        
        vns_solve(dist_matrix, pop_arr, coords_arr, args.p, args.beta, args.max_iter, 3, final_radius, output_dir)
        
    except Exception as e:
        print(f"ERRO FATAL: {e}")

if __name__ == "__main__":
    main()