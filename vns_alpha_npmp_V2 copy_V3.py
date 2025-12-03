import random
import numpy as np
import os
import time
import argparse
import sys
from datetime import datetime

INSTANCIA = f"pmed1.txt"
# =====================================================================
#   CONFIGURAÇÕES E IMPORTAÇÕES
# =====================================================================

sys.path.append(os.getcwd())
if os.path.exists(os.path.join(os.getcwd(), 'src')):
    sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from src.pmed_loader_to_vns_alpha_npmp import load_pmed_instance
    USING_MOCK = False
except ImportError:
    # Fallback para importação local ou Mock
    try:
        from pmed_loader_to_vns_alpha_npmp import load_pmed_instance
        USING_MOCK = False
    except ImportError:
        USING_MOCK = True

# Imports Opcionais (Plotagem)
HAS_EXTERNAL_PLOTS = False
try:
    from src.print_nodes import plot_solution
    from src.desenha_grafo_pmed import ler_instancia_pmed, desenhar_grafo
    HAS_EXTERNAL_PLOTS = True
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Dados Mock (SJC)
REGIOES_MOCK = ['Centro', 'Norte', 'Leste', 'Sudeste', 'Sul', 'Oeste', 'Rural', 'SFX']
POPULACAO_MOCK = {'Centro': 72401, 'Norte': 61940, 'Leste': 181463, 'Sudeste': 62541,
                  'Sul': 237572, 'Oeste': 64482, 'Rural': 15212, 'SFX': 1443}
DISTANCIA_MOCK = np.array([
    [0, 6, 8, 7, 10, 9, 25, 35], [6, 0, 12, 10, 14, 8, 20, 40],
    [8, 12, 0, 5, 15, 18, 30, 45], [7, 10, 5, 0, 12, 16, 28, 42],
    [10, 14, 15, 12, 0, 20, 22, 38], [9, 8, 18, 16, 20, 0, 15, 50],
    [25, 20, 30, 28, 22, 15, 0, 60], [35, 40, 45, 42, 38, 50, 60, 0]
], dtype=np.float32)

# =====================================================================
#   1. AVALIAÇÃO EM LOTE (HIPER-OTIMIZADA)
# =====================================================================

def evaluate_batch(solutions_batch, dist_matrix, populacao_arr, alpha):
    """
    Avalia múltiplos vizinhos SIMULTANEAMENTE usando matrizes 3D.
    Entrada: solutions_batch (Batch_Size, P)
    Saída: costs (Batch_Size,)
    """
    # 1. Indexação Fantasma 3D
    # Shape resultante: (Batch_Size, P, N_Nodes)
    dists_batch = dist_matrix[solutions_batch, :]

    if alpha == 1:
        # Pega a menor distância ao longo do eixo das facilidades (axis=1)
        # Result: (Batch_Size, N_Nodes)
        min_dists = np.min(dists_batch, axis=1)
        
        # Produto escalar com a população
        # Result: (Batch_Size,)
        costs = np.dot(min_dists, populacao_arr)
        return costs
    
    else:
        # Para Alpha > 1, usamos partition ou sort ao longo do eixo 1
        if alpha < dists_batch.shape[1]:
            partitioned = np.partition(dists_batch, alpha-1, axis=1)
            top_alpha = partitioned[:, :alpha, :]
            sum_dists = np.sum(top_alpha, axis=1)
        else:
            sum_dists = np.sum(dists_batch, axis=1)
            
        costs = np.dot(sum_dists, populacao_arr)
        return costs

def evaluate_single(solution, dist_matrix, populacao_arr, alpha):
    """Wrapper para avaliar uma única solução usando a lógica de batch."""
    sol_batch = np.array([solution], dtype=int)
    return evaluate_batch(sol_batch, dist_matrix, populacao_arr, alpha)[0]

# =====================================================================
#   2. BUSCA LOCAL OTIMIZADA (BATCH PROCESSING)
# =====================================================================

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
    """
    Busca Local que avalia vizinhos em pacotes (Chunks).
    Estratégia: First Improvement (Best Improvement dentro do Batch).
    """
    s_best = list(solution)
    cost_best = evaluate_single(s_best, dist_matrix, populacao_arr, alpha)
    
    # Tamanho do lote de avaliação
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
                
                # Quando o buffer enche, avaliamos o lote
                if len(neighbors_buffer) >= BATCH_SIZE:
                    batch_arr = np.array(neighbors_buffer, dtype=int)
                    costs = evaluate_batch(batch_arr, dist_matrix, populacao_arr, alpha)
                    
                    min_idx = np.argmin(costs)
                    min_cost_batch = costs[min_idx]
                    
                    if min_cost_batch < cost_best:
                        cost_best = min_cost_batch
                        s_best = neighbors_buffer[min_idx]
                        improved = True
                        break # First Improvement
                    
                    neighbors_buffer = []
            
            if improved: break 
        
        # Processa o restante do buffer
        if not improved and neighbors_buffer:
            batch_arr = np.array(neighbors_buffer, dtype=int)
            costs = evaluate_batch(batch_arr, dist_matrix, populacao_arr, alpha)
            min_idx = np.argmin(costs)
            
            if costs[min_idx] < cost_best:
                cost_best = costs[min_idx]
                s_best = neighbors_buffer[min_idx]
                improved = True
        
        if not improved:
            break
            
    return s_best, cost_best

# =====================================================================
#   3. VISUALIZAÇÃO (CORRIGIDA)
# =====================================================================

def gerar_visualizacao_personalizada(iter_count, current_s, dist_matrix, edges_list, n_vertices, output_dir):
    """Gera visualização do grafo salvando no diretório de execução."""
    if not HAS_EXTERNAL_PLOTS or edges_list is None: return
    
    centers_1based = [i + 1 for i in current_s]
    
    # Assignment simplificado para plot
    assignment = {}
    for i in range(n_vertices):
        dists = dist_matrix[i, current_s]
        idx_min = np.argmin(dists)
        assignment[i+1] = current_s[idx_min] + 1

    # CORREÇÃO CRÍTICA:
    # Usamos os.path.abspath para gerar o caminho completo (ex: C:\Users\...\relatorios\execucao\...)
    # Isso geralmente faz o os.path.join interno da função plot_solution ignorar a pasta padrão 'grafo_prints'
    caminho_relativo = os.path.join(output_dir, f"vns_iter_{iter_count:03d}")
    nome_arquivo_completo = os.path.abspath(caminho_relativo)
    
    try:
        dist_dict_sparse = {i+1: {} for i in range(n_vertices)}
        for i, center in assignment.items():
            if center:
                dist_dict_sparse[i][center] = float(dist_matrix[i-1, center-1])
        
        plot_solution(
            n_vertices=n_vertices,
            edges=edges_list,
            dist_dict=dist_dict_sparse,
            centers=centers_1based,
            assignment=assignment,
            filename=nome_arquivo_completo
        )
    except Exception as e:
        # Fallback de segurança: se o path absoluto falhar, tenta criar a estrutura dentro de grafo_prints
        print(f"[Aviso] Tentativa 1 de plot falhou: {e}. Tentando fallback...")
        try:
            # Se a lib insiste em usar grafo_prints, criamos a pasta lá dentro
            caminho_fallback = os.path.join("grafo_prints", output_dir)
            os.makedirs(caminho_fallback, exist_ok=True)
            
            # Tentamos passar o caminho relativo novamente agora que a pasta existe
            plot_solution(
                n_vertices=n_vertices,
                edges=edges_list,
                dist_dict=dist_dict_sparse,
                centers=centers_1based,
                assignment=assignment,
                filename=caminho_relativo
            )
        except Exception as e2:
            print(f"[Erro] Não foi possível salvar a imagem do grafo: {e2}")

def plot_convergencia(history_best, history_current, output_dir):
    """Plota gráfico de convergência no diretório de execução."""
    if not HAS_MATPLOTLIB: return
    plt.figure(figsize=(8, 4))
    iters = range(1, len(history_current) + 1)
    plt.plot(iters, history_current, label='Custo Atual', color='blue', alpha=0.3)
    plt.plot(iters, history_best, label='Melhor Global', color='green', linewidth=2)
    plt.xlabel('Iteração')
    plt.ylabel('Custo FO')
    plt.title('Evolução do VNS')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Salva no diretório específico
    plt.savefig(os.path.join(output_dir, "convergencia_vns.png"))
    plt.close()

# =====================================================================
#   4. VNS PRINCIPAL
# =====================================================================

def vns_solve(dist_matrix, populacao_arr, p, alpha, max_iter, k_max, regioes_names=None, edges_list=None, output_dir="relatorios"):
    num_nodes = len(populacao_arr)
    candidates_range = range(num_nodes)
    
    # Solução Inicial
    s_current = random.sample(candidates_range, p)
    cost_current = evaluate_single(s_current, dist_matrix, populacao_arr, alpha)
    
    s_best = list(s_current)
    cost_best = cost_current
    
    history_current = [cost_current]
    history_best = [cost_best]
    
    print(f"\n--- VNS OTIMIZADO (BATCH) INICIADO (P={p}, Alpha={alpha}) ---")
    print(f"Diretório de saída: {output_dir}")
    print(f"Custo Inicial: {cost_current:,.2f}")
    
    # Plot inicial
    if HAS_EXTERNAL_PLOTS and edges_list:
        gerar_visualizacao_personalizada(0, s_current, dist_matrix, edges_list, num_nodes, output_dir)
    
    # Arquivo de relatório dentro da pasta criada
    filename = os.path.join(output_dir, "relatorio_execucao.txt")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"VNS Optimized | P={p} | Alpha={alpha}\n")
        
        start_time = time.time()

        f.write(f"Custo inicial={cost_best} | Solução inicial={s_best}\n")
        f.write(f"Instancia={INSTANCIA}\n")
        
        for iter_count in range(1, max_iter + 1):
            iter_start = time.time()
            k = 1
            
            while k <= k_max:
                # a. Shaking
                s_prime = neighborhood_k_exchange_random(s_current, candidates_range, k)
                
                # b. Busca Local (BATCH OPTIMIZED)
                s_double, cost_double = local_search_batch(
                    s_prime, dist_matrix, populacao_arr, alpha, num_nodes
                )
                
                # c. Move
                if cost_double < cost_current:
                    s_current = s_double
                    cost_current = cost_double
                    
                    if cost_current < cost_best:
                        s_best = list(s_current)
                        cost_best = cost_current
                        print(f"Iter {iter_count} (k={k}): NOVO MELHOR: {cost_best:,.2f} | S = {s_best}")
                        f.write(f"Iter {iter_count} (k={k}): NOVO MELHOR: {cost_best:,.2f} | S = {s_best}\n")
                    k = 1 
                    iter_count += 1
                else:
                    iter_count += 1
                    k += 1
            
            duration = time.time() - iter_start
            history_current.append(cost_current)
            history_best.append(cost_best)
            
            f.write(f"Iter {iter_count}: {duration:.4f}s | Cost={cost_current:.2f}\n")
            
            # Logs periódicos
            if iter_count % 10 == 0:
                print(f"Iter {iter_count}/{max_iter} | Best: {cost_best:,.2f} | Time: {duration:.2f}s")
                if HAS_EXTERNAL_PLOTS and edges_list:
                    gerar_visualizacao_personalizada(iter_count, s_current, dist_matrix, edges_list, num_nodes, output_dir)
        
        f.write(f"Fim")
                
    print(f"\nTempo Total: {time.time() - start_time:.2f}s")
    plot_convergencia(history_best, history_current, output_dir)
    return s_best, cost_best

# =====================================================================
#   5. MAIN (GARANTIA DE PATH)
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="VNS Otimizado em Lote.")
    parser.add_argument("--arquivo", nargs='?', default="instancias/pmed12.txt")
    parser.add_argument("--p", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--k-max", type=int, default=3) 
    
    args = parser.parse_args()

    INSTANCIA = args.arquivo
    
    # -------------------------------------------------------------
    # CRIAÇÃO DO DIRETÓRIO DE EXECUÇÃO
    # -------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Normaliza o path para o sistema operacional atual (evita mistura de / e \)
    output_dir = os.path.normpath(os.path.join("relatorios", f"execucao_{timestamp}"))
    
    # Cria a pasta raiz do relatório
    os.makedirs(output_dir, exist_ok=True)
    
    # Leitura de dados
    if not USING_MOCK and os.path.exists(args.arquivo):
        print(f"--- Lendo {args.arquivo} ---")
        try:
            P_loaded, regioes, pop_dict, dist_matrix = load_pmed_instance(args.arquivo)
            p_final = args.p if args.p > 0 else (P_loaded if P_loaded else 4)
            pop_arr = np.array([pop_dict[r] for r in regioes], dtype=float)
            
            # Carrega arestas para plot
            edges_list = None
            if HAS_EXTERNAL_PLOTS:
                G_base, _ = ler_instancia_pmed(args.arquivo)
                edges_list = [(u, v, d.get('weight', 1.0)) for u, v, d in G_base.edges(data=True)]
                try:
                    # Salva grafo base com caminho absoluto
                    caminho_base = os.path.abspath(os.path.join(output_dir, "grafo_base.png"))
                    desenhar_grafo(G_base, caminho_base)
                except: pass

        except Exception as e:
            print(f"Erro no load: {e}. Usando Mock.")
            regioes = REGIOES_MOCK
            dist_matrix = DISTANCIA_MOCK
            pop_arr = np.array([POPULACAO_MOCK[r] for r in regioes], dtype=float)
            p_final = 4
            edges_list = None
    else:
        regioes = REGIOES_MOCK
        dist_matrix = DISTANCIA_MOCK
        pop_arr = np.array([POPULACAO_MOCK[r] for r in regioes], dtype=float)
        p_final = 4
        edges_list = None

    # Execução passando o diretório criado
    vns_solve(dist_matrix, pop_arr, p_final, args.alpha, args.max_iter, args.k_max, regioes, edges_list, output_dir=output_dir)

if __name__ == "__main__":
    main()