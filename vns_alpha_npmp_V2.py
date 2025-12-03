import random
import numpy as np
import os
import time
import argparse
import sys
from datetime import datetime

# =====================================================================
#   IMPORTAÇÕES E MOCKS
# =====================================================================

data_hora = datetime.now().strftime('%Y%m%d_%H%M%S')

# Garante que o diretório atual e 'src' sejam visíveis para importação
sys.path.append(os.getcwd())
if os.path.exists(os.path.join(os.getcwd(), 'src')):
    sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from src.pmed_loader_to_vns_alpha_npmp import load_pmed_instance
    USING_MOCK = False
except ImportError:
    try:
        # Tenta importar da raiz caso src falhe
        from pmed_loader_to_vns_alpha_npmp import load_pmed_instance
        USING_MOCK = False
    except ImportError:
        USING_MOCK = True

# --- IMPORTAÇÃO DAS SUAS FUNÇÕES PERSONALIZADAS (COM DIAGNÓSTICO) ---
HAS_EXTERNAL_PLOTS = False
try:
    # 1. Tenta importar de 'src' (estrutura organizada)
    try:
        from src.print_nodes import plot_solution
        from src.desenha_grafo_pmed import ler_instancia_pmed, desenhar_grafo
        HAS_EXTERNAL_PLOTS = True
        print("[DEBUG] Módulos gráficos importados de 'src/'.")
    except ImportError as e_src:
        # Se o erro for apenas "não achei a pasta src", tentamos da raiz.
        # Se o erro for "não achei networkx", isso vai ser pego lá embaixo.
        if "src" in str(e_src) or "No module named" in str(e_src):
            # 2. Tenta importar da raiz (estrutura plana)
            from print_nodes import plot_solution
            from desenha_grafo_pmed import ler_instancia_pmed, desenhar_grafo
            HAS_EXTERNAL_PLOTS = True
            print("[DEBUG] Módulos gráficos importados da raiz.")
        else:
            raise e_src # Era um erro dentro do arquivo (ex: falta matplotlib), repassa.

except ImportError as e:
    HAS_EXTERNAL_PLOTS = False
    print("\n" + "!"*60)
    print(f"[AVISO] Falha ao importar módulos gráficos: {e}")
    print("Verifique se:")
    print("  1. 'print_nodes.py' e 'desenha_grafo_pmed.py' estão na pasta.")
    print("  2. A biblioteca 'networkx' está instalada (pip install networkx).")
    print("!"*60 + "\n")

# Tenta importar matplotlib para gráficos auxiliares (convergência)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[AVISO] Matplotlib não encontrado. Gráfico de convergência não será gerado.")

# Dados Mock (Fallback)
REGIOES_MOCK = ['Centro', 'Norte', 'Leste', 'Sudeste', 'Sul', 'Oeste', 'Rural', 'SFX']
POPULACAO_MOCK = {
    'Centro': 72401, 'Norte': 61940, 'Leste': 181463, 'Sudeste': 62541,
    'Sul': 237572, 'Oeste': 64482, 'Rural': 15212, 'SFX': 1443
}
DISTANCIA_MOCK = np.array([
    [0, 6, 8, 7, 10, 9, 25, 35], [6, 0, 12, 10, 14, 8, 20, 40],
    [8, 12, 0, 5, 15, 18, 30, 45], [7, 10, 5, 0, 12, 16, 28, 42],
    [10, 14, 15, 12, 0, 20, 22, 38], [9, 8, 18, 16, 20, 0, 15, 50],
    [25, 20, 30, 28, 22, 15, 0, 60], [35, 40, 45, 42, 38, 50, 60, 0]
], dtype=np.float32)

# =====================================================================
#   1. FUNÇÕES DE AVALIAÇÃO (OTIMIZADA NUMPY)
# =====================================================================

def evaluate_solution_optimized(solution_indices, dist_matrix, populacao_arr, alpha):
    """
    Avalia a função objetivo (Minimizar soma das distâncias ponderadas).
    """
    dists_submatrix = dist_matrix[solution_indices, :]
    
    if alpha == 1:
        min_dists = np.min(dists_submatrix, axis=0)
        sum_dists = min_dists
    elif alpha < dists_submatrix.shape[0]:
        partitioned = np.partition(dists_submatrix, alpha-1, axis=0)
        top_alpha = partitioned[:alpha, :]
        sum_dists = np.sum(top_alpha, axis=0)
    else:
        sum_dists = np.sum(dists_submatrix, axis=0)

    total_cost = np.dot(sum_dists, populacao_arr)
    return total_cost

# =====================================================================
#   2. FUNÇÕES DE VIZINHANÇA
# =====================================================================

def neighborhood_k_exchange_random(solution, candidates_range, k):
    s_prime = list(solution)
    abertas = set(s_prime)
    fechadas = [i for i in candidates_range if i not in abertas]
    
    if len(abertas) < k or len(fechadas) < k:
        return s_prime

    for _ in range(k):
        if not abertas or not fechadas: break
        i_fechar = random.choice(list(abertas))
        abertas.remove(i_fechar)
        s_prime.remove(i_fechar)
        i_abrir = random.choice(fechadas)
        fechadas.remove(i_abrir)
        s_prime.append(i_abrir)
        
    return s_prime

def local_search_best_improvement(solution, dist_matrix, populacao_arr, alpha, num_regioes):
    s_best = list(solution)
    s_best_arr = np.array(s_best, dtype=int)
    cost_best = evaluate_solution_optimized(s_best_arr, dist_matrix, populacao_arr, alpha)
    
    while True:
        improved = False
        abertas = list(s_best)
        set_abertas = set(s_best)
        fechadas = [i for i in range(num_regioes) if i not in set_abertas]
        
        for i_fechar in abertas:
            for i_abrir in fechadas:
                neighbor = [x for x in s_best if x != i_fechar] + [i_abrir]
                neighbor_arr = np.array(neighbor, dtype=int)
                neighbor_cost = evaluate_solution_optimized(neighbor_arr, dist_matrix, populacao_arr, alpha)
                
                if neighbor_cost < cost_best:
                    cost_best = neighbor_cost
                    s_best = neighbor
                    improved = True
                    break 
            if improved: break
            
        if not improved:
            break
            
    return s_best, cost_best

# =====================================================================
#   3. VISUALIZAÇÃO COM SUAS BIBLIOTECAS
# =====================================================================

def gerar_visualizacao_personalizada(iter_count, current_s, dist_matrix, edges_list, n_vertices):
    """
    Prepara os dados do VNS (0-based) para usar a função plot_solution (1-based)
    do arquivo print_nodes.py.
    """
    if not HAS_EXTERNAL_PLOTS:
        return

    if edges_list is None:
        return

    # 1. Converter Solução (Indices 0-based -> 1-based)
    centers_1based = [i + 1 for i in current_s]
    
    # 2. Criar Dicionário de Distâncias (Necessário para plot_solution)
    dist_dict = {}
    for i in range(n_vertices):
        dist_dict[i+1] = {}
        for j in range(n_vertices):
            dist_dict[i+1][j+1] = float(dist_matrix[i, j])

    # 3. Calcular Atribuição (Assignment)
    assignment = {}
    for i in range(n_vertices):
        node_idx = i
        # Acha o centro mais próximo em current_s
        best_center = -1
        min_dist = float('inf')
        
        for center_idx in current_s:
            d = dist_matrix[node_idx, center_idx]
            if d < min_dist:
                min_dist = d
                best_center = center_idx
        
        # Guarda 1-based
        if best_center != -1:
            assignment[i+1] = best_center + 1
        else:
            assignment[i+1] = None

    # 4. Chamar sua função
    nome_arquivo = f"{data_hora}_vns_iter_{iter_count:03d}"
    print(f"   [Plot] Gerando imagem '{nome_arquivo}'...")
    
    try:
        plot_solution(
            n_vertices=n_vertices,
            edges=edges_list,
            dist_dict=dist_dict,
            centers=centers_1based,
            assignment=assignment,
            filename=nome_arquivo
        )
    except Exception as e:
        print(f"   [ERRO Plot] Falha ao chamar plot_solution: {e}")

def plot_convergencia(history_best, history_current):
    """Gera apenas o gráfico de convergência separadamente."""
    if not HAS_MATPLOTLIB: return
    
    plt.figure(figsize=(8, 4))
    iters = range(1, len(history_current) + 1)
    plt.plot(iters, history_current, label='Custo Atual', color='blue', alpha=0.6)
    plt.plot(iters, history_best, label='Melhor Global', color='green', linewidth=2)
    plt.xlabel('Iteração')
    plt.ylabel('Custo FO')
    plt.title('Evolução do VNS')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    out_dir = "relatorios/charts"
    os.makedirs(out_dir, exist_ok=True)
    caminho = f"{out_dir}/convergencia_vns_{data_hora}.png"
    plt.savefig(caminho)
    plt.close()
    print(f"[Plot Convergência] Salvo em: {caminho}")

# =====================================================================
#   4. VNS PRINCIPAL
# =====================================================================

def vns_solve(dist_matrix, populacao_arr, p, alpha, max_iter, k_max, regioes_names=None, edges_list=None):
    num_nodes = len(populacao_arr)
    candidates_range = range(num_nodes)
    
    # Solução Inicial
    s_current = random.sample(candidates_range, p)
    s_arr = np.array(s_current, dtype=int)
    cost_current = evaluate_solution_optimized(s_arr, dist_matrix, populacao_arr, alpha)
    
    s_best = list(s_current)
    cost_best = cost_current
    
    history_current = [cost_current]
    history_best = [cost_best]
    
    s_names_initial = [regioes_names[i] for i in s_current] if regioes_names else s_current
    print(f"\n--- VNS INICIADO (P={p}, Alpha={alpha}) ---")
    print(f"Custo Inicial: {cost_current:,.2f}")
    
    # Plot Inicial (Iter 0)
    if HAS_EXTERNAL_PLOTS:
        gerar_visualizacao_personalizada(0, s_current, dist_matrix, edges_list, num_nodes)
    
    data_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("relatorios", exist_ok=True)
    filename = f"relatorios/vns_alpha_npmp_{data_hora}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"VNS Alpha-NpMP | P={p} | Alpha={alpha}\n")
        f.write(f"Start S: {s_names_initial} | Cost: {cost_current}\n")
        f.write("-" * 50 + "\n")
        
        start_time = time.time()
        
        for iter_count in range(1, max_iter + 1):
            iter_start = time.time()
            k = 1
            
            while k <= k_max:
                s_prime = neighborhood_k_exchange_random(s_current, candidates_range, k)
                s_double, cost_double = local_search_best_improvement(
                    s_prime, dist_matrix, populacao_arr, alpha, num_nodes
                )
                
                if cost_double < cost_current:
                    s_current = s_double
                    cost_current = cost_double
                    k = 1
                    if cost_current < cost_best:
                        s_best = list(s_current)
                        cost_best = cost_current
                        print(f"Iter {iter_count}: NOVO MELHOR GLOBAL: {cost_best:,.2f}")
                        iter_count += 1
                else:
                    k += 1
                    iter_count += 1
            
            duration = time.time() - iter_start
            history_current.append(cost_current)
            history_best.append(cost_best)
            
            s_names_curr = [regioes_names[i] for i in s_current] if regioes_names else s_current
            f.write(f"Iter {iter_count}: {duration:.4f}s | Cost={cost_current:.2f} | S={s_names_curr}\n")
            
            if iter_count % 10 == 0:
                print(f"Iter {iter_count}/{max_iter} | Cost: {cost_current:,.2f}")
                # Chamada da sua função print_nodes
                if HAS_EXTERNAL_PLOTS:
                    gerar_visualizacao_personalizada(iter_count, s_current, dist_matrix, edges_list, num_nodes)
                
    total_time = time.time() - start_time
    print(f"\nTempo Total: {total_time:.2f}s")
    
    # Gera gráfico final de convergência
    plot_convergencia(history_best, history_current)
    
    return s_best, cost_best

# =====================================================================
#   5. MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Resolve alpha-NpMP usando VNS.")
    parser.add_argument("--arquivo", nargs='?', default="instancias/pmed12.txt", 
                        help="Caminho do arquivo pmed.")
    parser.add_argument("--p", type=int, default=0, help="Número de facilidades (P).")
    parser.add_argument("--alpha", type=int, default=1, help="Parâmetro Alpha.")
    parser.add_argument("--max-iter", type=int, default=50, help="Iterações máximas.")
    parser.add_argument("--k-max", type=int, default=3, help="Vizinhança máxima.")
    
    args = parser.parse_args()
    
    edges_list = None
    G_base = None

    # 1. Carregamento e Preparação dos Dados
    if not USING_MOCK and os.path.exists(args.arquivo):
        print(f"--- Lendo {args.arquivo} ---")
        try:
            # Carrega dados para o VNS
            P_loaded, regioes, pop_dict, dist_matrix = load_pmed_instance(args.arquivo)
            p_final = args.p if args.p > 0 else (P_loaded if P_loaded else 4)
            pop_arr = np.array([pop_dict[r] for r in regioes], dtype=float)
            
            # Carrega dados para o GRÁFICO (usando sua função desenha_grafo_pmed)
            if HAS_EXTERNAL_PLOTS:
                try:
                    G_base, _ = ler_instancia_pmed(args.arquivo)
                    # Extrai lista de arestas para o print_nodes: (u, v, weight)
                    edges_list = [(u, v, d.get('weight', 1.0)) for u, v, d in G_base.edges(data=True)]
                    
                    # Desenha o grafo base inicial
                    print(f"--- Gerando Grafo Base (desenha_grafo_pmed) ---")
                    try:
                        desenhar_grafo(G_base, caminho_imagem="grafo_inicial_base.png", mostrar_valores=False)
                    except Exception as e:
                        print(f"[AVISO] Falha ao desenhar grafo base: {e}")
                    
                except Exception as e:
                    print(f"Erro ao ler grafo para plotagem: {e}")

        except Exception as e:
            print(f"Erro ao carregar instância VNS: {e}. Usando Mock.")
            regioes = REGIOES_MOCK
            dist_matrix = DISTANCIA_MOCK
            pop_arr = np.array([POPULACAO_MOCK[r] for r in regioes], dtype=float)
            p_final = args.p if args.p > 0 else 4
    else:
        print("--- Usando Dados Mock/Padrão ---")
        regioes = REGIOES_MOCK
        dist_matrix = DISTANCIA_MOCK
        pop_arr = np.array([POPULACAO_MOCK[r] for r in regioes], dtype=float)
        p_final = args.p if args.p > 0 else 4
        
        # GERAÇÃO SINTÉTICA DE ARESTAS PARA O MOCK (Para garantir que plot funcione)
        if HAS_EXTERNAL_PLOTS:
            print("[INFO] Gerando arestas sintéticas para visualização Mock...")
            edges_list = []
            n_mock = len(regioes)
            for i in range(n_mock):
                for j in range(i + 1, n_mock):
                    weight = DISTANCIA_MOCK[i][j]
                    edges_list.append((i+1, j+1, float(weight)))

    # 2. Execução
    best_idx, best_val = vns_solve(
        dist_matrix=dist_matrix,
        populacao_arr=pop_arr,
        p=p_final,
        alpha=args.alpha,
        max_iter=args.max_iter,
        k_max=args.k_max,
        regioes_names=regioes,
        edges_list=edges_list
    )
    
    # 3. Resultado
    print("\n" + "="*50)
    print("✨ RESULTADO FINAL ✨")
    print(f"Melhor Custo (Z): {best_val:,.2f}")
    if regioes:
        names = [regioes[i] for i in best_idx]
        print(f"Facilidades: {names}")
    else:
        print(f"Facilidades: {best_idx}")
    print("="*50)

if __name__ == "__main__":
    main()