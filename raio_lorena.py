import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from datetime import datetime

"""
Calcula a distância média de um nó para seus 3 nós mais próximos (usado como estimativa para o raio de cobertura)

python -m "raio_lorena" --arquivo "data_lorena/coord324.txt" --k 3
"""

# =====================================================================
#   1. LOADER (Reutilizado para compatibilidade)
# =====================================================================

def load_coords_only(coord_file):
    """Lê apenas as coordenadas para análise de distância."""
    print(f"[Loader] Lendo coordenadas: {coord_file}")
    coords = []
    
    try:
        with open(coord_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            
        if not lines: raise ValueError("Arquivo vazio")
        
        # Cabeçalho
        header = lines[0].split()
        n_nodes = int(header[0])
        print(f"[Info] Instância com {n_nodes} nós.")
        
        # Dados
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 2:
                coords.append((float(parts[0]), float(parts[1])))
                
    except Exception as e:
        raise ValueError(f"Erro ao ler arquivo: {e}")
        
    coords_arr = np.array(coords, dtype=np.float32)
    
    # Matriz de Distância
    print("[Calc] Gerando matriz de distâncias...")
    delta = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(delta**2, axis=-1))
    
    return coords_arr, dist_matrix

# =====================================================================
#   2. ANÁLISE DE VIZINHOS
# =====================================================================

def analisar_vizinhos(dist_matrix, k=3):
    """
    Calcula a distância média para os K vizinhos mais próximos de cada nó.
    """
    n = dist_matrix.shape[0]
    if n <= k:
        print(f"[Aviso] Número de nós ({n}) menor ou igual a K ({k}). Ajustando K para {n-1}.")
        k = n - 1

    print(f"[Análise] Buscando os {k} vizinhos mais próximos para cada nó...")
    
    # Ordena cada linha da matriz (axis=1)
    # A coluna 0 será sempre 0.0 (distância para si mesmo)
    # As colunas 1 a K são os K vizinhos mais próximos
    sorted_dists = np.sort(dist_matrix, axis=1)
    
    # Pegamos as colunas 1 até k+1 (slice 1:k+1)
    nearest_k_dists = sorted_dists[:, 1:k+1]
    
    # Média por nó (axis=1)
    avg_per_node = np.mean(nearest_k_dists, axis=1)
    
    return avg_per_node, nearest_k_dists

# =====================================================================
#   3. VISUALIZAÇÃO
# =====================================================================

def plot_distribuicao(avg_per_node, k, output_dir):
    plt.figure(figsize=(10, 6))
    
    # Histograma
    plt.hist(avg_per_node, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Linhas de média
    mean_val = np.mean(avg_per_node)
    median_val = np.median(avg_per_node)
    
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Média Global: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Mediana: {median_val:.2f}')
    
    plt.title(f"Distribuição da Distância Média aos {k} Vizinhos Mais Próximos")
    plt.xlabel("Distância Média")
    plt.ylabel("Frequência (Número de Nós)")
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    filename = os.path.join(output_dir, f"histograma_k{k}_vizinhos.png")
    plt.savefig(filename)
    print(f"[Plot] Histograma salvo em: {filename}")
    plt.close()

# =====================================================================
#   MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Analisa distância média de vizinhos.")
    parser.add_argument("--arquivo", required=True, help="Arquivo de Coordenadas")
    parser.add_argument("--k", type=int, default=3, help="Número de vizinhos a considerar (Default: 3)")
    
    args = parser.parse_args()
    
    # Setup Pastas
    output_dir = "analise_vizinhanca"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Carregar
        coords, dist_matrix = load_coords_only(args.arquivo)
        
        # 2. Calcular
        avg_per_node, raw_k_dists = analisar_vizinhos(dist_matrix, k=args.k)
        
        # 3. Estatísticas Gerais
        global_mean = np.mean(avg_per_node)
        global_min = np.min(avg_per_node)
        global_max = np.max(avg_per_node)
        global_std = np.std(avg_per_node)
        
        print("\n" + "="*40)
        print(f" RESULTADOS PARA K={args.k} VIZINHOS")
        print("="*40)
        print(f"Média GLOBAL das médias: {global_mean:,.4f}")
        print(f"Distância Mínima encontrada: {global_min:,.4f}")
        print(f"Distância Máxima encontrada: {global_max:,.4f}")
        print(f"Desvio Padrão: {global_std:,.4f}")
        print("-" * 40)
        print("INTERPRETAÇÃO PARA MCLP:")
        print(f"-> Um Raio (S) abaixo de {global_min:.2f} deixará nós isolados.")
        print(f"-> Um Raio (S) próximo de {global_mean:.2f} deve cobrir vizinhanças locais.")
        print("="*40 + "\n")
        
        # 4. Plotar
        plot_distribuicao(avg_per_node, args.k, output_dir)
        
        # 5. Salvar Relatório TXT
        with open(os.path.join(output_dir, "analise_stats.txt"), "w") as f:
            f.write(f"Arquivo: {args.arquivo}\n")
            f.write(f"K Vizinhos: {args.k}\n")
            f.write(f"Global Mean: {global_mean}\n")
            f.write(f"Min: {global_min}\n")
            f.write(f"Max: {global_max}\n")
            
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()