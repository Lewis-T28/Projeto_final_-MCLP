import random
import numpy as np
import time

# --- 1. DADOS DE ENTRADA (SÃO JOSÉ DOS CAMPOS) ---

REGIOES = ['Centro', 'Norte', 'Leste', 'Sudeste', 'Sul', 'Oeste', 'Rural', 'SFX']
POPULACAO = {
    'Centro': 72401, 'Norte': 61940, 'Leste': 181463, 'Sudeste': 62541,
    'Sul': 237572, 'Oeste': 64482, 'Rural': 15212, 'SFX': 1443
}
# Matriz de Distâncias (d_ij em km) - Aprox.
DISTANCIA = np.array([
    [0, 6, 8, 7, 10, 9, 25, 35], [6, 0, 12, 10, 14, 8, 20, 40], 
    [8, 12, 0, 5, 15, 18, 30, 45], [7, 10, 5, 0, 12, 16, 28, 42], 
    [10, 14, 15, 12, 0, 20, 22, 38], [9, 8, 18, 16, 20, 0, 15, 50], 
    [25, 20, 30, 28, 22, 15, 0, 60], [35, 40, 45, 42, 38, 50, 60, 0]
])

P = 7  # Número de facilidades a abrir
ALPHA = 2 # Número de vizinhos (resiliência)


# --- 2. FUNÇÕES AUXILIARES ---

def evaluate_solution(solution_indices, dist_matrix, populacao_dict, alpha):
    """
    Calcula o valor da Função Objetivo (FO) para o αNpMP.
    FO: Minimizar a soma ponderada das distâncias às ALPHA facilidades mais próximas.
    """
    total_cost = 0
    
    # 1. Para cada região de demanda (cliente)
    for j_idx, j_regiao in enumerate(REGIOES):
        populacao_j = populacao_dict[j_regiao]
        
        # 2. Encontrar as distâncias para as facilidades abertas
        distancias_abertas = []
        for i_idx in solution_indices:
            distancias_abertas.append(dist_matrix[i_idx, j_idx])
            
        # 3. Ordenar e selecionar as ALPHA menores distâncias
        distancias_abertas.sort()
        
        if len(distancias_abertas) < alpha:
            # Solução inválida (menos de ALPHA facilidades abertas)
            return float('inf') 
            
        alpha_distancias = distancias_abertas[:alpha]
        
        # 4. Adicionar o custo ponderado à FO
        total_cost += populacao_j * sum(alpha_distancias)
        
    return total_cost

def generate_initial_solution(p_val):
    """Gera uma solução inicial aleatória (conjunto de P índices de facilidades abertas)."""
    return random.sample(range(len(REGIOES)), p_val)

# --- 3. ESTRUTURA DO VNS (Variable Neighborhood Search) ---

def shaking(solution, k):
    """
    Aplica a perturbação (shaking) na solução usando k-swap.
    k: número de trocas a serem feitas.
    """
    s_prime = list(solution)
    
    # Índices de facilidades abertas (para fechar)
    abertas = list(s_prime)
    # Índices de facilidades fechadas (para abrir)
    fechadas = [i for i in range(len(REGIOES)) if i not in abertas]
    
    # Realiza k trocas (fechar k abertas e abrir k fechadas)
    for _ in range(k):
        if not abertas or not fechadas:
            break # Não há mais trocas possíveis
            
        # 1. Escolhe uma facilidade aberta para fechar
        i_fechar = random.choice(abertas)
        abertas.remove(i_fechar)
        s_prime.remove(i_fechar)
        
        # 2. Escolhe uma facilidade fechada para abrir
        i_abrir = random.choice(fechadas)
        fechadas.remove(i_abrir)
        s_prime.append(i_abrir)
        
    return s_prime

def local_search(solution, dist_matrix, populacao_dict, alpha):
    """
    Aplica a busca local (1-swap) para encontrar um mínimo local.
    """
    s_star = list(solution)
    cost_star = evaluate_solution(s_star, dist_matrix, populacao_dict, alpha)
    
    while True:
        best_neighbor = None
        best_cost = cost_star
        
        # Estrutura de vizinhança 1-swap (troca 1 aberta por 1 fechada)
        abertas = list(s_star)
        fechadas = [i for i in range(len(REGIOES)) if i not in abertas]
        
        melhoria_encontrada = False
        
        for i_fechar in abertas:
            for i_abrir in fechadas:
                # Gera o vizinho (troca i_fechar por i_abrir)
                neighbor = [i for i in s_star if i != i_fechar] + [i_abrir]
                neighbor_cost = evaluate_solution(neighbor, dist_matrix, populacao_dict, alpha)
                
                if neighbor_cost < best_cost:
                    best_cost = neighbor_cost
                    best_neighbor = neighbor
                    melhoria_encontrada = True
        
        if melhoria_encontrada:
            s_star = best_neighbor
            cost_star = best_cost
        else:
            break # Mínimo local encontrado
            
    return s_star, cost_star

def vns_solve(p_val, alpha_val, dist_matrix, populacao_dict, max_iter=100, k_max=4):
    """
    Algoritmo Variable Neighborhood Search (VNS) para o αNpMP.
    """
    # 1. Geração da Solução Inicial
    s = generate_initial_solution(p_val)
    cost = evaluate_solution(s, dist_matrix, populacao_dict, alpha_val)
    
    s_best = list(s)
    cost_best = cost
    
    print(f"Custo Inicial: {cost_best:,.2f}")
    
    for iter_count in range(max_iter):
        k = 1
        while k <= k_max:
            # 2. Shaking (Perturbação)
            s_prime = shaking(s, k)
            
            # 3. Busca Local (Busca o mínimo local s'' na vizinhança 1-swap)
            s_double_prime, cost_double_prime = local_search(s_prime, dist_matrix, populacao_dict, alpha_val)
            
            # 4. Movimento (Aceitação da Solução)
            if cost_double_prime < cost:
                s = s_double_prime
                cost = cost_double_prime
                k = 1 # Reinicia a busca com a vizinhança mais simples
                
                if cost < cost_best:
                    s_best = s
                    cost_best = cost
                    print(f"Iter {iter_count+1}: NOVO MELHOR CUSTO: {cost_best:,.2f} (k={k})")
            else:
                k += 1 # Aumenta a vizinhança para tentar escapar do mínimo local
        iter_count += 1
                
    return s_best, cost_best

# --- 4. EXECUÇÃO E RESULTADOS ---

if __name__ == '__main__':
    
    print("="*50)
    print(f"ESTRUTURA VNS PARA αNpMP (α={ALPHA}, P={P})")
    print("="*50)
    
    # Execução do VNS
    best_solution_indices, best_cost = vns_solve(P, ALPHA, DISTANCIA, POPULACAO, max_iter=50, k_max=4)
    
    # Conversão dos índices para nomes de regiões
    best_solution_regions = [REGIOES[i] for i in best_solution_indices]
    
    print("\n" + "="*50)
    print("RESULTADO FINAL DO VNS")
    print("="*50)
    print(f"Melhor Custo (Z): {best_cost:,.2f} hab·km")
    print(f"Melhores Facilidades Abertas: {best_solution_regions}")
    
    pop_total = sum(POPULACAO.values())
    dist_media = best_cost / (ALPHA * pop_total)
    print(f"Distância Média Ponderada (Primária + Secundária): {dist_media:,.4f} km")
