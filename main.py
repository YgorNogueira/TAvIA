import numpy as np
from qap_loader import carregar_instancia_qap
from tc_core import calcular_custo, ParticulaGeneticaMovel

def algoritmo_proto_g(matriz_fluxo, matriz_distancia, n_populacao, n_iteracoes, tamanho_time_memes):
    n, _ = matriz_fluxo.shape

    # Passo 2 (Quadro III): Gerar população inicial e avaliá-la
    populacao = [np.random.permutation(n) for _ in range(n_populacao)]
    custos = np.array([calcular_custo(p, matriz_fluxo, matriz_distancia) for p in populacao])
    
    # Passo 1 (Quadro III): Carregar um Banco de Memes.
    # Uso de um "time" das melhores soluções encontradas até agora.
    indices_ordenados = np.argsort(custos)
    banco_de_memes = [populacao[i] for i in indices_ordenados[:tamanho_time_memes]]
    melhor_custo_global = custos[indices_ordenados[0]]
    melhor_solucao_global = populacao[indices_ordenados[0]]
    
    print(f"Iteração 0: Melhor Custo = {melhor_custo_global}")

    # Passo 3 (Quadro III): Repita
    for i in range(1, n_iteracoes + 1):
        # Passo 4 (Quadro III): Gerar um agente através de competição de memes
        # Escolha de uma solução fonte do nosso banco para extrair um meme.
        solucao_fonte = banco_de_memes[np.random.randint(0, len(banco_de_memes))]
        
        # O tamanho do meme é aleatório
        tamanho_meme = np.random.randint(2, max(3, n // 5)) 
        locais_meme = np.random.choice(n, tamanho_meme, replace=False)
        meme_dict = {local: solucao_fonte[local] for local in locais_meme}
        
        agente_pgm = ParticulaGeneticaMovel(meme_dict)

        # Passo 5 (Quadro III): Para todo cromossomo da população
        for j in range(n_populacao):
            cromossomo_atual = populacao[j]
            custo_atual = custos[j]
            
            # Tenta a manipulação
            cromossomo_manipulado = agente_pgm.transcrever(cromossomo_atual)
            custo_manipulado = calcular_custo(cromossomo_manipulado, matriz_fluxo, matriz_distancia)
            
            # Passo 6: Critério de aceitação da PGM (melhoria estrita) [cite: 334]
            if custo_manipulado < custo_atual:
                populacao[j] = cromossomo_manipulado
                custos[j] = custo_manipulado
                
                # Passo 7: Realimentação imunológica 
                # Se a nova solução é melhor que a pior do banco, ela entra.
                pior_custo_banco = calcular_custo(banco_de_memes[-1], matriz_fluxo, matriz_distancia)
                if custo_manipulado < pior_custo_banco:
                    banco_de_memes.append(cromossomo_manipulado)
                    # Mantém o banco ordenado e com tamanho fixo
                    banco_de_memes.sort(key=lambda s: calcular_custo(s, matriz_fluxo, matriz_distancia))
                    banco_de_memes.pop()

        # Atualiza a melhor solução global encontrada
        melhor_indice_iteracao = np.argmin(custos)
        if custos[melhor_indice_iteracao] < melhor_custo_global:
            melhor_custo_global = custos[melhor_indice_iteracao]
            melhor_solucao_global = populacao[melhor_indice_iteracao]
            print(f"Iteração {i}: Novo Melhor Custo Global = {melhor_custo_global}")
            
    return melhor_solucao_global, melhor_custo_global


if __name__ == '__main__':
    # --- Configuração do Experimento ---
    CAMINHO_INSTANCIA = 'instancias/nug25.dat'
    N_POPULACAO = 200      # Número de indivíduos na população
    N_ITERACOES = 5000     # Número de gerações
    TAMANHO_TIME_MEMES = 10 # Tamanho do banco de soluções para gerar memes

    # --- Execução ---
    n, fluxo, distancia = carregar_instancia_qap(CAMINHO_INSTANCIA)
    
    print(f"Resolvendo instância de tamanho {n}...")
    
    solucao_final, custo_final = algoritmo_proto_g(
        fluxo, 
        distancia,
        n_populacao=N_POPULACAO,
        n_iteracoes=N_ITERACOES,
        tamanho_time_memes=TAMANHO_TIME_MEMES
    )
    
    print("\n--- Resultados Finais ---")
    print(f"Melhor solução encontrada: {solucao_final}")
    print(f"Custo da melhor solução: {custo_final}")