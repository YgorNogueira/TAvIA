import numpy as np

def carregar_instancia_qap(filepath):
    """
    Lê um arquivo de instância do QAPLib e retorna as matrizes.
    """
    with open(filepath, 'r') as f:
        linhas = f.readlines()
    
    linhas = [linha.strip() for linha in linhas if linha.strip()]
    
    n = int(linhas[0])
    
    matriz_fluxo = []
    matriz_distancia = []
    
    # As matrizes podem estar em várias linhas
    dados_matriz_fluxo = ' '.join(linhas[1:n+1]).split()
    dados_matriz_distancia = ' '.join(linhas[n+1:]).split()

    # Converte os dados para inteiros e remodela para matrizes n x n
    matriz_fluxo_flat = [int(val) for val in dados_matriz_fluxo]
    matriz_distancia_flat = [int(val) for val in dados_matriz_distancia]
    
    matriz_fluxo = np.array(matriz_fluxo_flat).reshape((n, n))
    matriz_distancia = np.array(matriz_distancia_flat).reshape((n, n))
    
    return n, matriz_fluxo, matriz_distancia