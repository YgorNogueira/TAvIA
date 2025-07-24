import numpy as np

def calcular_custo(solucao, matriz_fluxo, matriz_distancia):
    """Calcula o custo total de uma solução para o PQA."""
    n = len(solucao)
    custo_total = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                instalacao_i = solucao[i]
                instalacao_j = solucao[j]
                custo_total += matriz_fluxo[instalacao_i, instalacao_j] * matriz_distancia[i, j]
    return custo_total

class ParticulaGeneticaMovel:
    def __init__(self, meme):
        """
        O meme é um dicionário representando um fragmento de solução.
        Ex: {local_alvo: instalacao_desejada}
        """
        self.meme = meme

    def transcrever(self, cromossomo_alvo):
        """
        Tenta aplicar o meme ao cromossomo alvo através de trocas (swaps).
        Retorna um novo cromossomo modificado.
        """
        cromossomo_modificado = cromossomo_alvo.copy()
        
        for local_alvo, instalacao_desejada in self.meme.items():
            # Encontra onde a 'instalacao_desejada' está atualmente
            local_atual_da_instalacao = np.where(cromossomo_modificado == instalacao_desejada)[0][0]
            
            # Pega a instalação que está no 'local_alvo'
            instalacao_no_local_alvo = cromossomo_modificado[local_alvo]
            
            # Realiza a troca para colocar a 'instalacao_desejada' no 'local_alvo'
            cromossomo_modificado[local_alvo] = instalacao_desejada
            cromossomo_modificado[local_atual_da_instalacao] = instalacao_no_local_alvo
            
        return cromossomo_modificado