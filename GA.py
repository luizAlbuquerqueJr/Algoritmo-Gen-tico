# generate random integer values
from random import seed
from random import randint
import numpy as np
import math

#Combinação em torneio
NUM_INVIDUOS_TORNEIO = 150
NUM_GENES_PERMUTADOS = 32
NUM_CRUZAMENTO_POR_EPOCA = 1
#
TAXA_MUTACAO = 0.01  #  AINDA NAO IMPLEMENTADA
NUM_BITS_MUTACAO = 10
NUM_POP = 200
NUM_BITS = 64
SQR_NUM_BITS = int(pow(NUM_BITS,1/2))
NUM_RAINHA = 8
def mutacao(populacao):
    
    for individuo in populacao:
        a = randint(1, int(1/TAXA_MUTACAO))
        if(a == 1):
            for _ in range(NUM_BITS_MUTACAO):
                individuo = individuo.reshape(-1,NUM_BITS)[0]
                # print('indiviuo')
                # print(individuo)
                i = randint(0, NUM_BITS - 1)
                j = randint(0, NUM_BITS - 1)
                # print(i,j)
                # print(individuo[2])
                while individuo[i] == individuo[j%(NUM_BITS-1)]:
                    j =j + 1
                # print(j,i)
                individuo[i], individuo[j%(NUM_BITS-1)] = individuo[j%(NUM_BITS-1)], individuo[i]
                # print(individuo)
                individuo = individuo.reshape(-1,SQR_NUM_BITS)
                
        

def calculateFitness(populacao):
    fitness = np.zeros(populacao.shape[0])
    ## calula numero de choques
    count = 0
    for individuo in populacao:
        choques = 0 
        #choques na horizontal e vertical
        for index in range(SQR_NUM_BITS):
            linha = individuo[index]
            coluna = individuo[:,index]
            num = np.count_nonzero(linha == 1) 
            if num > 1:
                choques += num-1
            num = np.count_nonzero(coluna == 1) 
            if num > 1:
                choques += num-1
        #num Choques diagonal direita
        for index in range(-(SQR_NUM_BITS+2),SQR_NUM_BITS-1):
            diagonal = individuo.diagonal(index)
            num = np.count_nonzero(diagonal == 1) 
            if num > 1:
                choques += num-1

            diagonal = np.flipud(individuo).diagonal(index)  # Vertical flip
            num = np.count_nonzero(diagonal == 1) 
            if num > 1:
                choques += num-1

        fitness[count] = 1/(1+choques)
        count += 1
    #
    return fitness
def combinacao_torneio(populacao,fitness):
    for _ in range(NUM_CRUZAMENTO_POR_EPOCA):
        individuosSelecionados = np.random.choice(NUM_POP, NUM_INVIDUOS_TORNEIO, replace=False)
        # print(populacao[individuosSelecionados])
        # print(fitness[individuosSelecionados])
        a = np.argsort(fitness[individuosSelecionados])
        a = a[-2:] #pega os dois mais aptos
        #Crossing-over
        # print(a)
        # print("populacao")
        # print(populacao)
        
        individuo1 = populacao[a[0]].reshape(-1,NUM_BITS)[0].copy()
        individuo2 = populacao[a[1]].reshape(-1,NUM_BITS)[0].copy()
        # print("individuos selecionados")
        # print(individuo1)
        # print(individuo2)
        corte = randint(0, NUM_BITS - 1 - NUM_GENES_PERMUTADOS)
        # print(corte)
        # print(individuo1[corte:corte+NUM_GENES_PERMUTADOS])
        # print(individuo2[corte:corte+NUM_GENES_PERMUTADOS])
        tmp = individuo2[corte:corte+NUM_GENES_PERMUTADOS].copy()
        individuo2[corte:corte+NUM_GENES_PERMUTADOS], individuo1[corte:corte+NUM_GENES_PERMUTADOS]  = individuo1[corte:corte+NUM_GENES_PERMUTADOS], tmp
        num1 = np.count_nonzero(individuo1 == 1) 
        num2 = 2*SQR_NUM_BITS - num1
        count = corte + NUM_GENES_PERMUTADOS
        
        while(num1 != num2):
            # print(num1,num2)
            # print(count)
            # print(count%63)
            count = count%63
            if(num2 - SQR_NUM_BITS< 0):
                # print(individuo1)
                # print(individuo2)
                
                if(individuo1[count] != individuo2[count] and individuo1[count] == 1):
                    # print("asadsaasf")
                    individuo1[count] = 0
                    individuo2[count] = 1
            if(num1 - SQR_NUM_BITS< 0):
                if(individuo1[count] != individuo2[count] and individuo2[count] == 1):
                    individuo1[count] = 1
                    individuo2[count] = 0
            # print("aaaaaaaaaaaaa")
            count += 1                            
            num1 = np.count_nonzero(individuo1 == 1) 
            num2 = 2*SQR_NUM_BITS - num1
            
        # print(individuo1)
        # print(individuo2)        
        # print(populacao)
        # print("populacao")
        
        populacao =  np.insert(populacao,0,individuo1.copy())
        populacao =  np.insert(populacao,0,individuo2.copy())
        
        populacao = np.reshape(populacao,(NUM_POP + 2,SQR_NUM_BITS,SQR_NUM_BITS))
        return populacao
        print(populacao)
def selecaoNatural(populacao,fitness):
    # print("excluindo")
    for _ in range(NUM_CRUZAMENTO_POR_EPOCA):
        individuosSelecionados = np.random.choice(NUM_POP, NUM_INVIDUOS_TORNEIO, replace=False)    
        a = np.argsort(fitness)
        # print(a)
        a = a[:2] #pega os dois menos aptos
        populacao = np.delete(populacao, a, axis = 0)
        # populacao = np.delete(populacao, a[1], axis = 0)
        return populacao
        # print(populacao)
    
    

def insereRainhas(individuo):   
    count = 0
    while(count != NUM_RAINHA):
        i = randint(0, SQR_NUM_BITS - 1)
        j = randint(0, SQR_NUM_BITS - 1)
        if(individuo[i][j] == 0):
            individuo[i][j] = 1
            count +=  1

seed()
# Inicializa população
populacao = np.zeros((NUM_POP,SQR_NUM_BITS,SQR_NUM_BITS))

for individuo in populacao:
    # insere 8 rainhas no tabuleiro
    insereRainhas(individuo)
###########################################
for geracao in range(100000):
    print("Geração: ", geracao)
    fitness = calculateFitness(populacao)

    mutacao(populacao)
    populacao = combinacao_torneio(populacao,fitness)
    fitness = calculateFitness(populacao)
    populacao = selecaoNatural(populacao,fitness)
    # print(populacao.shape)

    fitness = calculateFitness(populacao)
    # print(fitness)
    print("Média do Fitness da populacao: ",np.mean(fitness))
    print("Individuo mais apto: ",np.max(fitness))
    print(populacao[np.argmax(fitness)])
# print(populacao)


#sort by index
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
#

# print(fitness)

