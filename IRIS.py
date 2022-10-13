# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris # Importando o dataset iris da biblioteca sklearn

iris = load_iris()

#DESCR retorna a descrição do dataset. 
print(iris['DESCR'][:] + "\n...")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris # Importando o dataset iris da biblioteca sklearn

iris = load_iris()

numEpocas   = 700
numAmostras = 100

# atributos (sepal lenght, sepal_width, petal lenght, petal width) vamos usar os dois primeiros 
features = iris.data.T

sepal_length = features[0, 0:100]
sepal_width = features[1, 0:100]

# saída esperada (class)
expected_output = iris['target'][0:100] # dividindo o dataset para trabalhar com apenas duas classificações›, 0 = setosa and 1 = versicolor 

#bias 
bias = 1

# Perceptron 

X = np.vstack((sepal_length, sepal_width)) #matriz de duas colunas 
Y = np.vstack(expected_output)  # saída ==ou seja== método supervisionado

# Taxa de aprendizado 
eta = 0.1

# Definindo vetor de pesos inicializando com ZERO
W = np.zeros([1, 3]) # (uma linha e 3 colunas =) duas entradas + bias // uma camada com 3 nós 

# Array de erros
e = np.zeros(100)

def funcaoAtivacao(valor):
    return 1 if valor > 0. else 0

#------------------------------------------
#------------Treinamento------------------ 
#------------------------------------------

for j in range(numEpocas): 
  for k in range(numAmostras): 

    # insere bias no vetor 
    Xb = np.hstack((bias, X[:,k])) #hstack empilha o bias em todas as linhas

    # Calcula o vetor de campo induzido (multiplicação vetorial // a matriz de entrada )
    V = np.dot(W, Xb)

    # Calcula a saída do Perceptron 
    Yr = funcaoAtivacao(V)
    # Yr = np. tanh(V) # deve ser usado em camadas do meio e nunca de saida pois mascara a saída 
    # Yr = np.sign(V) 

    # Calcula o erro
    e[k] = Y[k] - Yr

    # Treinamento da rede neural
    W = W + eta*e[k]*Xb # peso + taxa de aprendizado * erro * entrada ajustada com o bias
    #print("Vetor de erros (e) = " + str(e))

# colocar dentro do for se quiser olhar cada época detalhadamente
print("Vetor de erros (e) = " + str(e))

# Plot sepal length against sepal width:
plt.scatter(sepal_length, sepal_width, c=iris.target[0:100])
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.show()

plt.plot(sepal_length, sepal_width, color = 'r')
plt.show()

W