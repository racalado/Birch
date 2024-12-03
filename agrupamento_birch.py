from sklearn.cluster import Birch
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from numpy import random

# Geração de dados sintéticos para agrupamento
random.seed(1)  # Define a semente para reprodutibilidade
x, _ = make_blobs(n_samples=400, centers=5, cluster_std=1.2)  # Cria 50 pontos com 5 centros

# Visualização dos dados gerados
plt.scatter(x[:, 0], x[:, 1])  # Plota os pontos no plano 2D
plt.title("Dados de Agrupamento Gerados")  # Adiciona título ao gráfico
plt.xlabel("Coordenada X")  # Rótulo do eixo X
plt.ylabel("Coordenada Y")  # Rótulo do eixo Y
plt.show()  # Exibe o gráfico

# Configuração e ajuste do modelo BIRCH
bclust = Birch(branching_factor=200, threshold=1)  # Cria o modelo BIRCH
bclust.fit(x)  # Ajusta o modelo aos dados

# Exibe os parâmetros do modelo ajustado
print("Parâmetros do BIRCH:", bclust.get_params())

# Predição dos clusters para os dados
labels = bclust.predict(x)  # Prediz os clusters aos quais os pontos pertencem

# Visualização dos clusters
plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis')  # Plota os pontos com cores por cluster
plt.title("Agrupamento com BIRCH")  # Adiciona título ao gráfico
plt.xlabel("Coordenada X")  # Rótulo do eixo X
plt.ylabel("Coordenada Y")  # Rótulo do eixo Y
plt.show()  # Exibe o gráfico
