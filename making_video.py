# Importar as bibliotecas necessárias
from sklearn.cluster import Birch
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Gerar dados sintéticos para clustering
np.random.seed(1)  # Definir a semente para reprodutibilidade
x, _ = make_blobs(n_samples=50, centers=5, cluster_std=1.2)  # Criar 50 amostras em 5 grupos com desvio padrão de 1.2

# Definir o método BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
bclust = Birch(branching_factor=25, threshold=1)  # Configurar os parâmetros do algoritmo

# Definir um mapa de cores personalizado: vermelho, verde e amarelo
colors = np.array(['red', 'green', 'yellow'])  # Array de cores para cada cluster

# Preparar a figura para a animação
fig, ax = plt.subplots()  # Criar uma figura e eixos
sc = ax.scatter([], [], c=[], cmap=None, vmin=0, vmax=2)  # Inicializar a plotagem de dispersão com cores personalizadas
ax.set_xlim(np.min(x[:, 0]) - 1, np.max(x[:, 0]) + 1)  # Definir limites do eixo x
ax.set_ylim(np.min(x[:, 1]) - 1, np.max(x[:, 1]) + 1)  # Definir limites do eixo y
ax.set_title("Animação de Clustering com BIRCH")  # Adicionar título ao gráfico
ax.set_xlabel("Característica 1")  # Rotular eixo x
ax.set_ylabel("Característica 2")  # Rotular eixo y

# Função para inicializar a animação
def init():
    sc.set_offsets(np.empty((0, 2)))  # Inicializar com um array vazio 2D
    sc.set_color([])  # Redefinir as cores
    return sc,

# Função de atualização para a animação
def update(frame):
    # Ajustar o BIRCH com os dados até o frame atual
    bclust.partial_fit(x[:frame + 1])  # Treinar incrementalmente
    labels = bclust.predict(x[:frame + 1])  # Prever os rótulos dos clusters

    # Mapear os rótulos para as cores definidas
    point_colors = colors[labels % len(colors)]  # Garantir que as cores sejam indexadas corretamente

    # Atualizar o gráfico de dispersão
    sc.set_offsets(x[:frame + 1])  # Atualizar os pontos
    sc.set_color(point_colors)  # Atualizar as cores
    ax.set_title(f"Animação de Clustering com BIRCH - Pontos Processados: {frame + 1}")  # Atualizar o título
    return sc,

# Criar a animação
ani = FuncAnimation(fig, update, frames=len(x), init_func=init, repeat=False)  # Configurar animação

# Salvar a animação como arquivo para visualização em Colab
ani.save('birch_clustering3.mp4', writer='ffmpeg', fps=5)  # Salvar como arquivo MP4 com 5 quadros por segundo

# Exibir a animação no Jupyter Notebook
from IPython.display import Video
Video('birch_clustering3.mp4')  # Reproduzir o vídeo da animação
