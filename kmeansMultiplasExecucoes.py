# Chamada das bibliotecas
import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Gerando dados de exemplo
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Configurações do algoritmo
num_clusters = 4
num_repeats = 10

# Inicialização com k-means++ e repetição do algoritmo
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=num_repeats, random_state=0)
kmeans.fit(X)

# Visualização dos clusters
plt.scatter(X[:, 0], X[:, 1], s=50, c=kmeans.labels_, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('Clustering com K-means++ e Repetições Múltiplas')
plt.show()