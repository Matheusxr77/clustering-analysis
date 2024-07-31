# Chamada das bibliotecas
import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
import random

# Função para calcular a distância euclidiana
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Implementação do k-means++
def kmeans_plusplus_init(X, k):
    centers = []
    centers.append(random.choice(X))
    while len(centers) < k:
        distances = np.array([min([euclidean_distance(x, c) for c in centers]) for x in X])
        probabilities = distances / distances.sum()
        cumulative_probabilities = np.cumsum(probabilities)
        r = random.random()
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                centers.append(X[j])
                break
    return np.array(centers)

# Função para executar o k-means
def kmeans(X, k, max_iters=100):
    centers = kmeans_plusplus_init(X, k)
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        for x in X:
            distances = [euclidean_distance(x, center) for center in centers]
            clusters[np.argmin(distances)].append(x)
        new_centers = np.array([np.mean(cluster, axis=0) for cluster in clusters])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, clusters

# Gerando dados de exemplo
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Configurações do algoritmo
num_clusters = 4
num_repeats = 10

best_inertia = float('inf')
best_centers = None
best_clusters = None

# Executar o k-means várias vezes
for _ in range(num_repeats):
    centers, clusters = kmeans(X, num_clusters)
    inertia = sum([euclidean_distance(x, centers[idx]) ** 2 for idx, cluster in enumerate(clusters) for x in cluster])
    if inertia < best_inertia:
        best_inertia = inertia
        best_centers = centers
        best_clusters = clusters

# Visualização dos clusters
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for idx, cluster in enumerate(best_clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], s=50, c=colors[idx % len(colors)], label=f'Cluster {idx + 1}')
plt.scatter(best_centers[:, 0], best_centers[:, 1], c='black', s=200, alpha=0.75, label='Centroides')
plt.title('Clustering com Implementação Manual do K-means++ e Repetições Múltiplas')
plt.legend()
plt.show()