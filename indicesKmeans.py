# Chamada das bibliotecas
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from urllib.request import urlopen

# Carregar os dados
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-pix"
data = urlopen(url)
dataset = np.loadtxt(data)

# Normalizar os dados
scaler = StandardScaler()
dataset_scaled = scaler.fit_transform(dataset)

# Função para calcular os índices
def calculate_indices(X, labels):
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    
    # Índice de Dunn
    min_dist_between_clusters = np.inf
    max_diameter_within_clusters = -np.inf
    for i in range(len(np.unique(labels))):
        for j in range(i + 1, len(np.unique(labels))):
            dist_clusters = np.linalg.norm(X[labels == i].mean(axis=0) - X[labels == j].mean(axis=0))
            if dist_clusters < min_dist_between_clusters:
                min_dist_between_clusters = dist_clusters
        
        diameter_cluster_i = np.max([np.linalg.norm(x - X[labels == i].mean(axis=0)) for x in X[labels == i]])
        if diameter_cluster_i > max_diameter_within_clusters:
            max_diameter_within_clusters = diameter_cluster_i
    
    dunn_index = min_dist_between_clusters / max_diameter_within_clusters
    return silhouette, davies_bouldin, calinski_harabasz, dunn_index

# Função K-means
def kmeans_algorithm(X, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    indices = calculate_indices(X, labels)
    return indices

# Número de clusters
k_values = [2, 3, 4, 5, 6]

# Calcular e imprimir os índices
for k in k_values:
    silhouette, davies_bouldin, calinski_harabasz, dunn_index = kmeans_algorithm(dataset_scaled, k)
    print("===============================================")
    print(f"Para k={k}: ")
    print(f"Silhouette Index: {silhouette}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz}")
    print(f"Dunn Index: {dunn_index}")