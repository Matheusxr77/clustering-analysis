# Chamada das bibliotecas
import numpy as np

from urllib.request import urlopen

# Carregar os dados
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-pix"
data = urlopen(url)
dataset = np.loadtxt(data)

# Normalizar os dados
def standard_scaler(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_scaled = (X - mean) / std
    return X_scaled

dataset_scaled = standard_scaler(dataset)

# Função para calcular os centróides
def calculate_centroids(X, labels, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        centroids[i, :] = X[labels == i].mean(axis=0)
    return centroids

# Função para calcular as distâncias
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Implementação do K-means
def kmeans(X, k, max_iters=100):
    np.random.seed(42)
    initial_centroids_idx = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[initial_centroids_idx, :]
    
    for _ in range(max_iters):
        labels = np.array([np.argmin([euclidean_distance(x, c) for c in centroids]) for x in X])
        new_centroids = calculate_centroids(X, labels, k)
        
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids

# Função para calcular os índices
def calculate_indices(X, labels):
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    
    # Índice de Dunn
    min_dist_between_clusters = np.inf
    max_diameter_within_clusters = -np.inf
    unique_labels = np.unique(labels)
    
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            dist_clusters = euclidean_distance(X[labels == unique_labels[i]].mean(axis=0), 
                                               X[labels == unique_labels[j]].mean(axis=0))
            if dist_clusters < min_dist_between_clusters:
                min_dist_between_clusters = dist_clusters
        
        diameter_cluster_i = np.max([euclidean_distance(x, X[labels == unique_labels[i]].mean(axis=0)) for x in X[labels == unique_labels[i]]])
        if diameter_cluster_i > max_diameter_within_clusters:
            max_diameter_within_clusters = diameter_cluster_i
    
    dunn_index = min_dist_between_clusters / max_diameter_within_clusters
    return silhouette, davies_bouldin, calinski_harabasz, dunn_index

# Funções para calcular as métricas
def silhouette_score(X, labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1
    
    from sklearn.metrics import pairwise_distances
    A = np.zeros(X.shape[0])
    B = np.zeros(X.shape[0])
    
    D = pairwise_distances(X, X)
    
    for i in range(len(unique_labels)):
        mask = labels == unique_labels[i]
        for j in range(len(unique_labels)):
            if i != j:
                mask_other = labels == unique_labels[j]
                B[mask] = np.minimum(B[mask], D[mask][:, mask_other].mean(axis=1))
    
    for i in range(len(unique_labels)):
        mask = labels == unique_labels[i]
        A[mask] = D[mask][:, mask].mean(axis=1)
    
    S = (B - A) / np.maximum(A, B)
    return np.mean(S)

def davies_bouldin_score(X, labels):
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    centroids = calculate_centroids(X, labels, k)
    intra_cluster_dists = np.zeros(k)
    
    for i in range(k):
        intra_cluster_dists[i] = np.mean([euclidean_distance(x, centroids[i]) for x in X[labels == unique_labels[i]]])
    
    db_score = 0
    for i in range(k):
        max_ratio = 0
        for j in range(k):
            if i != j:
                dist = euclidean_distance(centroids[i], centroids[j])
                ratio = (intra_cluster_dists[i] + intra_cluster_dists[j]) / dist
                if ratio > max_ratio:
                    max_ratio = ratio
        db_score += max_ratio
    
    db_score /= k
    return db_score

def calinski_harabasz_score(X, labels):
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    n = X.shape[0]
    centroids = calculate_centroids(X, labels, k)
    overall_mean = X.mean(axis=0)
    
    between_cluster_dispersion = np.sum([len(X[labels == unique_labels[i]]) * euclidean_distance(centroids[i], overall_mean)**2 for i in range(k)])
    within_cluster_dispersion = np.sum([np.sum([euclidean_distance(x, centroids[labels[i]])**2 for x in X[labels == unique_labels[i]]]) for i in range(k)])
    
    ch_score = (between_cluster_dispersion / (k - 1)) / (within_cluster_dispersion / (n - k))
    return ch_score

# Função para rodar o K-means e calcular os índices
def kmeans_algorithm(X, k):
    labels, _ = kmeans(X, k)
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