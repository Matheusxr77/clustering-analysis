# Chamada das bibliotecas
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin_min
from deap import base, creator, tools, algorithms

# Configuração do problema
n_clusters = 3
data, _ = make_blobs(n_samples=300, centers=n_clusters, cluster_std=0.60, random_state=0)

# Definição das funções de avaliação
def evaluate(individual):
    clusters = np.unique(individual)
    if len(clusters) < 2:
        return float('inf'), float('inf')

    # Calculando as distâncias intra-cluster
    intra_distances = []
    for cluster in clusters:
        points = data[individual == cluster]
        if len(points) > 1:
            intra_distances.append(np.mean(pairwise_distances_argmin_min(points, points)[1]))
    
    intra_cluster_distance = np.mean(intra_distances)

    # Calculando as distâncias inter-cluster
    inter_distances = []
    for i, cluster1 in enumerate(clusters[:-1]):
        points1 = data[individual == cluster1]
        for cluster2 in clusters[i+1:]:
            points2 = data[individual == cluster2]
            inter_distances.append(np.mean(pairwise_distances_argmin_min(points1, points2)[1]))
    
    inter_cluster_distance = np.mean(inter_distances)
    
    return intra_cluster_distance, -inter_cluster_distance

# Setup do algoritmo evolutivo
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("indices", np.random.permutation, len(data))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.1)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# Execução do algoritmo
population = toolbox.population(n=50)
ngen, cxpb, mutpb = 50, 0.7, 0.2

algorithms.eaMuPlusLambda(population, toolbox, mu=50, lambda_=100, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                          stats=None, halloffame=None, verbose=True)

# Análise dos resultados
best_individual = tools.selBest(population, 1)[0]
clusters = np.unique(best_individual)

print(f"Best individual: {best_individual}")
print(f"Number of clusters: {len(clusters)}")