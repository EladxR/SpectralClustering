import numpy as np
import mykmeanssp as km


def k_means_pp(K, N, d, observation):
    np.random.seed(0)

    MAX_ITER = 300

    centroids = np.zeros(K, np.int64)
    centroids[0] = np.random.randint(0, N)

    for j in range(1, K):
        distances = np.zeros(N)
        for i in range(0, N):
            D_i = calculate_D(observation, i, j, centroids)
            distances[i] = D_i
        sum_distances = np.sum(distances)
        probs = distances / sum_distances
        centroids[j] = np.random.choice(N, 1, p=probs)

  #  print_cenroids(centroids)
    return km.k_means(K, N, d, MAX_ITER, centroids.tolist(), observation.tolist())


def calculate_D(observation, i, j, cluster):
    minimum = -1
    for z in range(j):
        sub_arr = np.subtract(observation[i], observation[cluster[z]])
        current_d = 0
        for num in sub_arr:
            current_d += (num ** 2)
        if current_d < minimum or minimum == -1:
            minimum = current_d
    return minimum


def check_args(K, N, d, MAX_ITER):
    if K <= 0 or N <= 0 or d <= 0 or MAX_ITER <= 0 or N <= K:
        print("wrong arguments")
        exit(0)


def print_cenroids(centroids):
    for i in range(len(centroids)):
        if i < (len(centroids) - 1):
            print(centroids[i], ",", sep='', end='')
        else:
            print(centroids[i], sep='')
