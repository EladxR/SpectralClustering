"""
This module generates the first K centroids used for kmeans algorithm in capi- mykmeanssp
(according to hw2)
"""
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
        if sum_distances == 0:
            # when sum of distances is equal zero, all the observation was already chosen before reaching k
            # which means there are less than k different observation in the generated data
            print("generated less than k different observations, please try again")
            exit(0)
        probs = distances / sum_distances
        centroids[j] = np.random.choice(N, 1, p=probs)[0]

    return km.k_means(K, N, d, MAX_ITER, centroids.tolist(), observation.tolist())


def calculate_D(observation, i, j, cluster):
    minimum = -1
    for z in range(j):
        sub_arr = np.subtract(observation[i], observation[cluster[z]])
        current_d = np.sum(np.power(sub_arr, 2))
        if current_d < minimum or minimum == -1:
            minimum = current_d
    return minimum


