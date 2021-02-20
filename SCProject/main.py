import numpy as np
import pandas as pd
import kmeans_pp
from sklearn.datasets import make_blobs
import argparse
import random

maximum_capacity_n = 200
maximum_capacity_k = 10


def GramSchmidt(A):
    U = np.copy(A)
    n = np.size(A, 0)
    R = np.zeros(np.shape(A), np.float64)
    Q = np.zeros(np.shape(A), np.float64)
    for i in range(n):
        R[i, i] = np.linalg.norm(U[:, i])
        Q[:, i] = (1 / R[i, i]) * U[:, i]

        for j in range(i + 1, n):
            R[i, j] = ((np.transpose(Q[:, i])) @ U[:, j])
            U[:, j] = np.subtract(U[:, j], R[i, j] * Q[:, i])
    print(np.transpose(Q) @ Q)

    return (Q, R)


def QRIterationAlgorithm(A):
    Ac = np.copy(A)
    n = np.size(A, 0)
    Qc = np.eye(n)
    e = 0.0001
    for i in range(n):
        (Q, R) = GramSchmidt(Ac)
        Ac = R @ Q
        dist = np.abs(Qc) - np.abs(Qc @ Q)
        if np.all(np.abs(dist) <= e):
            return (Ac, Qc)
        Qc = Qc @ Q
    return (Ac, Qc)


def FormW(A, n):
    W = np.zeros((n, n), np.float64)
    for i in range(n):
        for j in range(n):
            dist = np.linalg.norm(A[i] - A[j])
            W[i, j] = np.e ** (-dist / 2)
    return W


def initLnorm(W, n):
    DHalf = np.zeros((n, n), np.float64)
    for i in range(n):
        DHalf[i, i] = 1 / np.sqrt(np.sum(W[i]))
    Lnorm = np.eye(n) - DHalf @ W @ DHalf
    return Lnorm


def ComputeUnK(Lnorm, n):
    (Ac, Qc) = QRIterationAlgorithm(Lnorm)
    eigenvalues = Ac.diagonal()
    np.sort(eigenvalues)
    delta = np.abs(np.diff(eigenvalues))
    k = np.argmax(delta[:n / 2])
    U = Qc[:, k]
    return (U, k)


def ComputeT(U, n, k):
    T = np.zeros((n, k), np.float64)
    temp = np.sqrt(np.sum(np.square(U), axis=1))
    zeroes = np.zeros(n, k)
    temp = temp[:, np.newaxis] + zeroes
    T = U / temp
    return T


def NormalizedSpectralClustering(A, Random, inputK):  # A- nXd
    n = np.size(A, 0)
    W = FormW(A, n)
    Lnorm = initLnorm(W, n)
    (U, k) = ComputeUnK(Lnorm, n)
    T = ComputeT(U, n, k)
    d = k
    if not Random:  # if random use both algorithms the input K o.w use calculated k
        k = inputK
    kmeans_pp.k_means_pp(k, n, d, T)

    # created cluster.txt

    return k


# main:
parser = argparse.ArgumentParser()
parser.add_argument("K", type=int, help="K")
parser.add_argument("N", type=int, help="N")
parser.add_argument("Random", type=bool, help="Random")

args = parser.parse_args()
K = args.K
N = args.N
Random = args.Random
d = random.choice(2, 3)
if Random:
    K = random.randint(maximum_capacity_k / 2, maximum_capacity_k + 1)
    N = random.randint(maximum_capacity_n / 2, maximum_capacity_n + 1)
Observations = make_blobs(N, d)

temp = NormalizedSpectralClustering(Observations, Random, K)
if Random:
    K = temp

kmeans_pp.k_means_pp(K, N, d, Observations)
