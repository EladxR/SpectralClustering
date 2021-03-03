import kmeans_pp
import numpy as np
from sklearn.datasets import make_blobs
import argparse
import random
import math
import matplotlib.pyplot as plt

maximum_capacity_n = 10
maximum_capacity_k = 4


def GramSchmidt(A):
    U = np.copy(A)
    n = np.size(A, 0)
    R = np.zeros(np.shape(A), np.float64)
    Q = np.zeros(np.shape(A), np.float64)
    for i in range(n):
        R[i, i] = np.linalg.norm(U[:, i])
        if R[i, i] == 0:  # check divide by zero
            print("error divide by zero in Gram Schmidt")
            exit(0)
        Q[:, i] = (1 / R[i, i]) * U[:, i]

        for j in range(i + 1, n):
            R[i, j] = ((np.transpose(Q[:, i])) @ U[:, j])
            U[:, j] = np.subtract(U[:, j], R[i, j] * Q[:, i])

    return Q, R


def QRIterationAlgorithm(A):
    Ac = np.copy(A)
    n = np.size(A, 0)
    Qc = np.eye(n)
    e = 0.0001
    for i in range(n):
        (Q, R) = GramSchmidt(Ac)
        Ac = R @ Q
        newQ = Qc @ Q
        dist = np.abs(Qc) - np.abs(newQ)
        if np.all(np.abs(dist) <= e):
            return (Ac, Qc)
        Qc = newQ
    return (Ac, Qc)


def FormW(A, n):
    W = np.zeros((n, n), np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:  # if they are equal its already set to zero
                dist = np.linalg.norm(A[i] - A[j])
                W[i, j] = np.float_power(np.e, (-dist / 2))
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
    Qc = np.concatenate((Qc, np.array([eigenvalues])), axis=0)  # adding eigenvalues to last row
    Qc = Qc[:, Qc[n].argsort()]  # sorting columns ny last row
    eigenvalues = Qc[n]  # last row of eigenvalues is now sorted
    delta = np.abs(np.diff(eigenvalues))  # calculating The Eigengap Heuristic
    k = np.argmax(delta[:math.floor(n+1 / 2)]) + 1
    Qc = Qc[:n, :]
    U = Qc[:, :k]  # take the first k eigenvectors which they are already sorted by their eigenvalues
    return U, k


def ComputeT(U, n, k):
    temp = np.sqrt(np.sum(np.square(U), axis=1))
    # zeroes = np.zeros(n, k)
    # temp = temp[:, np.newaxis] + zeroes  # broadcasting temp to U's shape in order to compute T
    T = np.divide(U, temp[:, np.newaxis])
    return T


def NormalizedSpectralClustering(A, Random, inputK, n):  # A- nXd
    # n = np.shape(A)[0]
    W = FormW(A, n)
    Lnorm = initLnorm(W, n)
    (U, k) = ComputeUnK(Lnorm, n)
    T = ComputeT(U, n, k)
    d = k
    if not Random:  # if not random use both algorithms the input K o.w use calculated k
        k = inputK
    resultsSpectral = kmeans_pp.k_means_pp(k, n, d, T)

    return (k, resultsSpectral)


def matrixToString(A):
    listOfStrings = []
    for lst in A:
        stringList = [str(index) for index in lst]
        listOfStrings.append(','.join(stringList))
    return "\n".join(listOfStrings)


def CreateClustersTxt(observations, Random, K, N, d):
    (K, resultsSpectral) = NormalizedSpectralClustering(observations, Random, K, N)
    resKmeans = kmeans_pp.k_means_pp(K, N, d, observations)
    f = open("clusters.txt", "w")
    f.write(str(K) + "\n")
    f.write(matrixToString(resultsSpectral) + "\n")
    f.write(matrixToString(resKmeans))
    f.close()
    return resultsSpectral, resKmeans


def CreateDataTxt(observations, N, d):
    f = open("data.txt", "w")
    f.write(matrixToString(observations))
    f.close()


def ComputeJaccardMeasure(labels, results, N):
    counter = 0
    i = 0
    for label in labels:
        if label < len(results):
            if i in results[label]:
                counter += 1
        i += 1
    return counter / N


def CreateClustersPdf(labels, centers, observations, resultsSpectral, resKmeans, N, d, K):
    jacSpectral = ComputeJaccardMeasure(labels, resultsSpectral, N)
    jacKmeans = ComputeJaccardMeasure(labels, resKmeans, N)
    if (d == 2):
        plt.subplot(1,2,1)
        plt.scatter(observations[:,0],observations[:,1],)


def CheckInput(K, N, Random):
    if K > N and not Random:
        print("input K must be <= N when Random is false (using input K)")
        exit(0)


# main:
parser = argparse.ArgumentParser()
parser.add_argument("K", type=int, help="K")
parser.add_argument("N", type=int, help="N")
parser.add_argument("--Random", default=True, action='store_false', help="Random")

args = parser.parse_args()
K = args.K
N = args.N
Random = args.Random
d = random.choice([2, 3])

CheckInput(K, N, Random)

if Random:
    K = random.randint(maximum_capacity_k / 2, maximum_capacity_k + 1)
    N = random.randint(maximum_capacity_n / 2, maximum_capacity_n + 1)

(observations, labels, centers) = make_blobs(N, d, centers=K)

CreateDataTxt(observations, N, d)

(resultsSpectral, resKmeans) = CreateClustersTxt(observations, Random, K, N, d)

CreateClustersPdf(labels, centers, observations, resultsSpectral, resKmeans, N, d, K)
