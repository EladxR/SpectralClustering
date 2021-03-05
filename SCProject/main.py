import kmeans_pp
import numpy as np
from sklearn.datasets import make_blobs
import argparse
import random
import math
import matplotlib.pyplot as plt

maximum_capacity_n = 215
maximum_capacity_k = 10


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
    k = np.argmax(delta[:math.floor(n + 1 / 2)]) + 1
    Qc = Qc[:n, :]
    U = Qc[:, :k]  # take the first k eigenvectors which they are already sorted by their eigenvalues
    return U, k


def ComputeT(U):
    temp = np.sqrt(np.sum(np.square(U), axis=1))
    T = np.divide(U, temp[:, np.newaxis])
    return T


def NormalizedSpectralClustering(A, Random, inputK, n):  # A- nXd
    W = FormW(A, n)
    Lnorm = initLnorm(W, n)
    (U, k) = ComputeUnK(Lnorm, n)
    T = ComputeT(U)
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
    return resultsSpectral, resKmeans, K


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


def drawClusterColors2D(results, Colors, observations):
    for i in range(len(results)):
        cluster = results[i]
        xCluster = [observations[index, 0] for index in cluster]
        yCluster = [observations[index, 1] for index in cluster]
        plt.scatter(xCluster, yCluster, color=Colors[i])


def drawClusterColors3D(results, Colors, observations, ax):
    for i in range(len(results)):
        cluster = results[i]
        xCluster = [observations[index, 0] for index in cluster]
        yCluster = [observations[index, 1] for index in cluster]
        zCluster = [observations[index, 2] for index in cluster]

        ax.scatter(xCluster, yCluster, zCluster, color=Colors[i])


def CreateDescriptiveInformation(K, N, Kinput, jacSpectral, jacKmeans):
    info = "Data was generated from the values:\n"
    info += "n = " + str(N) + " , k = " + str(Kinput) + "\n"
    info += "The k the was used for both algorithms was " + str(K) + "\n"
    info += "The Jaccard measure for Spectral Clustering: " + str(jacSpectral) + "\n"
    info += "The Jaccard measure for K-means: " + str(jacKmeans) + "\n"
    return info


def CreateClustersPdf(labels, observations, resultsSpectral, resKmeans, N, d, K, Kinput):
    jacSpectral = ComputeJaccardMeasure(labels, resultsSpectral, N)
    jacKmeans = ComputeJaccardMeasure(labels, resKmeans, N)
    descInfo = CreateDescriptiveInformation(K, N, Kinput, jacSpectral, jacKmeans)
    f = plt.figure()
    Colors = plt.cm.viridis(np.linspace(0, 1, K), alpha=0.8)
    if d == 2:
        plt.subplot(1, 2, 1)
        drawClusterColors2D(resultsSpectral, Colors, observations)
        plt.title("Normalized Spectral Clustering")
        plt.subplot(1, 2, 2)
        drawClusterColors2D(resKmeans, Colors, observations)
        plt.title("K-means")
    else:  # d==3
        ax = f.add_subplot(121, projection='3d')
        drawClusterColors3D(resultsSpectral, Colors, observations, ax)
        plt.title("Normalized Spectral Clustering")
        ax = f.add_subplot(122, projection='3d')
        drawClusterColors3D(resKmeans, Colors, observations, ax)
        plt.title("K-means")

    plt.figtext(0.5, 0.01, descInfo, ha="center", fontsize=18, va='top')

    f.savefig("clusters.pdf", bbox_inches="tight")


def CheckInput(K, N, Random):
    if K > N and not Random:
        print("input K must be <= N when Random is false (using input K)")
        exit(0)
    if K <= 0 or N <= 0:
        print("input K or N is not a positive number")
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
print(d)

CheckInput(K, N, Random)
print(Random)
if Random:
    K = random.randint(maximum_capacity_k / 2, maximum_capacity_k + 1)
    N = random.randint(maximum_capacity_n / 2, maximum_capacity_n + 1)

(observations, labels) = make_blobs(N, d, centers=K)

CreateDataTxt(observations, N, d)

Kinput = K
# also update the K to the one used in both algorithms
(resultsSpectral, resKmeans, K) = CreateClustersTxt(observations, Random, K, N, d)

CreateClustersPdf(labels, observations, resultsSpectral, resKmeans, N, d, K, Kinput)
