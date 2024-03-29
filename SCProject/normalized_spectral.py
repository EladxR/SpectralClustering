"""
This module runs the normalized spectral clustering algorithm
contain all the required functions such as: GramSchmidt, QRIterationAlgorithm,
"""

import kmeans_pp
import numpy as np
import math

import time

epsilon = 0.0001


def GramSchmidt(A):
    """
    input: matrix nXn
    output: Q and R matrix as described in Gram Schmidt algorithm
    """
    U = np.copy(A)
    n = np.size(A, 0)  # n is the number or rows
    # init R and Q to zero
    R = np.zeros(np.shape(A), np.float64)
    Q = np.zeros(np.shape(A), np.float64)
    for i in range(n):
        R[i, i] = np.linalg.norm(U[:, i])  # compute l2 norma with numpy

        if R[i, i] != 0:
            Q[:, i] = U[:, i] / R[i, i]

        R[i][i + 1:n] = np.transpose(Q[:, i]) @ U[:, i + 1:n]
        temp = (R[i][:, np.newaxis] * Q[:, i])
        U[:, i + 1:n] = U[:, i + 1:n] - temp.transpose()[:, i + 1:n]

    return Q, R


def QRIterationAlgorithm(A):
    """
        input: matrix nXn
        output: A and Q matrix as described in QRIterationAlgorithm
        """
    Ac = np.copy(A)
    n = np.size(A, 0)  # n is the number or rows
    Qc = np.eye(n)  # init to I matrix
    for i in range(n):
        (Q, R) = GramSchmidt(Ac)
        Ac = R @ Q
        newQ = Qc @ Q
        dist = np.abs(Qc) - np.abs(newQ)  # dist is a matrix nXn with the distances of each cell
        if np.all(np.abs(dist) <= epsilon):  # check if all cells are between [-epsilon,epsilon]
            return Ac, Qc
        Qc = newQ

    return Ac, Qc


def FormW(A, n):
    """
        input: matrix A nXn, int n
        output: matrix W with shape nXn of the weights present the graph
        """

    W = np.zeros((n, n), np.float64)  # init W to zeros
    for i in range(n):
        for j in range(n):
            if i != j:  # if they are equal its already set to zero
                dist = np.linalg.norm(A[i] - A[j])
                W[i, j] = np.float_power(np.e, (-dist / 2))
    return W


def initLnorm(W, n):
    """
        input: matrix W nXn
        output: matrix Lnorm nXn according to the algorithm described
        """
    # init DHalf as the D^(-1/2)
    DHalf = np.zeros((n, n), np.float64)
    for i in range(n):
        DHalf[i, i] = 1 / np.sqrt(np.sum(W[i]))

    # compute Lnorm
    Lnorm = np.eye(n) - DHalf @ W @ DHalf
    return Lnorm


def ComputeUnK(Lnorm, n, inputK, Random):
    """
        input: matrix Lnorm nXn, int n
        output: matrix U nXk represent the first k eigenvector  of Q (from QRIterationAlgorithm)
                int k - the Eigengap Heuristic
        """
    t11=time.time()
    (Ac, Qc) = QRIterationAlgorithm(Lnorm)
    t12=time.time()
    eigenvalues = Ac.diagonal()  # array of the eigenvalues
    Qc = np.concatenate((Qc, np.array([eigenvalues])), axis=0)  # adding eigenvalues to last row
    Qc = Qc[:, Qc[n].argsort()]  # sorting columns ny last row
    eigenvalues = Qc[n]  # last row of eigenvalues is now sorted
    delta = np.abs(np.diff(eigenvalues))  # calculating The Eigengap Heuristic
    if Random:
        k = np.argmax(delta[:math.floor((n) / 2)]) + 1
    else:
        k = inputK
    Qc = Qc[:n, :]
    U = Qc[:, :k]  # take the first k eigenvectors which they are already sorted by their eigenvalues
    print("qr time:"+str(t12-t11))
    return U, k


def ComputeT(U):
    """
        input: matrix U nXk
        output: matrix T nXk as described in NormalizedSpectralClustering
        """
    temp = np.sqrt(
        np.sum(np.square(U), axis=1))  # temp[i] is the sqrt of the sum of row i of U^2 (each cell is squared)
    T = np.divide(U, temp[:, np.newaxis])  # divide each cell in row i in temp[i] using broadcasting
    return T


def NormalizedSpectralClustering(A, Random, inputK, n):
    """
        input: matrix A nXd, boolean Random, int inputK, int n
        output: int k- the k used to calculate Kmeans (according to Random)
                resultsSpectral List of k lists, each list contain the indexes of the observation in the cluster
        """
    W = FormW(A, n)
    Lnorm = initLnorm(W, n)
    Lnorm = np.eye(n) + Lnorm
    (U, k) = ComputeUnK(Lnorm, n, inputK, Random)
    T = ComputeT(U)

    resultsSpectral = kmeans_pp.k_means_pp(k, n, k, T)

    return (k, resultsSpectral)
