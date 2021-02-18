import numpy as np
import pandas as pd


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


A = np.arange(1.0, 10.0).reshape(3, 3)

print(GramSchmidt(A))

print(QRIterationAlgorithm(A))
print([0.23,0.52,0.818]@A)
