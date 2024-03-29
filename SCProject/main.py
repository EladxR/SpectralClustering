"""
This module is the main module of the project- used as the glue for the entire project
"""
import kmeans_pp
from normalized_spectral import NormalizedSpectralClustering
import create_output_files
from sklearn.datasets import make_blobs
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

maximum_capacity_n3d = 360
maximum_capacity_k3d = 20
maximum_capacity_n2d = 400
maximum_capacity_k2d = 20


def CheckInput(K, N, Random):
    """
        input:  int K, int N, boolean Random
        checking the input is legal
        """
    if Random:
        return
    if K > N:
        print("input K must be <= N when Random is false (using input K)")
        exit(0)

    if K == -1 or N == -1:
        print("Random is false and didnt enter k or n")
        exit(0)
    if K <= 0 or N <= 0:
        print("input K or N is not a positive number")
        exit(0)


def mainAlgorithm():
    # init arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("K", type=int, help="K")
    parser.add_argument("N", type=int, help="N")
    parser.add_argument("--Random", default=True, action='store_false', help="Random")

    args = parser.parse_args()
    K = args.K
    N = args.N
    Random = args.Random
    d = random.choice([2, 3])
    d = 2

    CheckInput(K, N, Random)
    if Random:
        if d == 2:
            K = random.randint(int(maximum_capacity_k2d / 2), maximum_capacity_k2d + 1)
            N = random.randint(int(maximum_capacity_n2d / 2), maximum_capacity_n2d + 1)
        else:
            K = random.randint(int(maximum_capacity_k3d / 2), maximum_capacity_k3d + 1)
            N = random.randint(int(maximum_capacity_n3d / 2), maximum_capacity_n3d + 1)

    # set random points
    (observations, labels) = make_blobs(N, d, centers=K)

    create_output_files.CreateDataTxt(observations, labels)  # create data.txt

    Kinput = K  # save the original K input

    # run the 2 algorithms and update the K to the one used in both algorithms
    (K, resultsSpectral) = NormalizedSpectralClustering(observations, Random, K, N)

    resKmeans = kmeans_pp.k_means_pp(K, N, d, observations)

    # create clusters.txt file
    create_output_files.CreateClustersTxt(resultsSpectral, resKmeans, K)

    # create clusters.pdf
    create_output_files.CreateClustersPdf(labels, observations, resultsSpectral, resKmeans, N, d, K, Kinput)


# main:
mainAlgorithm()
