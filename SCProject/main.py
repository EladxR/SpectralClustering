"""
This module is the main module of the project- used as the glue for the entire project
"""

import kmeans_pp
import normalized_spectral
import create_output_files
from sklearn.datasets import make_blobs
import argparse
import random

import time

maximum_capacity_n = 275
maximum_capacity_k = 10


def CheckInput(K, N, Random):
    """
        input:  int K, int N, boolean Random
        checking the input is legal
        """
    if K > N and not Random:
        print("input K must be <= N when Random is false (using input K)")
        exit(0)
    if K <= 0 or N <= 0:
        print("input K or N is not a positive number")
        exit(0)


# main:

t0 = time.time()

# informative message:
print("The max capacity of the algorithm is n=" + str(maximum_capacity_n) + " , k=" + str(maximum_capacity_k))

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

CheckInput(K, N, Random)
if Random:
    K = random.randint(int(maximum_capacity_k / 2), maximum_capacity_k + 1)
    N = random.randint(int(maximum_capacity_n / 2), maximum_capacity_n + 1)

# set random points
(observations, labels) = make_blobs(N, d, centers=K)

create_output_files.CreateDataTxt(observations)  # create data.txt

Kinput = K  # save the original K input

# run the 2 algorithms and update the K to the one used in both algorithms
(K, resultsSpectral) = normalized_spectral.NormalizedSpectralClustering(observations, Random, K, N)

resKmeans = kmeans_pp.k_means_pp(K, N, d, observations)

# create clusters.txt file
create_output_files.CreateClustersTxt(resultsSpectral, resKmeans, K)

# create clusters.pdf
create_output_files.CreateClustersPdf(labels, observations, resultsSpectral, resKmeans, N, d, K, Kinput)

t1 = time.time()
print("overall time:" + str(t1 - t0))
