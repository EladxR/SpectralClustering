"""
This module is responsible for creation of data file, clusters file and visual pdf file
"""

import numpy as np
import matplotlib.pyplot as plt


def matrixToString(A):
    """
        input: matrix A
        output: string of the matrix A- each row in separated line
        """
    listOfStrings = []
    for lst in A:
        stringList = [str(index) for index in lst]
        listOfStrings.append(','.join(stringList))
    return "\n".join(listOfStrings)


def dataToString(observations, labels):
    """
        input: observations and labels from make blobs
        output: string of the matrix A- each row in separated line and label in the end of the line
        """
    listOfStrings = []
    for i in range(len(observations)):
        ob = observations[i]
        stringList = [str(num) for num in ob]
        stringList.append(str(labels[i]))
        listOfStrings.append(','.join(stringList))

    return "\n".join(listOfStrings)


def CreateClustersTxt(resultsSpectral, resKmeans, K):
    """
        input: resultsSpectral/resKmeans List of K lists, each list contain the indexes of the observation in the cluster
                int K- the K used for both algorithms
        creating clusters.txt file as described
        """

    f = open("clusters.txt", "w")
    f.write(str(K) + "\n")  # first line is the K used for both algorithms
    f.write(matrixToString(resultsSpectral) + "\n")
    f.write(matrixToString(resKmeans))
    f.close()


def CreateDataTxt(observations, labels):
    """
        input: matrix observations NXd
        creating data.txt file in the format described
            """
    f = open("data.txt", "w")
    f.write(dataToString(observations, labels))
    f.close()


def ComputeJaccardMeasure(labels, results, N):
    """
        input:  labels- array indicates the cluster of each observation from the generated make_blobs
                results List of K lists, each list contain the indexes of the observation in the cluster
                int N- number of observations
        output: the jaccard measure for the results
        """
    labelResults = np.zeros(N)
    for i in range(len(results)):
        cluster = results[i]
        for index in cluster:
            labelResults[index] = i

    counterMone = 0  # count all pairs clustered together in make blobs and kmeans
    counterMehane = 0
    for i in range(N):
        for j in range(i + 1, N):
            if labels[i] == labels[j] and labelResults[i] == labelResults[j]:
                counterMone += 1
            if labels[i] == labels[j] or labelResults[i] == labelResults[j]:
                counterMehane += 1
    if counterMehane == 0:
        return 1
    return counterMone / counterMehane


def drawClusterColors2D(results, Colors, observations):
    """
         input: results List of K lists, each list contain the indexes of the observation in the cluster
                 Colors- List of colors
                 observations - the nXd observations matrix
         scatter points in the plot with different color for each cluster
         """
    for i in range(len(results)):
        cluster = results[i]
        xCluster = [observations[index, 0] for index in cluster]
        yCluster = [observations[index, 1] for index in cluster]
        plt.scatter(xCluster, yCluster, color=Colors[i])


def drawClusterColors3D(results, Colors, observations, ax):
    """
            input: results List of K lists, each list contain the indexes of the observation in the cluster
                    Colors- List of colors
                    observations - the nXd observations matrix
                    ax- the 3d plot
            scatter points in the plot with different color for each cluster
            """
    for i in range(len(results)):
        cluster = results[i]
        xCluster = [observations[index, 0] for index in cluster]
        yCluster = [observations[index, 1] for index in cluster]
        zCluster = [observations[index, 2] for index in cluster]

        ax.scatter(xCluster, yCluster, zCluster, color=Colors[i])


def CreateDescriptiveInformation(K, N, Kinput, jacSpectral, jacKmeans):
    """
        input: int K, int N, jacSpectral, jacKmeans
        output: string of the Descriptive Information of the visual output
        """
    info = "Data was generated from the values:\n"
    info += "n = " + str(N) + " , k = " + str(Kinput) + "\n"
    info += "The k the was used for both algorithms was " + str(K) + "\n"
    info += "The Jaccard measure for Spectral Clustering: " + str(jacSpectral) + "\n"
    info += "The Jaccard measure for K-means: " + str(jacKmeans) + "\n"
    return info


def CreateClustersPdf(labels, observations, resultsSpectral, resKmeans, N, d, K, Kinput):
    """
        input:  labels- array indicates the cluster of each observation from the generated make_blobs
                observations- matrix NXd
                resultsSpectral/resKmeans List of K lists, each list contain the indexes of the observation in the cluster
                int N, int d, int K ,int Kinput
        creating the clusters.pdf with visual presentation and descriptive information
        """
    # compute jaccard measure for both algorithms
    jacSpectral = ComputeJaccardMeasure(labels, resultsSpectral, N)
    jacKmeans = ComputeJaccardMeasure(labels, resKmeans, N)
    descInfo = CreateDescriptiveInformation(K, N, Kinput, jacSpectral, jacKmeans)

    f = plt.figure()
    Colors = plt.cm.viridis(np.linspace(0, 1, K), alpha=0.8)  # init array of K different colors
    if d == 2:
        # draw first graph:
        plt.subplot(1, 2, 1)
        drawClusterColors2D(resultsSpectral, Colors, observations)
        plt.title("Normalized Spectral Clustering")
        # draw second graph:
        plt.subplot(1, 2, 2)
        drawClusterColors2D(resKmeans, Colors, observations)
        plt.title("K-means")
    else:  # d==3
        # draw first graph:
        ax = f.add_subplot(121, projection='3d')
        drawClusterColors3D(resultsSpectral, Colors, observations, ax)
        plt.title("Normalized Spectral Clustering")
        # draw second graph:
        ax = f.add_subplot(122, projection='3d')
        drawClusterColors3D(resKmeans, Colors, observations, ax)
        plt.title("K-means")

    # add below the descriptive information
    plt.figtext(0.5, 0.01, descInfo, ha="center", fontsize=18, va='top')

    f.savefig("clusters.pdf", bbox_inches="tight")  # save figure into pdf
