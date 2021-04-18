from invoke import task
import time

maximum_capacity_n3d = 360
maximum_capacity_k3d = 20
maximum_capacity_n2d = 400
maximum_capacity_k2d = 20


@task
def run(c, k=-1, n=-1, Random=True):
    t0 = time.time()
    # informative message:
    print("The max capacity of the algorithm for d=2 is n=" + str(maximum_capacity_n2d) + " , k=" + str(maximum_capacity_k2d))
    print("The max capacity of the algorithm for d=3 is n=" + str(maximum_capacity_n3d) + " , k=" + str(maximum_capacity_k3d))

    # build the kmeans.c
    c.run("python3.8.5 setup.py build_ext --inplace")
    # run main.py
    if Random:
        c.run("python3.8.5 main.py " + str(k) + " " + str(n))
    else:
        c.run("python3.8.5 main.py " + str(k) + " " + str(n) + " --Random")
    t1 = time.time()

    print("time: " + str(t1 - t0))
