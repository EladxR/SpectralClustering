from invoke import task

maximum_capacity_n = 360
maximum_capacity_k = 20


@task
def run(c, k=-1, n=-1, Random=True):
    # informative message:
    print("The max capacity of the algorithm is n=" + str(maximum_capacity_n) + " , k=" + str(maximum_capacity_k))
    # build the kmeans.c
    c.run("python3.8.5 setup.py build_ext --inplace")
    # run main.py
    if Random:
        c.run("python3.8.5 main.py " + str(k) + " " + str(n))
    else:
        c.run("python3.8.5 main.py " + str(k) + " " + str(n) + " --Random")
