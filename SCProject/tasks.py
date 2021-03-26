import time

from invoke import task


@task
def run(c, k, n, Random=True):
    t1 = time.time()
    # build the kmeans.c
    c.run("python3.8.5 setup.py build_ext --inplace")
    t2=time.time()
    # run main.py
    if Random:
        c.run("python3.8.5 main.py " + str(k) + " " + str(n))
    else:
        c.run("python3.8.5 main.py " + str(k) + " " + str(n) + " --Random")
    t3 = time.time()
    print("the real overall time is " + str(t3 - t1))

