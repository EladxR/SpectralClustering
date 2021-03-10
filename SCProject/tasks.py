from invoke import task


@task
def run(c, k, n, Random=True):
    # build the kmeans.c
    c.run("python3.8.5 setup.py build_ext --inplace")
    # run main.py
    if Random:
        c.run("python3.8.5 main.py " + str(k) + " " + str(n))
    else:
        c.run("python3.8.5 main.py " + str(k) + " " + str(n) + " --Random")
