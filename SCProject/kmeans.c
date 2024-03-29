#define PY_SSIZE_T_CLEAN  /* For all # variants of unit formats (s#, y#, etc.) use Py_ssize_t rather than int. */
#include <Python.h>       /* MUST include <Python.h>, this implies inclusion of the following standard headers:
                             <stdio.h>, <string.h>, <errno.h>, <limits.h>, <assert.h> and <stdlib.h> (if available). */

#include <stdio.h>
#include <stdlib.h>
#include "kmeans.h"

int epsilon = 0.0001;



static PyObject * k_means(int K, int N, int d, int MAX_ITER,int * init_centroids, double ** observation) {
    int i, j, iter, isChanged, k, cenUnchangedCounter;
    double **cen;
    groupItem **groups;

    /* init arrays*/
    cen = initCenArray(K, d);
    groups = initGroups(K);
    if(cen==NULL || groups==NULL){  /* allocation error occurred return NULL will generate the error  */
        return NULL;
    }

    for(i=0;i<K;i++){
        for(j=0;j<d;j++){
            cen[i][j]=observation[init_centroids[i]][j];
        }
    }

    /* main algorithm */
    iter = 0;
    isChanged = 1;

    while (isChanged && iter < MAX_ITER) {
        /*init groups*/
        for (k = 0; k < K; k++) {
             freeList(groups[k]);
             groups[k] = NULL; /* init groups to null for next iteration */
         }
        /*separate to groups*/
        separateToGroups(groups,N,K,d,observation,cen);

        /*update centroids*/
        cenUnchangedCounter= UpdateCentroids(cen,groups,observation,K,d);
        if(cenUnchangedCounter==-1){ /*error occurred*/
            return NULL;
        }

        if (cenUnchangedCounter == K) {
            isChanged = 0;
        }
        iter++;
    }
    PyObject* results=CreateResultsFromGroups(groups,N,K);

    /* free memories */
    freeAllMemories(observation,cen,groups,N,K);
    return results;
}

void freeAllMemories(double **observation, double **cen, groupItem **groups, int N, int K){
    int k;
    freeMemoryArray(observation, N);
    freeMemoryArray(cen, K);
    for (k = 0; k < K; k++) { /* free each group items */
             freeList(groups[k]);
             groups[k] = NULL;
    }
    free(groups);
}

void separateToGroups(groupItem** groups, int N, int K, int d,double** observation,double **cen){
    int i, j, k, kmin;
    double min;
    double distance;
    for (i = 0; i < N; i++) {
            min = -1; /* inf */
            kmin = -1;
            for (k = 0; k < K; k++) {
                distance = 0;
                /* compute distance */
                for (j = 0; j < d; j++) {
                    distance += (observation[i][j] - cen[k][j]) * (observation[i][j] - cen[k][j]);
                }
                if (min == -1 || distance < min) { /* check if smaller distance was found */
                    min = distance;
                    kmin = k;
                }
            }
            groups[kmin] = addToGroup(i, groups[kmin]); /*add the index i (observation i) to the cluster index kmin*/
    }
}

int UpdateCentroids(double **cen,groupItem** groups,double **observation, int K, int d){
/* compute the new centroids according to the clustering */
    int i, k, j, sizeGroup, equalsCounter,cenUnchangedCounter;
    groupItem* item;
    double *sum;
    cenUnchangedCounter=0;
    for (k = 0; k < K; k++) {
            sum = initSumArray(d);
            if(sum==NULL){
            return -1;
            }
            item = groups[k];
            sizeGroup = 0;
            while (item != NULL) {
                i = item->data;
                for (j = 0; j < d; j++) {
                    sum[j] += observation[i][j];
                }
                sizeGroup++;
                item = item->next;
            }
            equalsCounter = d;
            for (j = 0; j < d; j++) {
            /* check if cluster has changed more than epsilon (zero) */
                if (cen[k][j]-(sum[j] / sizeGroup)  > epsilon || cen[k][j]-(sum[j] / sizeGroup)  < -epsilon ) {
                    equalsCounter = 0;
                }
                cen[k][j] = sum[j] / sizeGroup;
            }
            if (equalsCounter == d) {
                cenUnchangedCounter++;
            }
    }
    return cenUnchangedCounter;
}


PyObject *CreateResultsFromGroups(groupItem** groups,int N,int K){
    groupItem * item;
    int i;
    PyObject* data;
    PyObject *list;
    PyObject *listOfLists;
    listOfLists = Py_BuildValue("[]"); /* list containing lists , each list represent cluster */
    for(i=0;i<K;i++){
        list = Py_BuildValue("[]"); /* list of a specific cluster*/
        item=groups[i];
        while(item !=NULL){ /* inserting observations to their cluster list*/
            data=Py_BuildValue("i",item->data);
            PyList_Append(list,data);
            item=item->next;
        }
        PyList_Append(listOfLists,list); /* adding cluster to list of clusters */
    }

    return listOfLists;
}

void freeMemoryArray(double **arr, int size) {
    int i;
    for (i = 0; i < size; i++) {
        free(arr[i]);
    }
    free(arr);
}

double *initSumArray(int d) {
    double *sum = calloc(d, sizeof(double));
    if(sum==NULL){
        PyErr_SetString(PyExc_NameError,"allocation error");
        return NULL;
    }
    return sum;
}

groupItem **initGroups(int K) {
    groupItem **groups = calloc(K, sizeof(groupItem *));
    if(groups==NULL){
        PyErr_SetString(PyExc_NameError,"allocation error in groups array");
    }
    return groups;
}

double **initObsArray(int N, int d) {
    int i;
    double **obs = calloc(N, sizeof(double *));
    if(obs==NULL){
        PyErr_SetString(PyExc_NameError,"allocation error in observation array");
        return NULL;
    }
    for (i = 0; i < N; i++) {
        obs[i] = (double *) calloc(d, sizeof(double));
        if(obs[i]==NULL){
            PyErr_SetString(PyExc_NameError,"allocation error in observation array");
            return NULL;
        }
    }
    return obs;
}

double **initCenArray(int K, int d) {
    int i;
    double **cen = calloc(K, sizeof(double *));
    if(cen==NULL){
        PyErr_SetString(PyExc_NameError,"allocation error in centroids array");
        return NULL;
    }
    for (i = 0; i < K; i++) {
        cen[i] = (double *) calloc(d, sizeof(double));
        if(cen[i]==NULL){
            PyErr_SetString(PyExc_NameError,"allocation error in centroids array");
            return NULL;
        }
    }
    return cen;
}

groupItem *addToGroup(int data, groupItem *firstItemGroup) {
/* add item as first in the linked list */
    groupItem *item = (groupItem *) malloc(sizeof(groupItem));
    if(item==NULL){
        PyErr_SetString(PyExc_NameError,"allocation error");
        return NULL;
    }
    item->data = data;
    item->next = firstItemGroup;

    return item;
}

void freeList(groupItem *firstItemGroup){
    if(firstItemGroup==NULL){
        return;
    }
    freeList(firstItemGroup->next);
    free(firstItemGroup);
}


static double** init_observations(PyObject* py_observation,int N, int d){
/* convert Pyobject to matrix NXd og the observations */
    Py_ssize_t i,j;
    int n,obsize;
    PyObject* ob;
    PyObject* coordinate;
    double** observations;
    if(!PyList_Check(py_observation)){
        PyErr_SetString(PyExc_NameError,"not list at all!!!!!");
        return NULL;
    }
    n = PyList_Size(py_observation);
    if(n!=N){
        PyErr_SetString(PyExc_NameError,"size doesnt match!!!!!");
        return NULL;
    }
    observations=initObsArray(N,d);
    if(observations==NULL){
        return NULL;
    }
    for(i=0;i<n;i++){
        ob=PyList_GetItem(py_observation,i);
        if(!PyList_Check(ob)){
            PyErr_SetString(PyExc_NameError,"not list of lists!!!");
            return NULL;
        }
        obsize=PyList_Size(ob);
        if(obsize!=d){
             PyErr_SetString(PyExc_NameError,"size of observation doesnt match!!!!!");
             return NULL;
        }
        for(j=0;j<obsize;j++){
            coordinate=PyList_GetItem(ob,j);
            if(!PyFloat_Check(coordinate)){
                PyErr_SetString(PyExc_NameError,"not a double!!!!!");
                return NULL;
            }
            observations[i][j]=PyFloat_AsDouble(coordinate);
        }
    }
    return observations;
}

static int* init_index_centroids(PyObject* py_init_centroids,int K){
/* convert Pyobject to array of size K if the init centroids */

    Py_ssize_t i;
    int n;
    PyObject* item;
    int* init_centroids;
    if(!PyList_Check(py_init_centroids)){
        PyErr_SetString(PyExc_NameError,"not list at all!");
        return NULL;
    }
    n = PyList_Size(py_init_centroids);
    if(n!=K){
        PyErr_SetString(PyExc_NameError,"size doesnt match!");
        return NULL;
    }
    init_centroids=(int*)calloc(n,sizeof(int));
    if(init_centroids==NULL){
         PyErr_SetString(PyExc_NameError,"allocation error");
         return NULL;
    }
    for(i=0;i<n;i++){
        item=PyList_GetItem(py_init_centroids,i);
        if(!PyLong_Check(item)){
            PyErr_SetString(PyExc_NameError,"not an integer!");
            return NULL;
        }
        init_centroids[i]=(int)PyLong_AsLong(item);
    }
    return init_centroids;
}

/*
 * The wrapping function needs a PyObject* self argument.
 * This is a requirement for all functions and methods in the C API.
 * It has input PyObject *args from Python.
 */
static PyObject* k_means_capi(PyObject *self, PyObject *args)
{
    int K;
    int N;
    int d;
    int MAX_ITER;
    PyObject * py_init_centroids;
    PyObject * py_observation;
    /* This parses the Python arguments into a double (d)  variable named z and int (i) variable named n*/
    if(!PyArg_ParseTuple(args, "iiiiOO", &K, &N,&d,&MAX_ITER, &py_init_centroids,&py_observation)) {
        return NULL; /* In the CPython API, a NULL value is never valid for a
                        PyObject* so it is used to signal that an error has occurred. */
    }
    double** observation = init_observations(py_observation,N,d);
    int* init_centroids = init_index_centroids(py_init_centroids,K);
    if(observation==NULL || init_centroids==NULL){
        return NULL;
    }

    return k_means(K, N, d, MAX_ITER, init_centroids, observation);
}

/*
 * This array tells Python what methods this module has.
 * We will use it in the next structure
 */
static PyMethodDef capiMethods[] = {
    {"k_means",                   /* the Python method name that will be used */
      (PyCFunction) k_means_capi, /* the C-function that implements the Python function and returns static PyObject*  */
      METH_VARARGS,           /* flags indicating parametersaccepted for this function */
      PyDoc_STR("k means algorithm!!!!!!!!!!")}, /*  The docstring for the function */
    {NULL, NULL, 0, NULL}     /* The last entry must be all NULL as shown to act as a
                                 sentinel. Python looks for this entry to know that all
                                 of the functions for the module have been defined. */
};


/* This initiates the module using the above definitions. */

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    capiMethods /* the PyMethodDef array from before containing the methods of the extension */
};


/*
 * The PyModuleDef structure, in turn, must be passed to the interpreter in the module’s initialization function.
 * The initialization function must be named PyInit_name(), where name is the name of the module and should match
 * what we wrote in struct PyModuleDef.
 * This should be the only non-static item defined in the module file
 */

PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}


