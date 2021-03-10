double **initObsArray(int, int);

double **initCenArray(int, int);

typedef struct lst { /*linked list of each cluster  */
    int data;
    struct lst *next;
} groupItem;

groupItem *addToGroup(int data, groupItem *firstItemGroup);

groupItem **initGroups(int k);

double *initSumArray(int d);

void freeMemoryArray(double **, int);

void freeList(groupItem *firstItemGroup);

PyObject *CreateResultsFromGroups(groupItem** groups,int N,int K);

void separateToGroups(groupItem** groups, int N, int K, int d,double** observation,double **cen);
int UpdateCentroids(double **cen,groupItem** groups,double **observation, int K, int d);

void freeAllMemories(double **observation, double **cen, groupItem **groups, int N, int K);