#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <mkl_cblas.h>

#include <omp.h>

static PyObject * pykmeans_kmeans(PyObject * self, PyObject * args, PyObject * kwargs)
{
    PyArrayObject * pydata;
    PyArrayObject * pycentroids = NULL;
    unsigned long long K;
    int iters;
    float tau;

    static char * kwlist[] = {"data", "k", "iters", "epsilon", "centroids", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!Lif|O!", kwlist, &PyArray_Type, &pydata, &K, &iters, &tau, &PyArray_Type, &pycentroids)) 
        return NULL;

    fprintf(stderr, "K=%llu, iters=%d\n", K, iters);

    if (PyArray_NDIM(pydata) != 2) {
        PyErr_SetString(PyExc_TypeError, "data must be 2 dimensional");
        return NULL;
    }

    if (PyArray_DESCR(pydata)->type_num != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "data must be of type float32");
        return NULL;
    }

    const npy_intp * dims = PyArray_DIMS(pydata);
    unsigned long long N = dims[0];
    unsigned long long M = dims[1];

    if (K < 0 || K > N) {
        PyErr_SetString(PyExc_Exception, "k must be positive and less than the number of features");
        return NULL;
    }

    if (iters < 1) {
        PyErr_SetString(PyExc_Exception, "iters must be greater than 0");
        return NULL;
    }

    if (pycentroids) {
        if (PyArray_NDIM(pycentroids) != 2) {
            PyErr_SetString(PyExc_TypeError, "centroids must be 2 dimensional");
            return NULL;
        }

        if (PyArray_DESCR(pycentroids)->type_num != NPY_FLOAT) {
            PyErr_SetString(PyExc_TypeError, "centroids must be of type float32");
            return NULL;
        }

        const npy_intp * centroid_dims = PyArray_DIMS(pycentroids);
        if (centroid_dims[0] != K) {
            PyErr_SetString(PyExc_TypeError, "centroids must contain K rows");
            return NULL;
        }

        if (centroid_dims[1] != M) {
            PyErr_SetString(PyExc_TypeError, "centroids must have the same feature dimensionality as data");
            return NULL;
        }
    }

    int needs_init = !pycentroids;

    // n x K
    // compute distances for each feature, put all K contiguously
    float * distances = calloc(omp_get_max_threads()*K, sizeof(float));
    assert(distances);
    float * tmp = calloc(omp_get_max_threads()*M, sizeof(float));
    fprintf(stderr, "%d max threads\n", omp_get_max_threads());
    assert(tmp);
    int * nassign = calloc(K, sizeof(int));
    assert(nassign);
    float * convergence = calloc(K, sizeof(float));
    assert(convergence);
   
    npy_intp pyassign_dims[1];
    pyassign_dims[0] = N; 
    PyArrayObject * pyassign = (PyArrayObject*)PyArray_SimpleNew((npy_intp)1, pyassign_dims, NPY_INT);
    assert(pyassign);

    npy_intp pycentroid_dims[2];
    pycentroid_dims[0] = K;
    pycentroid_dims[1] = M;
    if (needs_init)
        pycentroids = (PyArrayObject*)PyArray_SimpleNew((npy_intp)2, pycentroid_dims, NPY_FLOAT);
    assert(pycentroids);
    PyArrayObject * pycentroids_bak = (PyArrayObject*)PyArray_SimpleNew((npy_intp)2, pycentroid_dims, NPY_FLOAT);
    npy_intp * centroid_strides = PyArray_STRIDES(pycentroids);

    npy_intp * strides = PyArray_STRIDES(pydata);

    int inc_data = strides[1]/sizeof(float);
    int inc_centroids = centroid_strides[1]/sizeof(float);

    unsigned long long i;
    if (needs_init) {
        // random init
        int * R = calloc(K, sizeof(int));
        assert(R);
        for (i = 0; i < K; ++i) {
            R[i] = i;
        }
        for (i = K+1; i < N; ++i) {
            unsigned long long j = rand()/(RAND_MAX + 1.0f)*i;
            if (j < K)
                R[j] = i;
        }
        #pragma omp parallel for
        for (i = 0; i < K; ++i) {
            cblas_scopy(
                M, 
                (float*)PyArray_GETPTR2(pydata, R[i], 0), 
                inc_data,
                (float*)PyArray_GETPTR2(pycentroids, i, 0), 
                inc_centroids
            );
        }
        free(R);
    }

    float delta = 0;

    assert(inc_centroids == 1);
    for (i = 0; i < iters; ++i) {
        fprintf(stderr, "Iter %llu\n", i);

        fprintf(stderr, "Mapper\n");

        // compute distance for each input feature to all centroids in parallel
        unsigned long long n;
        #pragma omp parallel for
        for (n = 0; n < N; ++n) {
            int thread_id = omp_get_thread_num();
            float * local_distances = distances + thread_id*K;
            float * local_tmp = tmp + thread_id*M;
            unsigned long long k;
            for (k = 0; k < K; ++k) {
                memcpy((void*)local_tmp, PyArray_GETPTR2(pycentroids, k, 0), M*sizeof(float));
                cblas_saxpy(
                    M, 
                    -1, 
                    PyArray_GETPTR2(pydata, n, 0), 
                    inc_data,
                    local_tmp, 
                    inc_centroids
                );
                local_distances[k] = cblas_snrm2(M, local_tmp, inc_centroids);
            }
            *(int*)PyArray_GETPTR1(pyassign, n) = cblas_isamin(K, local_distances, 1);
        }

        fprintf(stderr, "Reducer\n");

        PyArrayObject * tmp_array = pycentroids_bak;
        pycentroids_bak = pycentroids;
        pycentroids = tmp_array;

        memset(nassign, 0, sizeof(int)*K);
        memset(PyArray_DATA(pycentroids), 0, sizeof(float)*K*M);

        for (n = 0; n < N; ++n) {
            int assignment = *(int*)PyArray_GETPTR1(pyassign, n);
            cblas_saxpy(
                M, 
                1.0f, 
                (float*)PyArray_GETPTR2(pydata, n, 0), 
                inc_data,
                (float*)PyArray_GETPTR2(pycentroids, assignment, 0),
                inc_centroids
            );
            ++nassign[assignment];
        }
        int k;
        #pragma omp parallel for
        for (k = 0; k < K; ++k) {
            cblas_sscal(M, 1.0f/nassign[k], (float*)PyArray_GETPTR2(pycentroids, k, 0), inc_centroids);

            cblas_saxpy(
                M,
                -1,
                PyArray_GETPTR2(pycentroids, k, 0),
                inc_centroids,
                PyArray_GETPTR2(pycentroids_bak, k, 0),
                inc_centroids
            );

            convergence[k] = cblas_snrm2(M, PyArray_GETPTR2(pycentroids_bak, k, 0), inc_centroids);
        }
        delta = cblas_sasum(K, convergence, 1);
        fprintf(stderr, "Delta: %f\n", delta);
        if (delta < tau) {
            fprintf(stderr, "Threshold %f reached\n", tau);
            fprintf(stderr, "Terminating\n");
            break;
        }
    }

    free(convergence);
    free(distances);
    free(tmp);

    Py_DECREF(pycentroids_bak);

    return Py_BuildValue("OOd", pycentroids, pyassign, delta);
}

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "pykmeans",
    "A fast and simple main-memory kmeans implementation using OpenMP and blas.",
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if PY_MAJOR_VERSION < 3
static PyMethodDef pykmeans_methods[] = {
    {"kmeans", (PyCFunction)pykmeans_kmeans, METH_VARARGS|METH_KEYWORDS, "kmeans"},
    {NULL}
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
PyInit_pykmeans(void)
#else
PyMODINIT_FUNC
initpykmeans(void)
#endif
{
    import_array();

#if PY_MAJOR_VERSION >= 3
    PyObject * m = PyMOdule_Create(&moduledef);
#else
    Py_InitModule3("pykmeans", pykmeans_methods, "kmeans");
#endif

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
