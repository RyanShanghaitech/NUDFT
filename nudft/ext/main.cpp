#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_25_API_VERSION
#include "numpy/arrayobject.h" // Include any other Numpy headers, UFuncs for example.

#include <complex>
#include <omp.h>

typedef std::complex<double> complex;

int GetIdxCoor(int *piCoor, int iNdim, const long *plDim)
{
    int iIdx = piCoor[0];
    for (int i = 1; i < iNdim; ++i)
    {
        iIdx *= plDim[i-1];
        iIdx += piCoor[i];
    }
    return iIdx;
}

static PyObject *
dft(PyObject *self, PyObject *args)
{
    bool bRet;
    PyArrayObject *pyaDataSrc, *pyaCoorSrc, *pyaCoorDst;
    PyLongObject *poInv;
    bRet = PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &pyaDataSrc, &PyArray_Type, &pyaCoorSrc, &PyArray_Type, &pyaCoorDst, &PyLong_Type, &poInv);
    if (!bRet) return nullptr;

    // convert to expected type and arangement
    // it's very important to ensure C_CONTIGUOUS because some stupid asshole don't care the order consistency
    pyaDataSrc = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)pyaDataSrc, NPY_COMPLEX128, NPY_ARRAY_C_CONTIGUOUS);
    pyaCoorSrc = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)pyaCoorSrc, NPY_FLOAT64, NPY_ARRAY_C_CONTIGUOUS);
    pyaCoorDst = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)pyaCoorDst, NPY_FLOAT64, NPY_ARRAY_C_CONTIGUOUS);
    
    // derive shape
    npy_intp *plDimCoorSrc = PyArray_SHAPE(pyaCoorSrc);
    npy_intp *plDimCoorDst = PyArray_SHAPE(pyaCoorDst);

    // derive Npt of input, output and Nax
    int iNptSrc = plDimCoorSrc[0];
    int iNptDst = plDimCoorDst[0];
    int iNdim = plDimCoorSrc[1];

    // get data ptr
    complex *pcDataSrc = (complex*)PyArray_GETPTR1(pyaDataSrc, 0);
    double *pdCoorSrc = (double*)PyArray_GETPTR2(pyaCoorSrc, 0, 0);
    double *pdCoorDst = (double*)PyArray_GETPTR2(pyaCoorDst, 0, 0);

    complex *pcDataDst = new complex[iNptDst];

    // dft loop
    int iInv = PyLong_AsLong((PyObject*)poInv);
    double dSign = iInv ? +1e0 : -1e0;
    double dAng = 0;
    #pragma omp parallel for
    for (int iIdxDst = 0; iIdxDst < iNptDst; ++iIdxDst)
    {
        pcDataDst[iIdxDst] = 0;
        for (int iIdxSrc = 0; iIdxSrc < iNptSrc; ++iIdxSrc)
        {
            dAng = 0;
            for (int iIdxAx = 0; iIdxAx < iNdim; ++iIdxAx)
            {
                dAng += pdCoorSrc[iIdxSrc*iNdim+iIdxAx] * pdCoorDst[iIdxDst*iNdim+iIdxAx];
            }
            dAng *= dSign*2e0*M_PI;
            pcDataDst[iIdxDst] += pcDataSrc[iIdxSrc] * complex(std::cos(dAng), std::sin(dAng)); // std::exp(complex(0,dSign*2e0*M_PI*dAng));
        }
    }

    npy_intp pDims[1] = {iNptDst};
    PyArrayObject *pyaDataDst = (PyArrayObject*)PyArray_SimpleNewFromData(1, pDims, NPY_COMPLEX128, (void*)pcDataDst);
    PyArray_ENABLEFLAGS(pyaDataDst, NPY_ARRAY_OWNDATA);
    return PyArray_Return(pyaDataDst);
}

static PyMethodDef aMeth[] = {
    {"_dft", dft, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef sMod = {
    PyModuleDef_HEAD_INIT,
    "ext",
    NULL,
    -1,
    aMeth
};

PyMODINIT_FUNC
PyInit_ext(void)
{
    import_array();
    return PyModule_Create(&sMod);
}
