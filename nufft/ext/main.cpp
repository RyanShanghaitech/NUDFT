#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_25_API_VERSION
#include "numpy/arrayobject.h" // Include any other Numpy headers, UFuncs for example.

#include <complex>
#include <omp.h>
#include <fftw3.h>

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

void Shift(complex *pcDataSrc, int iNdim, const long *plDim, int iInv = 0)
{
    int *piCoorSrc = new int[iNdim];
    memset(piCoorSrc, 0x00, iNdim*sizeof(int));
    int *piCoorDst = new int[iNdim];
    memset(piCoorDst, 0x00, iNdim*sizeof(int));
    int iNpt = 1;
    for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim) iNpt *= plDim[iIdxDim];
    int iIdxSrc, iIdxDst;
    complex *pcDataDst = new complex[iNpt];
    while(1)
    {
        // derive corresponding src coord of dst coord
        for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim)
        {
            long lDim = plDim[iIdxDim];
            int iShift = 0;
            if (!iInv && lDim%2) iShift = lDim/2 + 1;
            else iShift = lDim/2;
            piCoorSrc[iIdxDim] = (piCoorDst[iIdxDim] + iShift)%lDim;
        }
        iIdxDst = GetIdxCoor(piCoorDst, iNdim, plDim);
        iIdxSrc = GetIdxCoor(piCoorSrc, iNdim, plDim);
        pcDataDst[iIdxDst] = pcDataSrc[iIdxSrc];

        // break
        int iBreak = 1;
        for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim)
        {
            if (piCoorDst[iIdxDim] != plDim[iIdxDim] - 1) iBreak = 0;
        }
        if (iBreak) break;

        // update coordinate
        for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim)
        {
            piCoorDst[iIdxDim] += 1;
            if (piCoorDst[iIdxDim] == plDim[iIdxDim]) piCoorDst[iIdxDim] = 0;
            else break;
        }

    }
    memcpy(pcDataSrc, pcDataDst, iNpt*sizeof(complex));

    delete[] piCoorSrc;
    delete[] piCoorDst;
    delete[] pcDataDst;
}

double I0(double dX)
{
    #define NORD_BESSEL (8)
    double dY = 0;
    double dStepMul = 1;
    for (int m = 0; m < NORD_BESSEL; ++m)
    {
        if (m>0) dStepMul *= m;
        dY += 1e0/(dStepMul*dStepMul)*pow(dX/2e0,2*m);
    }
    return dY;
}

double KB(double dU, double dL)
{
    double dB = 2.34*dL; // Fessler
    // double dB = M_PI*dL/2e0; // original KB
    return (1/dL)*I0(dB*sqrt(1 - pow(2*dU/dL,2)));
}

double C(int iNdim, double *dX, double dL)
{
    double dY = 1;
    for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim) dY *= KB(dX[iIdxDim],dL);
    return dY;
}

static PyObject *
dft(PyObject *self, PyObject *args)
{
    bool bRet;
    PyArrayObject *poDataSrc, *poCoorSrc, *poCoorDst;
    PyLongObject *poInv;
    bRet = PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &poDataSrc, &PyArray_Type, &poCoorSrc, &PyArray_Type, &poCoorDst, &PyLong_Type, &poInv);
    if (!bRet) return nullptr;

    // convert to expected type and arangement
    // it's very important to ensure C_CONTIGUOUS because some stupid asshole don't care the order consistency
    poDataSrc = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)poDataSrc, NPY_COMPLEX128, NPY_ARRAY_C_CONTIGUOUS);
    poCoorSrc = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)poCoorSrc, NPY_FLOAT64, NPY_ARRAY_C_CONTIGUOUS);
    poCoorDst = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)poCoorDst, NPY_FLOAT64, NPY_ARRAY_C_CONTIGUOUS);
    
    // derive shape
    volatile int iNdimDataSrc, iNdimCoorSrc, iNdimCoorDst;
    volatile npy_intp *plDimDataSrc, *plDimCoorSrc, *plDimCoorDst;
    iNdimDataSrc = PyArray_NDIM(poDataSrc);
    iNdimCoorSrc = PyArray_NDIM(poCoorSrc);
    iNdimCoorDst = PyArray_NDIM(poCoorDst);
    plDimDataSrc = PyArray_SHAPE(poDataSrc);
    plDimCoorSrc = PyArray_SHAPE(poCoorSrc);
    plDimCoorDst = PyArray_SHAPE(poCoorDst);

    // check shape
    if (iNdimDataSrc != 1)
    {
        PyErr_SetString(PyExc_ValueError, "I must be 1D array.");
        return NULL;
    }
    if (iNdimCoorSrc != 2)
    {
        PyErr_SetString(PyExc_ValueError, "X must be 2D array.");
        return NULL;
    }
    if (plDimDataSrc[0] != plDimCoorSrc[0])
    {
        PyErr_SetString(PyExc_ValueError, "X and I should have the same length.");
        return NULL;
    }
    if (iNdimCoorDst != 2)
    {
        PyErr_SetString(PyExc_ValueError, "K must be 2D array.");
        return NULL;
    }
    if (plDimCoorDst[1] != plDimCoorSrc[1])
    {
        PyErr_SetString(PyExc_ValueError, "K and X should have the same dims.");
        return NULL;
    }

    // derive Npt of input, output and Nax
    int iNptSrc, iNptDst, iNdim;
    iNptSrc = plDimCoorSrc[0];
    iNptDst = plDimCoorDst[0];
    iNdim = plDimCoorSrc[1];

    // get data ptr
    complex *pcDataSrc;
    double *pdCoorSrc, *pdCoorDst;
    pcDataSrc = (complex*)PyArray_GETPTR1(poDataSrc, 0);
    pdCoorSrc = (double*)PyArray_GETPTR2(poCoorSrc, 0, 0);
    pdCoorDst = (double*)PyArray_GETPTR2(poCoorDst, 0, 0);

    complex *pcDataDst = new complex[iNptDst];

    // dft loop
    int iInv = PyLong_AsLong((PyObject*)poInv);
    double dSign = iInv ? +1e0 : -1e0;
    #pragma omp parallel for
    for (int iIdxDst = 0; iIdxDst < iNptDst; ++iIdxDst)
    {
        pcDataDst[iIdxDst] = 0;
        for (int iIdxSrc = 0; iIdxSrc < iNptSrc; ++iIdxSrc)
        {
            double dKX = 0;
            for (int iIdxAx = 0; iIdxAx < iNdim; ++iIdxAx)
            {
                dKX += pdCoorSrc[iIdxSrc*iNdim+iIdxAx] * pdCoorDst[iIdxDst*iNdim+iIdxAx];
            }
            pcDataDst[iIdxDst] += pcDataSrc[iIdxSrc] * std::exp(complex(0,dSign*2e0*M_PI*dKX));
        }
    }

    npy_intp pDims[1] = {iNptDst};
    PyArrayObject *poDataDst = (PyArrayObject*)PyArray_SimpleNewFromData(1, pDims, NPY_COMPLEX128, (void*)pcDataDst);
    PyArray_ENABLEFLAGS(poDataDst, NPY_ARRAY_OWNDATA);
    return PyArray_Return(poDataDst);
}

static PyObject *
ifft(PyObject *self, PyObject *args)
{
    bool bRet;
    PyArrayObject *poDataSrc, *poCoorSrc, *poDimDst;
    bRet = PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &poDataSrc, &PyArray_Type, &poCoorSrc, &PyArray_Type, &poDimDst);
    if (!bRet) return nullptr;

    // ensure the dtype and arrangement
    poDataSrc = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)poDataSrc, NPY_COMPLEX128, NPY_ARRAY_C_CONTIGUOUS);
    poCoorSrc = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)poCoorSrc, NPY_FLOAT64, NPY_ARRAY_C_CONTIGUOUS);
    poDimDst = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)poDimDst, NPY_INT64, NPY_ARRAY_C_CONTIGUOUS);

    // derive shape
    volatile int iNdimDataSrc, iNdimCoorSrc, iNdimDimDst;
    volatile npy_intp *plDimDataSrc, *plDimCoorSrc, *plDimDimDst;
    iNdimDataSrc = PyArray_NDIM(poDataSrc);
    iNdimCoorSrc = PyArray_NDIM(poCoorSrc);
    iNdimDimDst = PyArray_NDIM(poDimDst);
    plDimDataSrc = PyArray_SHAPE(poDataSrc);
    plDimCoorSrc = PyArray_SHAPE(poCoorSrc);
    plDimDimDst = PyArray_SHAPE(poDimDst);

    // check shape
    if (iNdimDataSrc != 1)
    {
        PyErr_SetString(PyExc_ValueError, "DataSrc must be 1D array.");
        return NULL;
    }
    if (iNdimCoorSrc != 2)
    {
        PyErr_SetString(PyExc_ValueError, "CoorSrc must be 2D array.");
        return NULL;
    }
    if (plDimDataSrc[0] != plDimCoorSrc[0])
    {
        PyErr_SetString(PyExc_ValueError, "DataSrc and CoorSrc should have the same length.");
        return NULL;
    }
    if (iNdimDimDst != 1)
    {
        PyErr_SetString(PyExc_ValueError, "DimDst must be 1D array");
        return NULL;
    }
    if (plDimCoorSrc[1] != plDimDimDst[0])
    {
        PyErr_SetString(PyExc_ValueError, "CoorSrc.shape[1] should equal to DimDst.ndim.");
        return NULL;
    }

    // derive Npt of input and Ndim
    int iNptSrc, iNdim;
    iNptSrc = plDimDataSrc[0];
    iNdim = plDimDimDst[0];

    // get data ptr
    complex *pcDataSrc;
    double *pdCoorSrc;
    long *plDimDst;
    pcDataSrc = (complex*)PyArray_GETPTR1(poDataSrc, 0);
    pdCoorSrc = (double*)PyArray_GETPTR2(poCoorSrc, 0, 0);
    plDimDst = (long*)PyArray_GETPTR1(poDimDst, 0);

    int iNptDst = 1;
    for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim) iNptDst *= plDimDst[iIdxDim];
    complex *pcDataDst = new complex[iNptDst];
    memset((void*)pcDataDst, 0x00, iNptDst*sizeof(complex));

    // convert coor from (-0.5,0.5)/pix to (0,dim)/fov
    for (int iIdxCoor = 0; iIdxCoor < iNptSrc; ++iIdxCoor)
    {
        for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim)
        {
            pdCoorSrc[iIdxCoor*iNdim + iIdxDim] += 0.5;
            pdCoorSrc[iIdxCoor*iNdim + iIdxDim] *= plDimDst[iIdxDim];
        }
    }

    // grid loop
    #define KB_WID (6)
    #pragma omp parallel for
    for (int iIdxSrc = 0; iIdxSrc < iNptSrc; ++iIdxSrc)
    {
        int *piCoorDst_Min = new int[iNdim];
        int *piCoorDst_Max = new int[iNdim];
        
        // get min and max coord of KB window
        for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim)
        {
            piCoorDst_Min[iIdxDim] = ceil(pdCoorSrc[iIdxSrc*iNdim + iIdxDim]) - KB_WID/2;
            piCoorDst_Max[iIdxDim] = ceil(pdCoorSrc[iIdxSrc*iNdim + iIdxDim]) + KB_WID/2;
            if (piCoorDst_Min[iIdxDim] < 0) piCoorDst_Min[iIdxDim] = 0;
            if (piCoorDst_Max[iIdxDim] > plDimDst[iIdxDim]) piCoorDst_Max[iIdxDim] = plDimDst[iIdxDim];
        }

        // grid value using KB
        int *piCoorDst = new int[iNdim];
        for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim) piCoorDst[iIdxDim] = piCoorDst_Min[iIdxDim];
        double *pdDx = new double[iNdim];
        memset(pdDx, 0x00, iNdim*sizeof(double));
        double dC0 = C(iNdim, pdDx, KB_WID);
        while(1)
        {
            // derive idx from coor
            int iIdxDst = GetIdxCoor(piCoorDst, iNdim, plDimDst);
            
            // dx, dy, dz ...
            for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim)
            {
                pdDx[iIdxDim] = piCoorDst[iIdxDim] - pdCoorSrc[iIdxSrc*iNdim + iIdxDim];
            }
            
            // derive KB value
            pcDataDst[iIdxDst] += pcDataSrc[iIdxSrc]*C(iNdim, pdDx, KB_WID)/dC0;

            // break check
            int iBreak = 1;
            for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim)
            {
                if (piCoorDst[iIdxDim] != piCoorDst_Max[iIdxDim] - 1) iBreak = 0;
            }
            if (iBreak) break;

            // update coor
            for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim)
            {
                piCoorDst[iIdxDim] += 1;
                if (piCoorDst[iIdxDim] == piCoorDst_Max[iIdxDim]) piCoorDst[iIdxDim] = piCoorDst_Min[iIdxDim];
                else break;
            }
        }
        
        delete[] piCoorDst_Min;
        delete[] piCoorDst_Max;
        delete[] piCoorDst;
        delete[] pdDx;
    }

    // initialize FFTW plan
    fftw_complex *pfftwSrc, *pfftwDst;
    pfftwSrc = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * iNptDst);
    pfftwDst = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * iNptDst);
    fftw_plan plan;
    int *piDimDst = new int[iNdim];
    for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim) piDimDst[iIdxDim] = plDimDst[iIdxDim];
    plan = fftw_plan_dft(iNdim, piDimDst, pfftwSrc, pfftwDst, FFTW_BACKWARD, FFTW_ESTIMATE);
    delete[] piDimDst;

    // IFFT the grided data
    Shift(pcDataDst, iNdim, plDimDst);
    memcpy((void*)pfftwSrc, pcDataDst, sizeof(complex)*iNptDst);
    fftw_execute(plan);
    memcpy((void*)pcDataDst, pfftwDst, sizeof(complex)*iNptDst);
    Shift(pcDataDst, iNdim, plDimDst);
    for (int iIdxDst = 0; iIdxDst < iNptDst; ++iIdxDst) pcDataDst[iIdxDst] /= iNptDst;

    // generate kernel function
    int *piCoorOri = new int[iNdim];
    for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim) piCoorOri[iIdxDim] = plDimDst[iIdxDim]%2 ? plDimDst[iIdxDim]/2 + 1 : plDimDst[iIdxDim]/2;
    complex *pcKernel = new complex[iNptDst];
    memset((void*)pcKernel, 0x00, sizeof(complex)*iNptDst);
    int *piCoorKer_Min = new int[iNdim];
    for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim)
    {
        piCoorKer_Min[iIdxDim] = piCoorOri[iIdxDim] - KB_WID/2;
        if (piCoorKer_Min[iIdxDim] < 0) piCoorKer_Min[iIdxDim] = 0;
    }
    int *piCoorKer_Max = new int[iNdim];
    for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim)
    {
        piCoorKer_Max[iIdxDim] = piCoorOri[iIdxDim] + KB_WID/2;
        if (piCoorKer_Max[iIdxDim] > plDimDst[iIdxDim] - 1) piCoorKer_Max[iIdxDim] = plDimDst[iIdxDim] - 1;
    }
    int *piCoorKer = new int[iNdim];
    for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim) piCoorKer[iIdxDim] = piCoorKer_Min[iIdxDim];
    double *pdDx = new double[iNdim];
    memset((void*)pdDx, 0x00, sizeof(int)*iNdim);
    double dC0 = C(iNdim, pdDx, KB_WID);
    while(1)
    {
        // derive index from coord
        int iIdxKer = GetIdxCoor(piCoorKer, iNdim, plDimDst);
            
        // dx, dy, dz ...
        for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim)
        {
            pdDx[iIdxDim] = piCoorKer[iIdxDim] - piCoorOri[iIdxDim];
        }

        // calculate kernel value
        pcKernel[iIdxKer] = C(iNdim, pdDx, KB_WID)/dC0;

        // break
        int iBreak = 1;
        for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim)
        {
            if (piCoorKer[iIdxDim] != piCoorKer_Max[iIdxDim] - 1) iBreak = 0;
        }
        if (iBreak) break;

        // update coord
        for (int iIdxDim = 0; iIdxDim < iNdim; ++iIdxDim)
        {
            piCoorKer[iIdxDim] += 1;
            if (piCoorKer[iIdxDim] == piCoorKer_Max[iIdxDim]) piCoorKer[iIdxDim] = piCoorKer_Min[iIdxDim];
            else break;
        }
    }
    delete[] piCoorOri;
    delete[] piCoorKer_Min;
    delete[] piCoorKer_Max;
    delete[] piCoorKer;
    delete[] pdDx;

    // IFFT the kernel function
    complex *pcFilter = new complex[iNptDst];
    Shift(pcKernel, iNdim, plDimDst);
    memcpy((void*)pfftwSrc, pcKernel, iNptDst*sizeof(complex));
    fftw_execute(plan);
    memcpy((void*)pcFilter, pfftwDst, iNptDst*sizeof(complex));
    Shift(pcFilter, iNdim, plDimDst);
    for (int iIdxDst = 0; iIdxDst < iNptDst; ++iIdxDst) pcFilter[iIdxDst] /= iNptDst;
    delete[] pcKernel;

    // element-wise division
    for (int iIdxDst = 0; iIdxDst < iNptDst; ++iIdxDst)
    {
        pcDataDst[iIdxDst] /= pcFilter[iIdxDst];
    }
    delete[] pcFilter;

    // Deinitialize FFTW plan
    fftw_destroy_plan(plan);

    // create PyArray for return
    PyArrayObject *poDataDst = (PyArrayObject*)PyArray_SimpleNewFromData(iNdim, plDimDst, NPY_COMPLEX128, (void*)pcDataDst);
    PyArray_ENABLEFLAGS(poDataDst, NPY_ARRAY_OWNDATA);

    return PyArray_Return(poDataDst);
}

static PyMethodDef aMeth[] = {
    {"_dft", dft, METH_VARARGS, ""},
    {"_ifft", ifft, METH_VARARGS, ""},
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
