from numpy import *
from .ext import _dft, _ifft

def dft(arrSrc:ndarray, arrCoorSrc:ndarray, arrCoorDst:ndarray) -> ndarray:
    return _dft(arrSrc, arrCoorSrc, arrCoorDst, 0)

def idft(arrSrc:ndarray, arrCoorSrc:ndarray, arrCoorDst:ndarray) -> ndarray:
    return _dft(arrSrc, arrCoorSrc, arrCoorDst, 1)

def ifft(arrSrc:ndarray, arrCoorSrc:ndarray, arrCoorDst:ndarray) -> ndarray:
    return _ifft(arrSrc, arrCoorSrc, arrCoorDst)