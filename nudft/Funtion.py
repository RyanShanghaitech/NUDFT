from numpy import *
from .ext import _dft

def _checkDim(dimDataSrc:ndarray, dimCoorSrc:ndarray, dimCoorDst:ndarray) -> bool:
    if \
    (
        dimDataSrc.size != 2 or
        dimCoorSrc.size != 2 or
        dimCoorDst.size != 2 or
        dimDataSrc[1] != dimCoorSrc[0] or # Npt consistency
        dimDataSrc[0] not in [1, dimCoorDst[0]] or # Nt consistency
        dimCoorSrc[1] != dimCoorDst[1] # Ndim consistency
    ):
        return False
    else:
        return True
    
def dft(arrSrc:ndarray, arrCoorSrc:ndarray, arrCoorDst:ndarray) -> ndarray:
    if arrSrc.ndim == 1: arrSrc = arrSrc[newaxis,:]
    if not _checkDim(array(arrSrc.shape), array(arrCoorSrc.shape), array(arrCoorDst.shape)): raise RuntimeError("")
    return _dft(arrSrc, arrCoorSrc, arrCoorDst, 0)

def idft(arrSrc:ndarray, arrCoorSrc:ndarray, arrCoorDst:ndarray) -> ndarray:
    if arrSrc.ndim == 1: arrSrc = arrSrc[newaxis,:]
    if not _checkDim(array(arrSrc.shape), array(arrCoorSrc.shape), array(arrCoorDst.shape)): raise RuntimeError("")
    return _dft(arrSrc, arrCoorSrc, arrCoorDst, 1)
