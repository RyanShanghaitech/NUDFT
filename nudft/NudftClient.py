from numpy import *
from socket import *

class NudftClient:
    def __init__(self, ipServer:str="127.0.0.1", portServer:int=7885) -> None:
        self.objSocket = socket(AF_INET, SOCK_STREAM)
        try:
            self.objSocket.connect((ipServer, portServer))
        except:
            raise ConnectionError("in classNudftClient")

    def _packData(self, flagIdft:bool, lstCoorIn:ndarray, lstDataIn:ndarray, lstCoorOut:ndarray) -> ndarray:
        assert(lstCoorIn.ndim == 2 and lstDataIn.ndim == 1 and lstCoorOut.ndim == 2)
        assert(lstCoorIn.shape[0] == lstDataIn.shape[0]) # num of point consistency
        assert(lstCoorIn.shape[1] == lstCoorOut.shape[1]) # dim consistency

        # derive metadata
        numDim = uint8(lstCoorIn.shape[1])
        typeTf = uint8(1 + 2*(numDim-1)) if flagIdft else uint8(2*(numDim-1))
        numCoorIn = uint64(lstCoorIn.shape[0])
        numCoorOut = uint64(lstCoorOut.shape[0])
        lstCoorIn = lstCoorIn.astype(float64).flatten()
        lstDataIn = lstDataIn.astype(complex128)
        lstDataIn = array([lstDataIn.real, lstDataIn.imag], dtype=float64).T.flatten()
        lstCoorOut = lstCoorOut.astype(float64).flatten()

        # derive validation sum
        bytesPkgTx = typeTf.tobytes() + numCoorIn.tobytes() + numCoorOut.tobytes() + lstCoorIn.tobytes() + lstDataIn.tobytes() + lstCoorOut.tobytes()
        lstPkgTx = array(list(bytesPkgTx), dtype=uint8)

        sumBytes = uint8(sum(lstPkgTx))

        bytesPkgTx += sumBytes.tobytes()
        lstPkgTx = append(lstPkgTx, [sumBytes], 0)

        # add header, footer, and escape
        idx0xFA = where(lstPkgTx == 0xFA)[0]
        idx0xFB = where(lstPkgTx == 0xFB)[0]
        idx0xFC = where(lstPkgTx == 0xFC)[0]

        lstPkgTx[idx0xFA] = 0xFD
        lstPkgTx[idx0xFB] = 0xFE
        lstPkgTx[idx0xFC] = 0xFF

        lstPkgTx = insert(lstPkgTx, concatenate((idx0xFA, idx0xFB, idx0xFC), 0), [uint8(0xFB)], 0)
        lstPkgTx = insert(lstPkgTx, 0, [uint8(0xFA)], 0)
        lstPkgTx = append(lstPkgTx, [uint8(0xFC)], 0)

        # return
        return lstPkgTx
    
    def _unpackData(self, bytesPkgRx:bytes) -> ndarray:
        # check header, footer
        assert(bytesPkgRx[0] == 0xFA)
        assert(bytesPkgRx[-1] == 0xFC)

        # convert to array
        lstPkgRx = frombuffer(bytesPkgRx, dtype=uint8).copy()

        # remove header, footer and escape
        lstPkgRx = lstPkgRx[1:-1]
        assert(lstPkgRx[-1] != 0xFB)
        idx0xFB = where(lstPkgRx == 0xFB)[0]

        lstPkgRx[idx0xFB + 1] -= 0x03
        lstPkgRx = delete(lstPkgRx, idx0xFB, 0)

        # check sum
        assert(uint8(sum(lstPkgRx[:-1])) == lstPkgRx[-1])

        # derive data
        lstDataOut = frombuffer(lstPkgRx[:-1], dtype=float64)

        # return
        return lstDataOut

    def _acqPkg(self) -> bytes:
        # self.objSocket.setblocking(False)
        flagHeader = False
        bytesPkgRx = bytes()
        while True:
            # bytesPartRx = self.objSocket.recv(int(1e3))
            bytesPartRx = self.objSocket.recv(int(1e3))
            if flagHeader:
                bytesPkgRx += bytesPartRx
            else:
                if 0xFA in bytesPartRx:
                    flagHeader = True
                    bytesPartRx = bytesPartRx[bytesPartRx.index(0xFA):]
                    bytesPkgRx += bytesPartRx
                else:
                    pass
            if 0xFC in bytesPkgRx:
                bytesPkgRx = bytesPkgRx[:bytesPkgRx.index(0xFC)+1]
                break
        return bytesPkgRx
    
    def nudft(self, lstIx:ndarray, lstX:ndarray, lstK:ndarray) -> ndarray:
        assert(lstX.shape[0] == lstIx.shape[0]) # num of point consistency
        assert(lstX.shape[1] == lstK.shape[1]) # dim consistency
        lstPkgTx = self._packData(False, lstX, lstIx, lstK)
        self.objSocket.send(lstPkgTx)
        lstPkgRx = self._acqPkg()
        lstSk = self._unpackData(lstPkgRx)
        lstSk = lstSk[0::2] + 1j*lstSk[1::2]
        return lstSk
    
    def nuidft(self, lstSk:ndarray, lstK:ndarray, lstX:ndarray) -> ndarray:
        assert(lstK.shape[0] == lstSk.shape[0]) # num of point consistency
        assert(lstK.shape[1] == lstX.shape[1]) # dim consistency
        lstPkgTx = self._packData(True, lstK, lstSk, lstX)
        self.objSocket.send(lstPkgTx)
        bytesPkgRx = self._acqPkg()
        lstIx = self._unpackData(bytesPkgRx)
        lstIx = lstIx[0::2] + 1j*lstIx[1::2]
        return lstIx

