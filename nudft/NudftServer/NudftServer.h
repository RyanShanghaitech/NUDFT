/*
* packet definition: {0xFA, <data block>, 0xFC}
* 0xFA~0xFC: reserved for protocol
* 0xFA: Packet header
* 0xFB: escape char, 0xFB + 0xF<D~F> denotes for 0x7<A~C>
* 0xFC: Packet footer
*
* Rx data block definition:
* {<typeTransform>, <numInputCoor>, <numOutputCoor>, <listInputCoor>, <listInputData>, <listOutputCoor>, <sumBytes>}
* para@<typeTransform>: a byte specifying the type of transform, see below
* // 0x00: NUDFT_1D
* // 0x01: NUIDFT_1D
* // 0x02: NUDFT_2D
* // 0x03: NUIDFT_2D
* // 0x04: NUDFT_3D
* // 0x05: NUIDFT_3D,
* para@<numInputCoor>: specifying the number of input points
* para@<numOutputCoor>: specifying the number of output points
* para@<listInputCoor>: list of coordinates of input points, dim0: points dim1: x/y/z (size of dim1 could be 1, 2, 3 depending on the type of transform)
* para@<listInputData>: list of data of input points, dim0: points dim1: real/imag (size of dim1 must be 2)
* para@<listOutputCoor>: list of coordinates of output points, dim0: points dim1: x/y/z (size of dim1 could be 1, 2, 3 depending on the type of transform)
* para@<sumBytes>: sum of all bytes for verification
* type@<typeTransform>: uint8
* // 0x00: NUDFT_1D
* // 0x01: NUIDFT_1D
* // 0x02: NUDFT_2D
* // 0x03: NUIDFT_2D
* // 0x04: NUDFT_3D
* // 0x05: NUIDFT_3D,
* type@<numInputCoor>: uint64
* type@<numOutputCoor>: uint64
* type@<listInputCoor>: {{float64, [float64, float64]} ...}
* type@<listInputData>: {{float64, float64} ...}
* type@<listOutputCoor>: {{float64, [float64, float64]} ...}
* type@<sumBytes>: uint8
*
* Tx data block definition:
* {<listOutputData>, <sumBytes>}
* para@<listOutputData>: list of data of input points, dim0: points dim1: real/imag (size of dim1 must be 2)
* type@<listOutputData>: {{float64, float64} ...}
*
* range of k and xyz
* k: [-0.5, 0.5]
* xyz: [-inf, inf]
*
*/
#ifndef NUFFTSERVER_H
#define NUFFTSERVER_H

#include <QtNetwork/QTcpServer>
#include <QtNetwork/QTcpSocket>
#include <QtCore/QCoreApplication>
#include <thread>
#include <complex>
#include <iostream>
#include <fstream>

#define MEMORY_LIMIT (INT32_MAX)

class NudftServer: public QTcpServer
{
    Q_OBJECT
public:
    NudftServer(QObject *parent = nullptr);
    ~NudftServer();
private:
#define FILE_CFG ("NudftServer.cfg")
#define ADDR_SERVER_DEFAULT ("127.0.0.1")
#define PORT_SERVER_DEFAULT (7885)
    typedef struct{
        char addr[16];
        int port;
    }typeConfig;
    char addrServer[16] = {'\0'};
    uint16_t portServer;
    int64_t numThread;
    QTcpSocket* socket = nullptr;
    std::list<uint8_t> listRxPkt;
    std::list<uint8_t> listTxPkt;

    int getAddrPortDefault(NudftServer::typeConfig* cfg);
    int getAddrPortFile(NudftServer::typeConfig* cfg);
    int saveAddrPort(NudftServer::typeConfig* config);
    int getAddrPort(NudftServer::typeConfig* cfg);
    int parsePkt(const std::list<uint8_t> *listRxPkt, std::list<uint8_t> *listTxPkt);
    int packData(const std::list<uint8_t>* listTxPkt, QByteArray* qByteArraySocketTxData);
    static int nudft(
        const bool flagIDFT,
        const int64_t numDim,
        const int64_t lenDm0,
        const double* const arrCoorDm0,
        const double* const arrValDm0,
        const int64_t lenDm1,
        const double* const arrCoorDm1,
        double* const arrValDm1,
        const int64_t idxThread = -1);

    #define NUDFT(numDim, size_Dm0, arrCoorDm0, arrValDm0, size_Dm1, arrCoorDm1, arrValDm1, idxThread) \
        NudftServer::nudft(false, numDim, size_Dm0, arrCoorDm0, arrValDm0, size_Dm1, arrCoorDm1, arrValDm1, idxThread)
    #define NUIDFT(numDim, size_Dm0, arrCoorDm0, arrValDm0, size_Dm1, arrCoorDm1, arrValDm1, idxThread) \
        NudftServer::nudft(true, numDim, size_Dm0, arrCoorDm0, arrValDm0, size_Dm1, arrCoorDm1, arrValDm1, idxThread)
private slots:
    void slotNewConnection();
    void slotSocketDisconnected();
    void slotDataReceived();
};

#endif // SRV_NUDFT_H
