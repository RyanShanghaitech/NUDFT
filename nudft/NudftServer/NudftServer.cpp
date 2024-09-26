#include "NudftServer.h"

NudftServer::NudftServer(QObject *parent): QTcpServer(parent)
{
    typeConfig config;

    getAddrPort(&config);
    strcpy(addrServer, config.addr);
    portServer = config.port;

    // print config
    printf("[INFO] ADDR: %s\n", addrServer);
    printf("[INFO] PORT: %d\n", portServer);

    if(this->listen(QHostAddress(addrServer), portServer)){
        printf("[INFO] listening\n");
        connect(this, &QTcpServer::newConnection, this, &NudftServer::slotNewConnection);
    }else{
        printf("[ERRO] listening failed\n");
        throw std::runtime_error("Failed to bind required ADDR and PORT.");
    }
}

NudftServer::~NudftServer()
{
}

int NudftServer::getAddrPortDefault(NudftServer::typeConfig* config)
{
    strcpy(config->addr, ADDR_SERVER_DEFAULT);
    config->port = PORT_SERVER_DEFAULT;

    return 0;
}

int NudftServer::getAddrPortFile(NudftServer::typeConfig* config)
{
    config->addr[0] = '\0';
    config->port = 0;
    FILE* fileConfig = fopen(FILE_CFG, "r");

    // validate file
    if(fileConfig == nullptr){
        return 1;
    }else{
        fseek(fileConfig, 0, SEEK_SET);
    }

    // read config from file
    int numByteAddr = fscanf(fileConfig, "[ADDR] %15s\n", (char*)&config->addr);
    int numBytePort = fscanf(fileConfig, "[PORT] %d\n", &config->port);
    fclose(fileConfig);

    if(numByteAddr != 1 || numBytePort != 1){
        return 1;
    }

    return 0;
}

int NudftServer::saveAddrPort(NudftServer::typeConfig* config)
{
    // store config to file
    FILE* fileConfig = fopen(FILE_CFG, "w");
    if(fileConfig == nullptr){
        printf("[ERRO] file open error\n");
        return 1;
    }else{
        fprintf(fileConfig, "[ADDR] %s\n", (char*)config->addr);
        fprintf(fileConfig, "[PORT] %d\n", config->port);
        fclose(fileConfig);
        return 0;
    }
}

int NudftServer::getAddrPort(NudftServer::typeConfig* config)
{
    int rtFun = 0;
    if(getAddrPortFile(config)){
        getAddrPortDefault(config);
        saveAddrPort(config);
    }
    return 0;
}

void NudftServer::slotNewConnection()
{
    printf("[INFO] new connection\n");
    if(socket != nullptr && socket->isOpen()) socket->close();
    socket = this->nextPendingConnection();
    while(!socket->open(QIODevice::ReadWrite));
    connect(socket, &QTcpSocket::readyRead, this, &NudftServer::slotDataReceived);
    connect(socket, &QTcpSocket::disconnected, this, &NudftServer::slotSocketDisconnected);
}

void NudftServer::slotSocketDisconnected()
{
    printf("[INFO] disconnected\n");
    socket->close();
    listRxPkt.clear();
    listTxPkt.clear();
}

void NudftServer::slotDataReceived()
{
    // printf("[INFO] data received\n");
    static int64_t flagHeader = 0;
    static int64_t flagEscape = 0;

    QByteArray qByteArraySocketRxData = socket->readAll();
    QByteArray qByteArraySocketTxData;
    int64_t lenPkt = qByteArraySocketRxData.length();
    uint8_t byte = 0x00;
    for(int64_t idxPkt = 0; idxPkt < lenPkt; ++idxPkt)
    {
        byte = qByteArraySocketRxData.data()[idxPkt];
        switch(byte)
        {
        case 0xFA:
            flagHeader = 1;
            listRxPkt.clear();
            listTxPkt.clear();
            break;
        case 0xFB:
            flagEscape = 1;
            break;
        case 0xFC:
            flagHeader = 0;
            if(!parsePkt(&listRxPkt, &listTxPkt)){
                packData(&listTxPkt, &qByteArraySocketTxData);
                socket->write(qByteArraySocketTxData);
                socket->flush();
            }else{
                printf("[ERRO] error parsing packet\n");
            }
            break;
        default:
            if(listRxPkt.size() < MEMORY_LIMIT){
                listRxPkt.push_back(flagEscape?byte-0x03:byte);
            }else{
                printf("[ERRO] memory limit exceeded\n");
            }
            flagEscape = 0;
            break;
        }
    }
}

int NudftServer::parsePkt(const std::list<uint8_t> *listRxPkt, std::list<uint8_t> *listTxPkt)
{
    int64_t lenPkt = listRxPkt->size();
    std::unique_ptr<uint8_t[]> arrPkt(new uint8_t[lenPkt]);
    uint8_t typeTransform;
    int64_t numInput;
    int64_t numOutput;
    double *arrInputCoor;
    double *arrInputData;
    double *arrOutputCoor;
    uint8_t sumBytes;
    
    int64_t idxPkt = 0;
    int64_t lenDesired = 0;

    // convert std::list to C array
    idxPkt = 0;
    std::list<uint8_t>::const_iterator itPkt = listRxPkt->begin();
    do{
        arrPkt[idxPkt++] = *(itPkt++);
    }while(itPkt != listRxPkt->end());

    // parse parameters
    lenDesired = 2*sizeof(uint8_t) + 2*sizeof(uint64_t);
    if((size_t)lenPkt < lenDesired){
        printf("[ERRO] header size error\n");
        return 1;
    } // array size check

    int64_t ptrBias = 0;
    typeTransform = *(uint8_t*)&arrPkt[ptrBias];

    int64_t numDim;
    bool flagIDFT;
    switch(typeTransform)
    {
        case 0x00: numDim = 1; flagIDFT = false; break;
        case 0x01: numDim = 1; flagIDFT = true; break;
        case 0x02: numDim = 2; flagIDFT = false; break;
        case 0x03: numDim = 2; flagIDFT = true; break;
        case 0x04: numDim = 3; flagIDFT = false; break;
        case 0x05: numDim = 3; flagIDFT = true; break;
        default: printf("[ERRO] type error\n"); return 1;
    }

    ptrBias += sizeof(uint8_t);
    numInput = *(uint64_t*)&arrPkt[ptrBias];

    ptrBias += sizeof(uint64_t);
    numOutput = *(uint64_t*)&arrPkt[ptrBias];

    lenDesired = 
        2*sizeof(uint8_t) + // typeTransform, sumBytes
        2*sizeof(uint64_t) + // numInput, numOutput
        (numInput)*(numDim)*(sizeof(double)) + // listInputCoor
        (numInput)*(2*sizeof(double)) + // listInputData
        (numOutput)*(numDim)*(sizeof(double)); // listOutputCoor
    
    if((size_t)lenPkt != lenDesired){ // listInputData
        printf("[ERRO] data size error\n");
        return 1;
    } // array size check

    ptrBias += sizeof(uint64_t);
    arrInputCoor = (double*)&arrPkt[ptrBias];
    double* ptrInputCoor = arrInputCoor;

    ptrBias += numInput*numDim*sizeof(double);
    arrInputData = (double*)&arrPkt[ptrBias];
    double* ptrInputData = arrInputData;

    ptrBias += numInput*2*sizeof(double);
    arrOutputCoor = (double*)&arrPkt[ptrBias];
    double* ptrOutputCoor = arrOutputCoor;

    ptrBias += numOutput*numDim*sizeof(double);
    sumBytes = *(uint8_t*)&arrPkt[ptrBias];

    uint8_t derivedSum = 0;
    idxPkt = 0;
    do{
        derivedSum += arrPkt[idxPkt++];
    }while(idxPkt != lenPkt - 1);
    if(derivedSum != sumBytes){
        printf("[ERRO] sum error\n");
        return 1;
    }
    
    // derive point number per thread
    int64_t numThread = std::thread::hardware_concurrency() - 1;
    numThread = (numThread == 0)?(1):(numThread);
    std::unique_ptr<int64_t[]> arrPtsPerThread(new int64_t[numThread]);
    int64_t* ptrPtsPerThread = arrPtsPerThread.get();
    for(int64_t idxThread = 0; idxThread < numThread; ++idxThread){
        if(idxThread < numOutput%numThread){
            *ptrPtsPerThread = numOutput/numThread + 1;
        }else{
            *ptrPtsPerThread = numOutput/numThread;
        }
        ++ptrPtsPerThread;
    }

    // reserve memory for output data
    std::unique_ptr<double[]> arrOutputData(new double[numOutput*2]);
    double* ptrOutputData = arrOutputData.get();
    std::list<std::thread> listThread;
    std::list<std::thread>::iterator itThread = listThread.begin();

    // start threads
    ptrPtsPerThread = arrPtsPerThread.get();
    ptrOutputCoor = arrOutputCoor;
    ptrOutputData = arrOutputData.get();;
    for(int64_t idxThread = 0; idxThread < numThread; ++idxThread){
        listThread.push_back(std::thread(
            NudftServer::nudft,
            flagIDFT,
            numDim,
            numInput,
            arrInputCoor,
            arrInputData,
            *ptrPtsPerThread,
            ptrOutputCoor,
            ptrOutputData,
            idxThread
            ));
        ptrOutputCoor += numDim*(*ptrPtsPerThread);
        ptrOutputData += 2*(*ptrPtsPerThread);
        ++ptrPtsPerThread;
    }
    
    // wait for threads complete
    itThread = listThread.begin();
    for(int64_t idxThread = 0; idxThread < numThread; ++idxThread){
        itThread->join();
        ++itThread;
    }

    // generate Tx packet
    uint8_t* ptrOutputdataU8 = (uint8_t*)arrOutputData.get();
    uint8_t sumOutput = 0x00;
    for(int64_t idxOutputData = 0; idxOutputData < numOutput*2*sizeof(double); ++idxOutputData){ // 2*numOutput: real and imag
        sumOutput += *ptrOutputdataU8;
        listTxPkt->push_back(*ptrOutputdataU8);
        ++ptrOutputdataU8;
    }
    listTxPkt->push_back(sumOutput);

    return 0;
}

int NudftServer::packData(const std::list<uint8_t>* listTxPkt, QByteArray* qByteArraySocketTxData)
{
    qByteArraySocketTxData->append(0xFA);
    for(std::list<uint8_t>::const_iterator itTxPkt = listTxPkt->begin();
        itTxPkt != listTxPkt->end();
        ++itTxPkt){
        if(*itTxPkt >= 0xFA && *itTxPkt <= 0xFC){
            qByteArraySocketTxData->append(0xFB);
            qByteArraySocketTxData->append(*itTxPkt + 0x03);
        }else{
            qByteArraySocketTxData->append(*itTxPkt);
        }
    }
    qByteArraySocketTxData->append(0xFC);
    return 0;
}

int NudftServer::nudft(
    const bool flagIDFT,
    const int64_t numDim,
    const int64_t lenDm0,
    const double* const arrCoorDm0,
    const double* const arrValDm0,
    const int64_t lenDm1,
    const double* const arrCoorDm1,
    double* const arrValDm1,
    const int64_t idxThread)
{
    static const double pi = std::abs(std::acos((double)-1));
    const double prodSign2Pi = (flagIDFT)?(2*pi):(-2*pi);
    std::complex<double> tempDm0(0, 0);
    std::complex<double> tempDm1(0, 0);

    double* ptrValDm1 = arrValDm1;
    for(int64_t idxDm1 = 0; idxDm1 != lenDm1; ++idxDm1){
        tempDm1 = {0, 0};
        
        const double* ptrCoorDm0 = arrCoorDm0;
        const double* ptrValDm0 = arrValDm0;
        for(int64_t idxDm0 = 0; idxDm0 != lenDm0; ++idxDm0){
            tempDm0.real(*ptrValDm0++);
            tempDm0.imag(*ptrValDm0++);
            double prodXK = 0;
            const double* ptrCoorDm1 = arrCoorDm1 + idxDm1*numDim;
            for(int64_t idxDim = 0; idxDim != numDim; ++idxDim){
                prodXK += (*ptrCoorDm1++)*(*ptrCoorDm0++);
            }
            tempDm1 += tempDm0*std::exp(std::complex<double>(0, prodSign2Pi*prodXK));
        }

        *ptrValDm1++ = tempDm1.real();
        *ptrValDm1++ = tempDm1.imag();

        // if(idxThread == 0 && idxDm1%100 == 0){
        //     printf("[INFO] thread[%d]: progress = %.2f%%\n", (int)idxThread, 100*(double)idxDm1/lenDm1);
        // }
    }

    return 0;
}
