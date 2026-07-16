//Abstract interface
#pragma once
#include"NNPacket.h"

class NNTransport {
public:
    virtual ~NNTransport() {}
    virtual bool begin() = 0;
    virtual bool send(const NNPacket& pkt) = 0;
    virtual bool receive(NNPacket& outPkt) = 0;
    virtual void poll() = 0;
};