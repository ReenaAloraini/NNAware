// Module built for testing only 
// Simulates "sending" by immediately making the packet available to "receive", this lets the ENTIRE framework be tested
// end-to-end on a laptop, with zero hardware
#pragma once
#include"NNTransport.h"
#include"NNBuffer.h"

class NNTransportLoopback : public NNTransport {
public:
    bool begin() override { return true; }

    bool send(const NNPacket& pkt) override {
        return queue.push(pkt);  // 
    }

    bool receive(NNPacket& outPkt) override {
        return queue.pop(outPkt);
    }

    void poll() override { /* nothing to service — no real hardware */ }

private:
    NNPacketQueue queue;  
};