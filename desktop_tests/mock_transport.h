// mock_transport.h — TEST-ONLY multi-node broadcast simulation.
//
// NNTransportLoopback (src/NNTransportLoopback.h) only loops a single
// node's own send() back to its own receive() -- fine for single-node
// tests, but genuine multi-node integration tests need several
// independent NNTransport instances that can actually hear each other.
// This file provides exactly that, implementing the REAL NNTransport
// interface faithfully so it's a drop-in stand-in for real hardware
// transports in test code, without touching any production file.

#pragma once
#include <vector>
#include "NNTransport.h"
#include "NNPacket.h"
#include "NNBuffer.h"

// The shared medium every NNMockTransport instance broadcasts onto and
// listens to. One instance per test; every NNMockTransport registers
// itself with it at construction.
class NNMockBroadcastMedium {
public:
    // Registers a new participant and returns its index, used internally
    // to exclude the sender from receiving its own broadcast (matching
    // real-world broadcast transports, which generally don't self-deliver
    // either -- consistent with this project's own BLE testing experience).
    int registerParticipant(NNPacketQueue* inbox) {
        inboxes.push_back(inbox);
        return static_cast<int>(inboxes.size()) - 1;
    }

    void broadcast(const NNPacket& pkt, int senderIndex) {
        for (size_t i = 0; i < inboxes.size(); i++) {
            if (static_cast<int>(i) == senderIndex) continue;  // no self-delivery
            inboxes[i]->push(pkt);  // NNPacketQueue::push already handles a full
                                      // queue by rejecting silently (Milestone 3) --
                                      // acceptable for these bounded, short test passes
        }
    }

    // Lets a test inject a packet directly (e.g. simulated sensor inputs
    // x0/x1, which have no NNNode/NNTransport of their own) without any
    // participant "sending" it.
    void injectDirect(const NNPacket& pkt) {
        broadcast(pkt, -1);  // -1 never matches a real registered index
    }

private:
    std::vector<NNPacketQueue*> inboxes;
};

class NNMockTransport : public NNTransport {
public:
    explicit NNMockTransport(NNMockBroadcastMedium& sharedMedium) : medium(sharedMedium) {
        myIndex = medium.registerParticipant(&inbox);
    }

    bool begin() override { return true; }

    bool send(const NNPacket& pkt) override {
        medium.broadcast(pkt, myIndex);
        return true;
    }

    bool receive(NNPacket& outPkt) override {
        return inbox.pop(outPkt);
    }

    void poll() override { /* nothing to service -- pure in-process simulation */ }

private:
    NNMockBroadcastMedium& medium;
    NNPacketQueue inbox;  // reused, real, unmodified class from NNBuffer.h
    int myIndex;
};