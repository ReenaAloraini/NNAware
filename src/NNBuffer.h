#pragma once
#include<stdint.h>
#include"NNPacket.h"


constexpr uint8_t NN_QUEUE_CAPACITY = 8;  // tune based on expected burstiness, not neural-net semantics

class NNPacketQueue {
public:
    NNPacketQueue() : head(0), tail(0), count(0) {}

    // Returns false (and drops the packet) if the queue is full & drops new packets 
    bool push(const NNPacket& pkt) {
        if (count >= NN_QUEUE_CAPACITY) return false;
        items[tail] = pkt;
        tail = (tail + 1) % NN_QUEUE_CAPACITY;
        count++;
        return true;
    }

    bool pop(NNPacket& out) {
        if (count == 0) return false;
        out = items[head];
        head = (head + 1) % NN_QUEUE_CAPACITY;
        count--;
        return true;
    }

    bool isEmpty() const { return count == 0; }
    bool isFull()  const { return count >= NN_QUEUE_CAPACITY; }
    uint8_t size()  const { return count; }

private:
    NNPacket items[NN_QUEUE_CAPACITY];
    uint8_t head, tail, count;
};


//synchronization gate 

constexpr uint8_t NN_MAX_PREDECESSORS = 16;  

class NNInputBuffer {
public:
    NNInputBuffer() : receivedMask(0) {
        for (uint8_t i = 0; i < NN_MAX_PREDECESSORS; i++) counts[i] = 0;
    }

    // Stores a predecessor's input, keyed by its 4-bit nodeId (0-15).
    // overwrites any prior value from the same sender
    bool storeInput(uint8_t senderNodeId, const float* srcValues, uint8_t count) {
        if (senderNodeId >= NN_MAX_PREDECESSORS || count > NN_MAX_PAYLOAD_FLOATS) return false;
        for (uint8_t i = 0; i < count; i++) values[senderNodeId][i] = srcValues[i];
        counts[senderNodeId] = count;
        receivedMask |= (uint16_t(1) << senderNodeId);
        return true;
    }

    // A pass is complete once every predecessor in `expectedMask` has a set bit in `receivedMask`. 
    bool isComplete(uint16_t expectedMask) const {
        return (receivedMask & expectedMask) == expectedMask;
    }

    const float* getInput(uint8_t senderNodeId, uint8_t& outCount) const {
        outCount = counts[senderNodeId];
        return values[senderNodeId];
    }

    // Must be called between inference passes 
    void reset() {
        receivedMask = 0;
        for (uint8_t i = 0; i < NN_MAX_PREDECESSORS; i++) counts[i] = 0;
    }

private:
    float values[NN_MAX_PREDECESSORS][NN_MAX_PAYLOAD_FLOATS];
    uint8_t counts[NN_MAX_PREDECESSORS];
    uint16_t receivedMask;
};