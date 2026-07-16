#pragma once
#include<stdint.h>
#include"NNAddress.h"
#include <cstring>

// Limit for how many floats a single packet can carry, to control large vectors
constexpr uint8_t NN_MAX_PAYLOAD_FLOATS = 16;

enum class NNPacketType : uint8_t {
    DATA    = 0,  // carries a node's activation output(s)
    CONTROL = 1,  // for scheduling/sync messages
    ACK     = 2,   // reserved for a future reliability layer
};

struct NNPacketHeader {
    uint16_t     sourceAddress;  
    uint8_t      targetLayerId;   
    NNPacketType type;            
    uint8_t      sequenceNumber;  // increments per send from this node (for retransmission)
    uint8_t      payloadCount;    // how many of the payload floats are valid out of the max limit  
    uint8_t      flags;           // reserved for future use (advanced configurations)
    uint8_t      checksum;        // simple additive checksum 
};

struct NNPacket {
    NNPacketHeader header;
    float payload[NN_MAX_PAYLOAD_FLOATS];
};

// Computes a simple additive checksum: sum of all bytes, truncated to 8 bits. (integrity check against accidental corruption, not a security mechanism)
inline uint8_t computeChecksum(const uint8_t* data, uint16_t length) {
    uint16_t sum = 0;
    for (uint16_t i = 0; i < length; i++) {
        sum += data[i];
    }
    return static_cast<uint8_t>(sum & 0xFF);
}

// Serializes an NNPacket into a caller-provided byte buffer. Returns the number of bytes written, or 0 on failure (buffer too small).
// This function defines the ACTUAL WIRE FORMAT of the protocol
inline uint16_t serializePacket(const NNPacket& pkt, uint8_t* buffer, uint16_t bufferSize) {
    const uint16_t totalSize = 8 /* header size, hand-counted below */
                             + pkt.header.payloadCount * sizeof(float);
    if (bufferSize < totalSize) return 0;

    uint16_t offset = 0;

    // sourceAddress, big-endian (most significant byte first)
    buffer[offset++] = static_cast<uint8_t>(pkt.header.sourceAddress >> 8);
    buffer[offset++] = static_cast<uint8_t>(pkt.header.sourceAddress & 0xFF);

    buffer[offset++] = pkt.header.targetLayerId;
    buffer[offset++] = static_cast<uint8_t>(pkt.header.type);
    buffer[offset++] = pkt.header.sequenceNumber;
    buffer[offset++] = pkt.header.payloadCount;
    buffer[offset++] = pkt.header.flags;

    // Reserve the checksum byte's position; fill it in after we know the rest.
    uint16_t checksumIndex = offset;
    buffer[offset++] = 0; // placeholder

    // Serialize each payload float, 4 bytes each, byte-by-byte
    for (uint8_t i = 0; i < pkt.header.payloadCount; i++) {
        uint32_t bits;
        memcpy(&bits, &pkt.payload[i], sizeof(float)); 
        buffer[offset++] = static_cast<uint8_t>(bits >> 24);
        buffer[offset++] = static_cast<uint8_t>(bits >> 16);
        buffer[offset++] = static_cast<uint8_t>(bits >> 8);
        buffer[offset++] = static_cast<uint8_t>(bits & 0xFF);
    }

    buffer[checksumIndex] = computeChecksum(buffer, offset);
    return offset;
}

// Deserializes a byte buffer back into an NNPacket. Returns true on success,false if the checksum doesn't match
inline bool deserializePacket(const uint8_t* buffer, uint16_t length, NNPacket& outPkt) {
    if (length < 8) return false;

    
    const uint8_t checksumIndex = 7; // checksum is always at 7 in this format
    uint8_t receivedChecksum = buffer[checksumIndex];

    uint8_t tempBuffer[128];
    if (length > sizeof(tempBuffer)) return false;

    memcpy(tempBuffer, buffer, length);
    tempBuffer[checksumIndex] = 0;

    uint8_t computedChecksum = computeChecksum(tempBuffer, length);

    if (computedChecksum != receivedChecksum) {
        return false;
    }

    uint16_t offset = 0;

    outPkt.header.sourceAddress =
        (uint16_t(buffer[offset]) << 8) | buffer[offset + 1];
    offset += 2;

    outPkt.header.targetLayerId  = buffer[offset++];
    outPkt.header.type           = static_cast<NNPacketType>(buffer[offset++]);
    outPkt.header.sequenceNumber = buffer[offset++];
    outPkt.header.payloadCount   = buffer[offset++];
    outPkt.header.flags          = buffer[offset++];

    outPkt.header.checksum = buffer[offset++];

    if (outPkt.header.payloadCount > NN_MAX_PAYLOAD_FLOATS) {
        return false;
    }

    uint16_t expectedLength = 8 + outPkt.header.payloadCount * sizeof(float);

    if (length != expectedLength) {
        return false;
    }

    for (uint8_t i = 0; i < outPkt.header.payloadCount; i++) {
        uint32_t bits =
            (uint32_t(buffer[offset]) << 24) |
            (uint32_t(buffer[offset + 1]) << 16) |
            (uint32_t(buffer[offset + 2]) << 8) |
            uint32_t(buffer[offset + 3]);

        memcpy(&outPkt.payload[i], &bits, sizeof(float));
        offset += 4;
    }

    return true;
}