#include<cassert>
#include<cstring>
#include"NNPacket.h"
#include <iostream>

int main() {
    NNPacket original{};
    original.header.sourceAddress  = encodeAddress({3, 2, 1, 0}); // from Milestone 1
    original.header.targetLayerId  = 3;
    original.header.type           = NNPacketType::DATA;
    original.header.sequenceNumber = 42;
    original.header.payloadCount   = 2;
    original.header.flags          = 0;
    original.payload[0] = 0.75f;
    original.payload[1] = -1.5f;

    uint8_t buffer[32];
    uint16_t written = serializePacket(original, buffer, sizeof(buffer));
    assert(written > 0);

    NNPacket recovered{};
    bool ok = deserializePacket(buffer, written, recovered);
    assert(ok);
    assert(recovered.header.sourceAddress == original.header.sourceAddress);
    assert(recovered.payload[0] == 0.75f);
    assert(recovered.payload[1] == -1.5f);

    // Corruption test
    buffer[2] ^= 0xFF; // flip targetLayerId's bits
    NNPacket corrupted{};
    bool shouldFail = deserializePacket(buffer, written, corrupted);
    std::cout << "Corrupted packet result: " << shouldFail << std::endl;

    assert(shouldFail == false); // your completed checksum logic must catch this
    std::cout << "Checksum corruption test passed!" << std::endl;
    
    return 0;
}