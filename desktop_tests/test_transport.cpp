#include<cassert>
#include"NNTransportLoopback.h"

int main() {
    NNTransportLoopback transportA;
    transportA.begin();

    NNPacket pkt{};
    pkt.header.sourceAddress = encodeAddress({0, 1, 0, 0});
    pkt.header.payloadCount = 1;
    pkt.payload[0] = 3.14f;

    assert(transportA.send(pkt));

    NNPacket received{};
    assert(transportA.receive(received));
    assert(received.payload[0] == 3.14f);

    NNPacket empty{};
    assert(!transportA.receive(empty));  // queue now empty — must return false, not block

    return 0;
}