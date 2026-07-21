#include<cassert>
#include<cmath>
#include"NNNode.h"

int main() {
    static const float weights[3] = {0.5f, -0.25f, 1.0f};

    NNNodeConfig cfg{};
    cfg.address = {3, 2, 0, 0};                 // Node D: nodeId=3, layer=2
    cfg.predecessorMask = 0b0000000000000111;   // predecessors 0 (A), 1 (B), 2 (C)
    cfg.predecessorLayerId = 1;                 // A/B/C all live in layer 1 (see encodeAddress calls below)
    cfg.successorLayerId = 3;
    cfg.activationType = NNActivationType::RELU;
    cfg.weights = weights;
    cfg.weightCount = 3;

    NNNode nodeD(cfg);

    NNPacket fromA{}, fromB{}, fromC{};
    fromA.header.sourceAddress = encodeAddress({0, 1, 0, 0});
    fromA.header.payloadCount = 1; fromA.payload[0] = 2.0f;   // A -> 2.0
    fromB.header.sourceAddress = encodeAddress({1, 1, 0, 0});
    fromB.header.payloadCount = 1; fromB.payload[0] = 4.0f;   // B -> 4.0
    fromC.header.sourceAddress = encodeAddress({2, 1, 0, 0});
    fromC.header.payloadCount = 1; fromC.payload[0] = 1.0f;   // C -> 1.0

    assert(!nodeD.readyToExecute());
    nodeD.onPacketReceived(fromA);
    nodeD.onPacketReceived(fromB);
    assert(!nodeD.readyToExecute());  // still missing C
    nodeD.onPacketReceived(fromC);
    assert(nodeD.readyToExecute());

    nodeD.execute();
    // Hand-computed: (2.0*0.5) + (4.0*-0.25) + (1.0*1.0) = 1.0 - 1.0 + 1.0 = 1.0
    // ReLU(1.0) = 1.0
    NNPacket out = nodeD.buildOutputPacket();
    assert(fabs(out.payload[0] - 1.0f) < 0.0001f);

    return 0;
}