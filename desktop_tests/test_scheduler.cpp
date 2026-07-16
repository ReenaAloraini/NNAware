#include<cassert>
#include"NNScheduler.h"

int main() {
    static const float weights[1] = {2.0f};
    NNNodeConfig cfgA{};
    cfgA.address = {0, 1, 0, 0};          // Node A: nodeId=0, layer=1
    cfgA.predecessorMask = 0;              // Layer 1 has no predecessors
    cfgA.weights = weights; cfgA.weightCount = 0;
    cfgA.activationType = NNActivationType::LINEAR;
    NNNode nodeA(cfgA);

    NNWindowConfig winA{};
    winA.ownLayerId = 1;
    winA.precedingSiblingsMask = 0;        // Node A is Slot 1 — no one precedes it

    NNScheduler schedA(nodeA, winA);

    // NNNode with predecessorMask=0 must be immediately "complete" (0 & anything == 0)
    schedA.tick();  // WAITING_FOR_INPUT -> READY_TO_EXECUTE (trivially ready, no predecessors)
    assert(schedA.getState() == NNNodeState::READY_TO_EXECUTE);
    schedA.tick();  // -> EXECUTING
    schedA.tick();  // -> executes, -> WAITING_FOR_TURN
    schedA.tick();  // no preceding siblings required -> READY_TO_TRANSMIT immediately
    assert(schedA.getState() == NNNodeState::READY_TO_TRANSMIT);

    NNPacket out;
    assert(schedA.hasOutputReady(out));
    assert(schedA.getState() == NNNodeState::TRANSMITTED);

    return 0;
}