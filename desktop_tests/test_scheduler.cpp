#include<cassert>
#include<cstdio>
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

    // A predecessorMask==0 node seeded externally (node.seedOutput(), e.g. a
    // real physical input-layer device) sets hasExecuted=true BEFORE the
    // scheduler ever ticks -- readyToExecute() (!hasExecuted && ...) is then
    // false forever, so the normal WAITING_FOR_INPUT->READY_TO_EXECUTE path
    // above can never fire for it. notifySeeded() is the bridge: it must
    // skip straight to WAITING_FOR_TURN so the existing turn-taking/transmit
    // machinery still runs unmodified from there.
    NNNodeConfig cfgB{};
    cfgB.address = {1, 0, 0, 0};   // Node B: nodeId=1, layer=0 (a second input-layer sibling)
    cfgB.predecessorMask = 0;
    cfgB.weightCount = 0;
    cfgB.activationType = NNActivationType::LINEAR;
    NNNode nodeB(cfgB);

    NNWindowConfig winB{};
    winB.ownLayerId = 0;
    winB.precedingSiblingsMask = 0b1;  // must observe sibling nodeId 0 transmit first

    NNScheduler schedB(nodeB, winB);
    nodeB.seedOutput(3.5f);
    schedB.notifySeeded();
    assert(schedB.getState() == NNNodeState::WAITING_FOR_TURN);

    schedB.tick();  // sibling 0 hasn't been observed yet -- still waiting
    assert(schedB.getState() == NNNodeState::WAITING_FOR_TURN);

    NNPacket siblingPkt{};
    siblingPkt.header.sourceAddress = encodeAddress({0, 0, 0, 0});
    schedB.onPacketObserved(siblingPkt);
    schedB.tick();
    assert(schedB.getState() == NNNodeState::READY_TO_TRANSMIT);

    NNPacket outB;
    assert(schedB.hasOutputReady(outB));
    assert(outB.payload[0] == 3.5f);
    assert(schedB.getState() == NNNodeState::TRANSMITTED);

    // notifySeeded() must be a no-op once already past WAITING_FOR_INPUT --
    // calling it again (e.g. a stray duplicate call) must not rewind state.
    schedB.notifySeeded();
    assert(schedB.getState() == NNNodeState::TRANSMITTED);

    printf("ALL SCHEDULER TESTS PASSED\n");
    return 0;
}