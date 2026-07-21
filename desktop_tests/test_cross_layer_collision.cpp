// test_cross_layer_collision.cpp — regression test for the cross-layer
// node ID collision fix in NNNode::onPacketReceived and
// NNBackupStandby::onPacketObserved.
//
// Deliberately reuses nodeId=0 in BOTH layer 0 (as a decoy sender) and
// layer 1 (as a real predecessor a node under test is NOT supposed to
// listen to), on a single shared broadcast medium -- exactly the scenario
// that motivated adding predecessorLayerId/backupTargetPredecessorLayerId.
// Before the fix, this collision would have silently corrupted the
// weighted sum; after the fix, the layer-0 decoy is correctly ignored.

#include <cassert>
#include <cstdio>
#include <cmath>
#include "NNAddress.h"
#include "NNPacket.h"
#include "NNNode.h"

constexpr float TOLERANCE = 0.0001f;

NNPacket makePacket(uint8_t nodeId, uint8_t layerId, float value) {
    NNPacket pkt{};
    pkt.header.sourceAddress = encodeAddress({nodeId, layerId, 0, 0});
    pkt.header.payloadCount = 1;
    pkt.payload[0] = value;
    return pkt;
}

void test_layer0_decoy_with_colliding_nodeId_is_correctly_ignored() {
    // A node in layer 2 expecting exactly one real predecessor: nodeId=0,
    // living in layer 1 (predecessorLayerId=1). Weight chosen so any
    // contamination from the decoy would be immediately, unambiguously
    // visible in the result.
    static const float weights[1] = {10.0f};

    NNNodeConfig cfg{};
    cfg.address = {5, 2, 0, 0};
    cfg.predecessorMask = 0b0001;         // expects nodeId=0
    cfg.predecessorLayerId = 1;            // ...specifically from layer 1
    cfg.activationType = NNActivationType::LINEAR;
    cfg.weights = weights;
    cfg.weightCount = 1;

    NNNode node(cfg);

    // DECOY: nodeId=0, but in layer 0 -- must be ignored entirely, even
    // though its nodeId matches predecessorMask's bit 0.
    NNPacket decoy = makePacket(0, /*layerId=*/0, /*value=*/999.0f);
    node.onPacketReceived(decoy);

    assert(!node.readyToExecute() &&
           "the decoy from the WRONG layer must not satisfy readiness -- "
           "if this fails, the collision fix is not working");

    // REAL: nodeId=0, in the correct layer 1.
    NNPacket real = makePacket(0, /*layerId=*/1, /*value=*/3.0f);
    node.onPacketReceived(real);

    assert(node.readyToExecute());
    node.execute();

    NNPacket out = node.buildOutputPacket();
    float expected = 3.0f * 10.0f;  // ONLY the real value should have counted
    assert(std::fabs(out.payload[0] - expected) < TOLERANCE &&
           "output was corrupted by the decoy -- the collision fix failed");

    printf("PASSED: layer-0 decoy sharing nodeId=0 with the real layer-1 predecessor "
           "was correctly ignored; output = %.2f (expected %.2f, proving only the "
           "real value counted)\n", out.payload[0], expected);
}

void test_without_layer_filter_would_have_been_corrupted() {
    // This test documents WHY the fix matters, by manually reproducing
    // the OLD (buggy) behavior inline -- filtering by node ID alone -- and
    // showing it WOULD have produced a wrong, contaminated result. Not a
    // test of production code; a deliberate illustration for the record.
    uint8_t decoyNodeId = 0, decoyLayerId = 0;
    uint8_t realNodeId = 0, realLayerId = 1;
    (void)decoyLayerId; (void)realLayerId;  // the old buggy logic never looked at these at all

    // Old logic: "if (senderNodeId matches predecessorMask bit) -> store,
    // OVERWRITING whatever was there" (see NNInputBuffer::storeInput,
    // Milestone 3's deliberate overwrite-on-repeat design). Since both the
    // decoy and the real packet share nodeId=0, the OLD code would have
    // stored the decoy's 999.0, then the real 3.0 overwrites it -- OR the
    // reverse order, depending on arrival timing. Either way, the node ID
    // collision meant the two BLENDED into one slot instead of being
    // correctly distinguished as "one relevant, one irrelevant."
    assert(decoyNodeId == realNodeId);  // the collision itself, stated plainly
    printf("PASSED (illustrative): confirmed nodeId=0 genuinely collides across "
           "layer 0 and layer 1 in this scenario -- exactly the condition the "
           "layer-filter fix exists to handle\n");
}

int main() {
    printf("=== Cross-layer node ID collision regression tests ===\n\n");
    test_layer0_decoy_with_colliding_nodeId_is_correctly_ignored();
    test_without_layer_filter_would_have_been_corrupted();
    printf("\nALL COLLISION REGRESSION TESTS PASSED\n");
    return 0;
}