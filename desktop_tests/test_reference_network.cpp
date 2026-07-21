// test_reference_network.cpp — end-to-end integration tests for the
// reference network, using the REAL NNNode/NNScheduler/NNFailover classes
// wired together via NNMockBroadcastMedium, not reimplemented logic.
//
// Reference network:
//   x0=1.0, x1=2.0
//   H0: weights [0.5, 1.0],  ReLU -> expected 2.5
//   H1: weights [-1.0, 2.0], ReLU -> expected 3.0
//   O0: weights [2.0, -0.5], ReLU -> expected 3.5
//
// DESIGN NOTE, confirmed by inspecting the real NNNode.h: onPacketReceived
// matches predecessors by NODE ID ONLY, not layer -- so on a single shared
// broadcast medium, node IDs must be unique ACROSS THE WHOLE NETWORK, not
// just within each layer. Assigned here as: x0=0, x1=1, H0=2, H1=3, O0=4.
//
// BACKUP TOPOLOGY: H0 and H1 mutually back each other up (same layer,
// per the project's per-layer-sibling design). O0 is alone in layer 2 and
// therefore CANNOT have a backup under this design -- a real, structural
// consequence of the reference network's shape, not a limitation of the
// test. Single-failure tests target H1 (backed up by H0).
//
// MULTIPLE SIMULTANEOUS FAILURES: explicitly deferred per an earlier,
// direct decision in this project -- not built or tested here. See the
// skipped placeholder test at the bottom of this file.

#include <cassert>
#include <cstdio>
#include <cmath>
#include "NNAddress.h"
#include "NNPacket.h"
#include "NNNode.h"
#include "NNScheduler.h"
#include "NNFailover.h"
#include "mock_transport.h"

static unsigned long fakeClock = 0;
unsigned long testClock() { return fakeClock; }

constexpr float TOLERANCE = 0.0001f;

// --- Reference network weight tables ---------------------------------------
static const float h0Weights[2] = {0.5f, 1.0f};
static const float h1Weights[2] = {-1.0f, 2.0f};
static const float o0Weights[2] = {2.0f, -0.5f};

// --- Config builders ---------------------------------------------------------

NNNodeConfig makeH0Config() {
    NNNodeConfig cfg{};
    cfg.address = {2, 1, 0, 0};          // nodeId=2, layer=1
    cfg.predecessorMask = 0b0011;        // x0 (nodeId 0), x1 (nodeId 1)
    cfg.successorLayerId = 2;
    cfg.transmitSlot = 0;
    cfg.activationType = NNActivationType::RELU;
    cfg.weights = h0Weights;
    cfg.weightCount = 2;

    // H0 also backs up H1 (mutual pairing).
    cfg.hasBackupRole = true;
    cfg.backupTargetAddress = {3, 1, 0, 0};       // H1: nodeId=3, layer=1
    cfg.backupTargetPredecessorMask = 0b0011;      // H1's own predecessors: x0, x1
    cfg.backupTargetActivationType = NNActivationType::RELU;
    cfg.backupWeights = h1Weights;                 // a COPY of H1's real weights
    cfg.backupWeightCount = 2;
    cfg.resendGraceMs = 50;
    cfg.layerRosterMask = 0b1100;                   // H0 (bit2) + H1 (bit3)
    cfg.predecessorLayerId = 0;                      // x0/x1 live in layer 0
    cfg.backupTargetPredecessorLayerId = 0;           // H1's own predecessors also live in layer 0
    return cfg;
}

NNNodeConfig makeH1Config() {
    NNNodeConfig cfg{};
    cfg.address = {3, 1, 0, 0};          // nodeId=3, layer=1
    cfg.predecessorMask = 0b0011;
    cfg.successorLayerId = 2;
    cfg.transmitSlot = 1;
    cfg.activationType = NNActivationType::RELU;
    cfg.weights = h1Weights;
    cfg.weightCount = 2;

    // H1 also backs up H0 (the mutual half of the pairing).
    cfg.hasBackupRole = true;
    cfg.backupTargetAddress = {2, 1, 0, 0};       // H0: nodeId=2, layer=1
    cfg.backupTargetPredecessorMask = 0b0011;
    cfg.backupTargetActivationType = NNActivationType::RELU;
    cfg.backupWeights = h0Weights;                 // a COPY of H0's real weights
    cfg.backupWeightCount = 2;
    cfg.resendGraceMs = 50;
    cfg.layerRosterMask = 0b1100;
    cfg.predecessorLayerId = 0;                      // x0/x1 live in layer 0
    cfg.backupTargetPredecessorLayerId = 0;           // H0's own predecessors also live in layer 0
    return cfg;
}

NNNodeConfig makeO0Config() {
    NNNodeConfig cfg{};
    cfg.address = {4, 2, 0, 0};          // nodeId=4, layer=2
    cfg.predecessorMask = 0b1100;        // H0 (nodeId 2), H1 (nodeId 3)
    cfg.successorLayerId = 255;          // terminal -- no further layer
    cfg.transmitSlot = 0;
    cfg.activationType = NNActivationType::RELU;
    cfg.weights = o0Weights;
    cfg.weightCount = 2;
    cfg.predecessorLayerId = 1;          // H0/H1 live in layer 1
    // O0 has no backup role: it is alone in layer 2, no sibling to pair with.
    return cfg;
}

NNPacket makeInputPacket(uint8_t nodeId, float value) {
    NNPacket pkt{};
    pkt.header.sourceAddress = encodeAddress({nodeId, 0, 0, 0});
    pkt.header.type = NNPacketType::DATA;
    pkt.header.payloadCount = 1;
    pkt.payload[0] = value;
    return pkt;
}

// --- The test harness: wires H0, H1, O0 together on one shared medium,
//     drives their schedulers/standbys, and returns O0's final output. ---
struct NetworkResult {
    bool completed = false;
    float finalOutput = 0.0f;
    bool h0Substituted = false;
    bool h1Substituted = false;
    bool anyTeardown = false;
};

NetworkResult runReferenceNetwork(bool h1IsAlive) {
    fakeClock = 0;
    NNMockBroadcastMedium medium;
    NNMockTransport h0Transport(medium), h1Transport(medium), o0Transport(medium);

    NNNodeConfig h0Cfg = makeH0Config();
    NNNodeConfig h1Cfg = makeH1Config();
    NNNodeConfig o0Cfg = makeO0Config();

    NNNode h0Node(h0Cfg), h1Node(h1Cfg), o0Node(o0Cfg);

    NNWindowConfig h0Window{1, 0b0000};              // layer 1, slot 0 -- no preceding siblings
    NNWindowConfig h1Window{1, uint16_t(1 << 2)};    // layer 1, slot 1 -- must observe H0 (bit2) first
    NNWindowConfig o0Window{2, 0b0000};               // layer 2, alone

    NNScheduler h0Scheduler(h0Node, h0Window);
    NNScheduler h1Scheduler(h1Node, h1Window);
    NNScheduler o0Scheduler(o0Node, o0Window);

    NNBackupStandby h0Standby(h0Cfg, testClock);  // H0 backing up H1

    NNDuplicateSuppressor o0Suppressor;  // O0 receives from H0/H1 -- the only node needing this here

    // Inject the fixed reference inputs directly -- x0/x1 have no NNNode
    // of their own, matching the earlier design decision that raw sensor
    // inputs are simulated as directly-injected packets, not NNNode instances.
    medium.injectDirect(makeInputPacket(0, 1.0f));
    medium.injectDirect(makeInputPacket(1, 2.0f));

    NetworkResult result;

    for (int iteration = 0; iteration < 200; iteration++) {
        // --- drain H0's inbox ---
        NNPacket pkt;
        while (h0Transport.receive(pkt)) {
            h0Node.onPacketReceived(pkt);
            h0Scheduler.onPacketObserved(pkt);
            h0Standby.onPacketObserved(pkt);
        }

        // --- drain H1's inbox, only if H1 is actually "alive" this pass ---
        if (h1IsAlive) {
            while (h1Transport.receive(pkt)) {
                h1Node.onPacketReceived(pkt);
                h1Scheduler.onPacketObserved(pkt);
            }
        } else {
            // H1 is "dead": drain its inbox without processing anything,
            // simulating a powered-off device that never runs its own logic.
            while (h1Transport.receive(pkt)) { /* discard */ }
        }

        // --- drain O0's inbox, through the duplicate suppressor ---
        while (o0Transport.receive(pkt)) {
            if (!o0Suppressor.shouldAccept(pkt)) continue;  // late-arrival suppression
            o0Node.onPacketReceived(pkt);
            o0Scheduler.onPacketObserved(pkt);
        }

        // --- tick everything ---
        h0Scheduler.tick();
        if (h1IsAlive) h1Scheduler.tick();
        h0Standby.tick();
        o0Scheduler.tick();

        // --- send whatever's ready ---
        NNPacket outPkt;
        if (h0Scheduler.hasOutputReady(outPkt)) h0Transport.send(outPkt);
        if (h1IsAlive && h1Scheduler.hasOutputReady(outPkt)) h1Transport.send(outPkt);
        if (h0Standby.hasOutputReady(outPkt)) {
            if (h0Standby.didSubstitute()) result.h0Substituted = true;
            if (h0Standby.didTearDown()) result.anyTeardown = true;
            h0Transport.send(outPkt);
        }
        if (o0Scheduler.hasOutputReady(outPkt)) {
            result.completed = true;
            result.finalOutput = outPkt.payload[0];
        }

        fakeClock += 10;  // advance simulated time each iteration

        if (result.completed) break;
    }

    return result;
}

// ============================================================================
// TESTS
// ============================================================================

void test_healthy_execution_no_backup_used() {
    NetworkResult r = runReferenceNetwork(/*h1IsAlive=*/true);
    assert(r.completed);
    assert(std::fabs(r.finalOutput - 3.5f) < TOLERANCE);
    assert(!r.h0Substituted);
    assert(!r.anyTeardown);
    printf("PASSED: healthy execution -- final output %.4f (expected 3.5), no backup used\n", r.finalOutput);
}

void test_backup_activates_when_h1_never_transmits() {
    NetworkResult r = runReferenceNetwork(/*h1IsAlive=*/false);
    assert(r.completed);
    assert(r.h0Substituted);
    assert(!r.anyTeardown);
    printf("PASSED: H0's backup standby correctly activated when H1 never transmitted\n");
}

void test_final_output_remains_3_5_after_failure() {
    NetworkResult r = runReferenceNetwork(/*h1IsAlive=*/false);
    assert(r.completed);
    assert(std::fabs(r.finalOutput - 3.5f) < TOLERANCE);
    printf("PASSED: final network output remains %.4f (expected 3.5) despite H1's failure "
           "-- O0 correctly treated H0's substitute as H1's own output\n", r.finalOutput);
}

void test_backup_weights_are_structurally_separate_from_own_weights() {
    NNNodeConfig h0Cfg = makeH0Config();
    // H0's own weights and the weights it holds on H1's behalf must be
    // genuinely different arrays with different content -- not aliased,
    // not accidentally identical due to a copy-paste config mistake.
    assert(h0Cfg.weights != h0Cfg.backupWeights);
    assert(h0Cfg.weights[0] != h0Cfg.backupWeights[0]);
    printf("PASSED: backup weights are structurally separate from the node's own weights "
           "(different arrays, different values: own=[%.2f,%.2f] vs backup=[%.2f,%.2f])\n",
           h0Cfg.weights[0], h0Cfg.weights[1], h0Cfg.backupWeights[0], h0Cfg.backupWeights[1]);
}

void test_multiple_simultaneous_failures_deferred() {
    // EXPLICITLY DEFERRED: scenarios where H0 ALSO fails (i.e. the node
    // backing up H1 is itself unavailable) are not implemented or tested,
    // per an earlier, direct decision in this project to defer this until
    // after the framework's UI is built. This placeholder exists so the
    // gap is documented in the test suite itself, not silently absent.
    printf("SKIPPED (by design): multiple simultaneous failures -- explicitly deferred, "
           "not yet implemented. See NNFailover.h's top-of-file design note.\n");
}

void test_late_arrival_does_not_create_duplicate_inputs() {
    // Run the network with H1 dead, exactly as the substitute-activation
    // test does -- H0's standby will substitute for H1, and O0 will
    // complete using that substitute.
    NetworkResult r = runReferenceNetwork(/*h1IsAlive=*/false);
    assert(r.completed);
    assert(r.h0Substituted);
    assert(std::fabs(r.finalOutput - 3.5f) < TOLERANCE);

    // Now simulate H1 turning out to have been merely SLOW, not dead: a
    // "late arrival" packet under H1's real identity shows up after the
    // pass already completed. Deliberately uses an obviously WRONG marker
    // value (999.0f, not H1's real 3.0) rather than a plausible one -- if
    // the suppressor failed to reject this and it somehow altered anything,
    // that would be immediately, unambiguously visible, rather than
    // silently masked by the late value happening to match the substitute.
    NNDuplicateSuppressor freshSuppressor;
    NNPacket substituteMarker{};
    substituteMarker.header.sourceAddress = encodeAddress({3, 1, 0, 0});  // H1's identity
    substituteMarker.header.flags = NN_FLAG_FAILOVER_SUBSTITUTE;
    substituteMarker.header.payloadCount = 1;
    substituteMarker.payload[0] = 3.0f;  // the value the substitute actually carried

    NNPacket lateRealH1{};
    lateRealH1.header.sourceAddress = encodeAddress({3, 1, 0, 0});  // same H1 identity
    lateRealH1.header.flags = 0;  // NOT a substitute -- genuinely the late original
    lateRealH1.header.payloadCount = 1;
    lateRealH1.payload[0] = 999.0f;  // deliberately wrong marker value

    assert(freshSuppressor.shouldAccept(substituteMarker));   // the substitute claimed the slot first
    assert(!freshSuppressor.shouldAccept(lateRealH1));         // late real packet correctly rejected

    // The network's own final output, already captured above, is also
    // confirmed unchanged -- there is no code path in this test by which
    // a post-completion late packet could have altered r.finalOutput,
    // which is itself worth stating plainly: O0's own NNNode sets
    // hasExecuted=true after its one execute() call and never runs a
    // second time within the same pass, so even without the suppressor,
    // a late packet arriving after completion could only affect a FUTURE
    // pass, never retroactively change a result already produced.
    printf("PASSED: late arrival of H1's original packet (with a deliberately wrong "
           "marker value 999.0) was correctly rejected by the duplicate suppressor; "
           "final network output remains %.4f, unaffected\n", r.finalOutput);
}

int main() {
    printf("=== Reference network integration tests ===\n\n");

    test_healthy_execution_no_backup_used();
    test_backup_activates_when_h1_never_transmits();
    test_final_output_remains_3_5_after_failure();
    test_backup_weights_are_structurally_separate_from_own_weights();
    test_late_arrival_does_not_create_duplicate_inputs();
    test_multiple_simultaneous_failures_deferred();

    printf("\nALL REFERENCE NETWORK TESTS PASSED\n");
    return 0;
}