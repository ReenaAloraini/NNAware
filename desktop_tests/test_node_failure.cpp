#include <cassert>
#include <cstdio>
#include "NNAddress.h"
#include "NNPacket.h"
#include "NNNode.h"
#include "NNFailover.h"


static unsigned long fakeClock = 0;
unsigned long testClock() { return fakeClock; }

static const float h1BackupWeights[2] = {-1.0f, 2.0f};  // COPY of H1's real weights

// Layer 1 roster: H0 (node 0), H1 (node 1), H2 (node 2) -- three siblings,
// so "everyone else besides the target" is a genuine, non-trivial check.
NNNodeConfig makeH0ConfigWithBackupForH1() {
    NNNodeConfig cfg{};
    cfg.address = {0, 1, 0, 0};                    // H0 itself: nodeId=0, layer=1
    cfg.predecessorMask = 0b11;
    cfg.weights = nullptr;
    cfg.weightCount = 0;
    cfg.activationType = NNActivationType::RELU;

    cfg.hasBackupRole = true;
    cfg.backupTargetAddress = {1, 1, 0, 0};        // backs up H1: nodeId=1, layer=1 (SAME layer)
    cfg.backupTargetPredecessorMask = 0b11;
    cfg.backupTargetActivationType = NNActivationType::RELU;
    cfg.backupWeights = h1BackupWeights;
    cfg.backupWeightCount = 2;
    cfg.resendGraceMs = 50;
    cfg.layerRosterMask = 0b111;                   // H0, H1, H2 -- three siblings in layer 1
    return cfg;
}

NNPacket makeSiblingPacket(uint8_t nodeId, uint8_t layerId, float value) {
    NNPacket pkt{};
    pkt.header.sourceAddress = encodeAddress({nodeId, layerId, 0, 0});
    pkt.header.payloadCount = 1;
    pkt.payload[0] = value;
    return pkt;
}

int main() {
    // --- TEST 1: no resend fires while the round is still incomplete —
    //             H2 (another sibling, not the target) hasn't gone yet ---
    {
        fakeClock = 0;
        NNNodeConfig h0Cfg = makeH0ConfigWithBackupForH1();
        NNBackupStandby standby(h0Cfg, testClock);

        // H2 (nodeId 2) never transmits in this test -- the round is
        // genuinely incomplete, so detection must NOT fire, no matter how
        // many ticks pass (proving this really is event-driven, not a
        // disguised timer).
        for (int i = 0; i < 20; i++) {
            standby.tick();
            fakeClock += 1000;
        }
        assert(!standby.didRequestResend());
        printf("TEST 1 PASSED: detection correctly does NOT fire while the round "
               "is still incomplete (a non-target sibling, H2, hasn't gone yet), "
               "regardless of elapsed time\n");
    }

    // --- TEST 2: HEALTHY PATH — H1 and H2 both transmit, H0 stays silent ---
    {
        fakeClock = 0;
        NNNodeConfig h0Cfg = makeH0ConfigWithBackupForH1();
        NNBackupStandby standby(h0Cfg, testClock);

        standby.onPacketObserved(makeSiblingPacket(2, 1, 5.0f));   // H2 goes
        standby.onPacketObserved(makeSiblingPacket(1, 1, 3.0f));   // H1 (the target) goes too

        standby.tick();  // round IS complete now, but target already observed -> no action
        assert(!standby.didRequestResend());
        assert(!standby.didSubstitute());
        printf("TEST 2 PASSED: healthy path — H1 and H2 both transmitted, "
               "H0's standby correctly took no action\n");
    }

    // --- TEST 3: FAILURE DETECTED THE MOMENT THE ROUND COMPLETES —
    //             H2 goes, H1 (target) never does -> detection fires
    //             IMMEDIATELY on the very next tick(), no clock involved ---
    {
        fakeClock = 12345;  // deliberately a weird, non-zero starting clock value
        NNNodeConfig h0Cfg = makeH0ConfigWithBackupForH1();
        NNBackupStandby standby(h0Cfg, testClock);

        standby.onPacketObserved(makeSiblingPacket(2, 1, 5.0f));  // H2 goes; H1 does not

        standby.tick();  // round is now complete (H2 done, only H1/target missing)

        assert(standby.didRequestResend());
        NNPacket resendReq;
        assert(standby.hasOutputReady(resendReq));
        assert(resendReq.header.type == NNPacketType::CONTROL);
        NNAddress requested = decodeAddress(static_cast<uint16_t>(resendReq.payload[0]));
        assert(requested.nodeId == 1 && requested.layerId == 1);  // asking for H1 specifically

        printf("TEST 3 PASSED: failure detected the instant the round completed "
               "(H2 done, H1 missing) -- purely event-driven, no clock consulted "
               "for detection itself\n");
    }

    // --- TEST 4: resend succeeds — H1 was just slow, genuinely retransmits ---
    {
        fakeClock = 0;
        NNNodeConfig h0Cfg = makeH0ConfigWithBackupForH1();
        NNBackupStandby standby(h0Cfg, testClock);
        NNResendResponder h1Responder({1, 1, 0, 0});

        standby.onPacketObserved(makeSiblingPacket(2, 1, 5.0f));
        standby.onPacketObserved(makeSiblingPacket(0, 0, 1.0f));  // x0 input, for backup weights use
        standby.onPacketObserved(makeSiblingPacket(1, 0, 2.0f));  // x1 input

        standby.tick();  // detects failure, queues resend (clock is now consulted for the first time)
        NNPacket resendReq;
        assert(standby.hasOutputReady(resendReq));

        h1Responder.onPacketObserved(resendReq, /*haveOutput=*/true, /*value=*/3.0f);
        NNPacket h1Retransmit;
        assert(h1Responder.hasResendReady(h1Retransmit));

        standby.onPacketObserved(h1Retransmit);  // H1 speaks for itself now
        fakeClock += 100;  // past resendGraceMs
        standby.tick();

        assert(!standby.didSubstitute());
        assert(!standby.didTearDown());
        printf("TEST 4 PASSED: resend succeeds — H1 genuinely retransmitted after "
               "the request, standby correctly stood down\n");
    }

    // --- TEST 5: resend fails, substitute used ---
    {
        fakeClock = 0;
        NNNodeConfig h0Cfg = makeH0ConfigWithBackupForH1();
        NNBackupStandby standby(h0Cfg, testClock);

        standby.onPacketObserved(makeSiblingPacket(2, 1, 5.0f));
        standby.onPacketObserved(makeSiblingPacket(0, 0, 1.0f));
        standby.onPacketObserved(makeSiblingPacket(1, 0, 2.0f));

        standby.tick();
        NNPacket resendReq;
        assert(standby.hasOutputReady(resendReq));  // drain it

        fakeClock += 100;  // H1 never responds
        standby.tick();

        assert(standby.didSubstitute());
        NNPacket substitutePkt;
        assert(standby.hasOutputReady(substitutePkt));
        NNAddress src = decodeAddress(substitutePkt.header.sourceAddress);
        assert(src.nodeId == 1 && src.layerId == 1);
        assert(substitutePkt.header.flags & NN_FLAG_FAILOVER_SUBSTITUTE);
        float diff = substitutePkt.payload[0] - 3.0f;  // hand-computed, H1's real weights
        assert(diff > -0.0001f && diff < 0.0001f);
        printf("TEST 5 PASSED: resend fails, substitute correctly computed 3.0 "
               "under H1's identity\n");
    }

    // --- TEST 6: resend fails, no inputs -> real teardown ---
    {
        fakeClock = 0;
        NNNodeConfig h0Cfg = makeH0ConfigWithBackupForH1();
        NNBackupStandby standby(h0Cfg, testClock);

        standby.onPacketObserved(makeSiblingPacket(2, 1, 5.0f));  // round completes, but no x0/x1 given

        standby.tick();
        NNPacket resendReq;
        assert(standby.hasOutputReady(resendReq));
        fakeClock += 100;
        standby.tick();

        assert(standby.didTearDown());
        NNPacket teardownPkt;
        assert(standby.hasOutputReady(teardownPkt));
        assert(teardownPkt.header.type == NNPacketType::TEARDOWN);
        printf("TEST 6 PASSED: resend fails AND no inputs -> real TEARDOWN dispatched\n");
    }

    // --- TEST 7: reset clears the new layerObservedMask too ---
    {
        fakeClock = 0;
        NNNodeConfig h0Cfg = makeH0ConfigWithBackupForH1();
        NNBackupStandby standby(h0Cfg, testClock);

        standby.onPacketObserved(makeSiblingPacket(2, 1, 5.0f));
        standby.tick();
        assert(standby.didRequestResend());

        standby.resetForNextPass();
        assert(!standby.didRequestResend());

        // After reset, the round must be considered incomplete again --
        // detection must NOT immediately re-fire just because H2's
        // transmission from the PREVIOUS pass is somehow still "counted."
        standby.tick();
        assert(!standby.didRequestResend());
        printf("TEST 7 PASSED: resetForNextPass() correctly clears layerObservedMask, "
               "not just the resend/substitute flags\n");
    }

    printf("\nALL EVENT-DRIVEN DETECTION TESTS PASSED\n");
    return 0;
}