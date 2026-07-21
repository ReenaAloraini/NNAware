// test_node_failure.cpp — tests for NNFailover.h: NNBackupStandby (event-
// driven, roster-completion-based failure detection with a clock-only
// resend grace period), NNResendResponder, NNDuplicateSuppressor, and the
// desktop-only validateBackupConfig() helper.

#include <cassert>
#include <cstdio>
#include <cstring>
#include "NNAddress.h"
#include "NNPacket.h"
#include "NNNode.h"
#include "NNFailover.h"

static unsigned long fakeClock = 0;
unsigned long testClock() { return fakeClock; }

// Shared across both sections below: a COPY of H1's real weights, used
// both as H1's actual weights (in the validator section) and as H0's
// backupWeights standing in for H1 (in both sections) -- confirmed
// identical before merging, so a single definition is correct, not a
// coincidence of two independently-typed literals.
static const float h1Weights[2] = {-1.0f, 2.0f};

// ============================================================================
// SECTION A — NNBackupStandby / NNResendResponder / NNDuplicateSuppressor
// ============================================================================

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
    cfg.backupWeights = h1Weights;
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

void runBackupStandbyTests() {
    // --- 1: no resend fires while the round is still incomplete —
    //        H2 (another sibling, not the target) hasn't gone yet ---
    {
        fakeClock = 0;
        NNNodeConfig h0Cfg = makeH0ConfigWithBackupForH1();
        NNBackupStandby standby(h0Cfg, testClock);

        // H2 never transmits -- the round is genuinely incomplete, so
        // detection must NOT fire, no matter how many ticks pass (proving
        // this really is event-driven, not a disguised timer).
        for (int i = 0; i < 20; i++) {
            standby.tick();
            fakeClock += 1000;
        }
        assert(!standby.didRequestResend());
        printf("A1 PASSED: detection correctly does NOT fire while the round "
               "is still incomplete (H2 hasn't gone yet), regardless of elapsed time\n");
    }

    // --- 2: HEALTHY PATH — H1 and H2 both transmit, H0 stays silent ---
    {
        fakeClock = 0;
        NNNodeConfig h0Cfg = makeH0ConfigWithBackupForH1();
        NNBackupStandby standby(h0Cfg, testClock);

        standby.onPacketObserved(makeSiblingPacket(2, 1, 5.0f));
        standby.onPacketObserved(makeSiblingPacket(1, 1, 3.0f));  // the target itself

        standby.tick();
        assert(!standby.didRequestResend());
        assert(!standby.didSubstitute());
        printf("A2 PASSED: healthy path — H1 and H2 both transmitted, "
               "H0's standby correctly took no action\n");
    }

    // --- 3: FAILURE DETECTED THE MOMENT THE ROUND COMPLETES ---
    {
        fakeClock = 12345;  // deliberately a weird, non-zero starting clock value
        NNNodeConfig h0Cfg = makeH0ConfigWithBackupForH1();
        NNBackupStandby standby(h0Cfg, testClock);

        standby.onPacketObserved(makeSiblingPacket(2, 1, 5.0f));  // H2 goes; H1 does not
        standby.tick();

        assert(standby.didRequestResend());
        NNPacket resendReq;
        assert(standby.hasOutputReady(resendReq));
        assert(resendReq.header.type == NNPacketType::CONTROL);
        NNAddress requested = decodeAddress(static_cast<uint16_t>(resendReq.payload[0]));
        assert(requested.nodeId == 1 && requested.layerId == 1);

        printf("A3 PASSED: failure detected the instant the round completed "
               "(H2 done, H1 missing) -- purely event-driven, no clock consulted "
               "for detection itself\n");
    }

    // --- 4: resend succeeds — H1 was just slow, genuinely retransmits ---
    {
        fakeClock = 0;
        NNNodeConfig h0Cfg = makeH0ConfigWithBackupForH1();
        NNBackupStandby standby(h0Cfg, testClock);
        NNResendResponder h1Responder({1, 1, 0, 0});

        standby.onPacketObserved(makeSiblingPacket(2, 1, 5.0f));
        standby.onPacketObserved(makeSiblingPacket(0, 0, 1.0f));
        standby.onPacketObserved(makeSiblingPacket(1, 0, 2.0f));

        standby.tick();
        NNPacket resendReq;
        assert(standby.hasOutputReady(resendReq));

        h1Responder.onPacketObserved(resendReq, /*haveOutput=*/true, /*value=*/3.0f);
        NNPacket h1Retransmit;
        assert(h1Responder.hasResendReady(h1Retransmit));

        standby.onPacketObserved(h1Retransmit);
        fakeClock += 100;
        standby.tick();

        assert(!standby.didSubstitute());
        assert(!standby.didTearDown());
        printf("A4 PASSED: resend succeeds — H1 genuinely retransmitted after "
               "the request, standby correctly stood down\n");
    }

    // --- 5: resend fails, substitute used ---
    {
        fakeClock = 0;
        NNNodeConfig h0Cfg = makeH0ConfigWithBackupForH1();
        NNBackupStandby standby(h0Cfg, testClock);

        standby.onPacketObserved(makeSiblingPacket(2, 1, 5.0f));
        standby.onPacketObserved(makeSiblingPacket(0, 0, 1.0f));
        standby.onPacketObserved(makeSiblingPacket(1, 0, 2.0f));

        standby.tick();
        NNPacket resendReq;
        assert(standby.hasOutputReady(resendReq));

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
        printf("A5 PASSED: resend fails, substitute correctly computed 3.0 under H1's identity\n");
    }

    // --- 6: resend fails, no inputs -> real teardown ---
    {
        fakeClock = 0;
        NNNodeConfig h0Cfg = makeH0ConfigWithBackupForH1();
        NNBackupStandby standby(h0Cfg, testClock);

        standby.onPacketObserved(makeSiblingPacket(2, 1, 5.0f));  // round completes, no x0/x1 given

        standby.tick();
        NNPacket resendReq;
        assert(standby.hasOutputReady(resendReq));
        fakeClock += 100;
        standby.tick();

        assert(standby.didTearDown());
        NNPacket teardownPkt;
        assert(standby.hasOutputReady(teardownPkt));
        assert(teardownPkt.header.type == NNPacketType::TEARDOWN);
        printf("A6 PASSED: resend fails AND no inputs -> real TEARDOWN dispatched\n");
    }

    // --- 7: late-arrival suppression ---
    {
        NNDuplicateSuppressor suppressor;
        NNPacket substitutePkt{};
        substitutePkt.header.sourceAddress = encodeAddress({1, 1, 0, 0});
        substitutePkt.header.flags = NN_FLAG_FAILOVER_SUBSTITUTE;
        substitutePkt.header.payloadCount = 1;
        substitutePkt.payload[0] = 3.0f;

        NNPacket lateRealPkt{};
        lateRealPkt.header.sourceAddress = encodeAddress({1, 1, 0, 0});
        lateRealPkt.header.flags = 0;
        lateRealPkt.header.payloadCount = 1;
        lateRealPkt.payload[0] = 999.0f;

        assert(suppressor.shouldAccept(substitutePkt));
        assert(!suppressor.shouldAccept(lateRealPkt));
        printf("A7 PASSED: late-arrival suppression still works correctly\n");
    }

    // --- 8: reset clears layerObservedMask, not just the resend/substitute flags ---
    {
        fakeClock = 0;
        NNNodeConfig h0Cfg = makeH0ConfigWithBackupForH1();
        NNBackupStandby standby(h0Cfg, testClock);

        standby.onPacketObserved(makeSiblingPacket(2, 1, 5.0f));
        standby.tick();
        assert(standby.didRequestResend());

        standby.resetForNextPass();
        assert(!standby.didRequestResend());

        standby.tick();
        assert(!standby.didRequestResend());
        printf("A8 PASSED: resetForNextPass() correctly clears all state, "
               "including layerObservedMask\n");
    }
}

// ============================================================================
// SECTION B — validateBackupConfig()
// ============================================================================

NNNodeConfig makeRealH1Config() {
    NNNodeConfig cfg{};
    cfg.address = {1, 1, 0, 0};              // H1: nodeId=1, layer=1
    cfg.predecessorMask = 0b11;
    cfg.weights = h1Weights;
    cfg.weightCount = 2;
    cfg.activationType = NNActivationType::RELU;
    return cfg;
}

NNNodeConfig makeCorrectH0BackupConfigForValidation() {
    NNNodeConfig cfg{};
    cfg.address = {0, 1, 0, 0};              // H0: nodeId=0, layer=1 (same layer as H1)
    cfg.hasBackupRole = true;
    cfg.backupTargetAddress = {1, 1, 0, 0};   // matches H1's real address
    cfg.backupTargetPredecessorMask = 0b11;   // matches H1's real predecessorMask
    cfg.backupTargetActivationType = NNActivationType::RELU;  // matches H1's real activationType
    cfg.backupWeights = h1Weights;
    cfg.backupWeightCount = 2;                // matches H1's real weightCount
    cfg.layerRosterMask = 0b111;
    return cfg;
}

void runValidatorTests() {
    NNNodeConfig realH1 = makeRealH1Config();

    // --- 9: a correctly-matching pair reports fully valid ---
    {
        NNNodeConfig backupCfg = makeCorrectH0BackupConfigForValidation();
        auto result = validateBackupConfig(backupCfg, realH1);
        assert(result.isValid);
        assert(result.issueCount == 0);
        printf("B1 PASSED: correctly-matching backup config reports valid, zero issues\n");
    }

    // --- 10: hasBackupRole == false is reported as its own issue ---
    {
        NNNodeConfig backupCfg = makeCorrectH0BackupConfigForValidation();
        backupCfg.hasBackupRole = false;
        auto result = validateBackupConfig(backupCfg, realH1);
        assert(!result.isValid);
        assert(result.issueCount == 1);
        printf("B2 PASSED: hasBackupRole=false correctly reported, no further checks attempted\n");
    }

    // --- 11: mismatched activation type (the exact scenario that prompted this validator) ---
    {
        NNNodeConfig backupCfg = makeCorrectH0BackupConfigForValidation();
        backupCfg.backupTargetActivationType = NNActivationType::LINEAR;  // WRONG -- H1 is really RELU
        auto result = validateBackupConfig(backupCfg, realH1);
        assert(!result.isValid);
        bool found = false;
        for (uint8_t i = 0; i < result.issueCount; i++) {
            if (strstr(result.issues[i], "activationType") != nullptr) found = true;
        }
        assert(found);
        printf("B3 PASSED: mismatched backupTargetActivationType correctly caught\n");
    }

    // --- 12: mismatched predecessor mask ---
    {
        NNNodeConfig backupCfg = makeCorrectH0BackupConfigForValidation();
        backupCfg.backupTargetPredecessorMask = 0b01;  // WRONG -- H1 really needs 0b11
        auto result = validateBackupConfig(backupCfg, realH1);
        assert(!result.isValid);
        bool found = false;
        for (uint8_t i = 0; i < result.issueCount; i++) {
            if (strstr(result.issues[i], "PredecessorMask") != nullptr) found = true;
        }
        assert(found);
        printf("B4 PASSED: mismatched backupTargetPredecessorMask correctly caught\n");
    }

    // --- 13: mismatched weight count ---
    {
        NNNodeConfig backupCfg = makeCorrectH0BackupConfigForValidation();
        backupCfg.backupWeightCount = 3;  // WRONG -- H1 really has 2
        auto result = validateBackupConfig(backupCfg, realH1);
        assert(!result.isValid);
        bool found = false;
        for (uint8_t i = 0; i < result.issueCount; i++) {
            if (strstr(result.issues[i], "weightCount") != nullptr) found = true;
        }
        assert(found);
        printf("B5 PASSED: mismatched backupWeightCount correctly caught\n");
    }

    // --- 14: mismatched target address ---
    {
        NNNodeConfig backupCfg = makeCorrectH0BackupConfigForValidation();
        backupCfg.backupTargetAddress = {2, 1, 0, 0};  // WRONG -- claims to back up H2, not H1
        auto result = validateBackupConfig(backupCfg, realH1);
        assert(!result.isValid);
        bool found = false;
        for (uint8_t i = 0; i < result.issueCount; i++) {
            if (strstr(result.issues[i], "backupTargetAddress does not match") != nullptr) found = true;
        }
        assert(found);
        printf("B6 PASSED: mismatched backupTargetAddress correctly caught\n");
    }

    // --- 15: cross-layer backup relationship ---
    {
        NNNodeConfig backupCfg = makeCorrectH0BackupConfigForValidation();
        backupCfg.address = {0, 5, 0, 0};  // WRONG -- this node is now in layer 5, not layer 1
        auto result = validateBackupConfig(backupCfg, realH1);
        assert(!result.isValid);
        bool found = false;
        for (uint8_t i = 0; i < result.issueCount; i++) {
            if (strstr(result.issues[i], "SAME layer") != nullptr) found = true;
        }
        assert(found);
        printf("B7 PASSED: cross-layer backup relationship correctly caught\n");
    }

    // --- 16: multiple simultaneous issues are ALL reported, not just the first ---
    {
        NNNodeConfig backupCfg = makeCorrectH0BackupConfigForValidation();
        backupCfg.backupTargetActivationType = NNActivationType::LINEAR;
        backupCfg.backupWeightCount = 99;
        auto result = validateBackupConfig(backupCfg, realH1);
        assert(!result.isValid);
        assert(result.issueCount == 2);
        printf("B8 PASSED: multiple simultaneous issues (%d) all reported in one call, "
               "not just the first found\n", result.issueCount);
    }
}

int main() {
    printf("=== Section A: NNBackupStandby / retransmission / teardown ===\n");
    runBackupStandbyTests();
    printf("\n=== Section B: validateBackupConfig ===\n");
    runValidatorTests();
    printf("\nALL NODE FAILURE TESTS PASSED\n");
    return 0;
}