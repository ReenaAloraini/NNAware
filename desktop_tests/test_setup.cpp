#include <cassert>
#include <cstdio>
#include <cstring>
#include "NNSetupProtocol.h"
#include "NNTransportLoopback.h"
#include "NNFailover.h"

static unsigned long fakeClock = 0;
unsigned long testClock() { return fakeClock; }

class TestConfigStore : public NNConfigStore {
public:
    bool load(NNPersistedConfig& out) override {
        if (!hasSaved) return false;
        out = saved;
        return true;
    }
    bool save(const NNPersistedConfig& cfg) override {
        saved = cfg;
        hasSaved = true;
        return true;
    }
    void clear() override { hasSaved = false; }

    bool hasSaved = false;
    NNPersistedConfig saved{};
};

static bool sameAddress(const NNAddress& a, const NNAddress& b) {
    return a.nodeId == b.nodeId && a.layerId == b.layerId &&
           a.clusterId == b.clusterId && a.reserved == b.reserved;
}

// Pulls exactly one queued packet out of an agent's transport, failing loud
// if there isn't one -- every handler in NNSetupAgent replies to something.
static NNPacket drain(NNTransportLoopback& transport) {
    NNPacket pkt{};
    bool got = transport.receive(pkt);
    assert(got);
    return pkt;
}

int main() {
    const uint64_t HW_ID = 0x1122334455667788ULL;
    const uint64_t OTHER_HW_ID = 0x9999999999999999ULL;

    // --- TEST 1: happy path, no backup role, all the way to RUNNING ---
    {
        NNTransportLoopback transport;
        TestConfigStore store;
        NNSetupAgent agent(HW_ID, transport, store);
        agent.begin();
        assert(agent.getState() == NNSetupState::UNCONFIGURED);

        agent.tick(1000); // lastAnnounceMs starts at 0, announceIntervalMs at 1000 -- tick(0) would not fire yet
        NNPacket hello = drain(transport);
        assert(getSetupOpcode(hello) == NNSetupOpcode::HELLO);
        assert(unpackSetupMessage<NNHelloMsg>(hello).hardwareId == HW_ID);

        NNAddress addr{0, 1, 0, 0};
        NNPacket assignPkt{};
        NNAssignAddressMsg assignMsg{HW_ID, encodeAddress(addr), 0};
        packSetupMessage(assignPkt, NNSetupOpcode::ASSIGN_ADDRESS, assignMsg, 1);
        agent.onSetupPacket(assignPkt);
        NNPacket ack1 = drain(transport);
        assert(getSetupOpcode(ack1) == NNSetupOpcode::ACK);
        assert(unpackSetupMessage<NNAckMsg>(ack1).ackedOpcode == static_cast<uint8_t>(NNSetupOpcode::ASSIGN_ADDRESS));

        const float weights[3] = {0.1f, -0.2f, 0.3f};
        NNTopologyInfoMsg topoMsg{HW_ID, 0b11, 0, 2, static_cast<uint8_t>(NNActivationType::RELU), 3, 0, 0};
        NNPacket topoPkt{};
        packSetupMessage(topoPkt, NNSetupOpcode::TOPOLOGY_INFO, topoMsg, 2);
        agent.onSetupPacket(topoPkt);
        NNPacket ack2 = drain(transport);
        assert(getSetupOpcode(ack2) == NNSetupOpcode::ACK);
        assert(agent.getState() == NNSetupState::RECEIVING_CONFIG);

        NNWeightsChunkMsg chunkMsg{};
        chunkMsg.hardwareId = HW_ID;
        chunkMsg.chunkIndex = 0;
        chunkMsg.chunkCount = 1;
        chunkMsg.valuesInChunk = 3;
        memcpy(chunkMsg.values, weights, sizeof(weights));
        NNPacket chunkPkt{};
        packSetupMessage(chunkPkt, NNSetupOpcode::WEIGHTS_CHUNK, chunkMsg, 3);
        agent.onSetupPacket(chunkPkt);
        NNPacket ack3 = drain(transport);
        assert(getSetupOpcode(ack3) == NNSetupOpcode::ACK);
        assert(agent.getState() == NNSetupState::VERIFYING);

        NNCommitRequestMsg commitReq{HW_ID};
        NNPacket commitPkt{};
        packSetupMessage(commitPkt, NNSetupOpcode::COMMIT_REQUEST, commitReq, 4);
        agent.onSetupPacket(commitPkt);
        NNPacket reply = drain(transport);
        assert(getSetupOpcode(reply) == NNSetupOpcode::COMMIT_REPLY);
        auto replyMsg = unpackSetupMessage<NNCommitReplyMsg>(reply);
        assert(replyMsg.success == 1);
        assert(replyMsg.hasBackupRole == 0);
        assert(replyMsg.backupChecksum == 0);
        uint8_t expectedChecksum = computeChecksum(reinterpret_cast<const uint8_t*>(weights), sizeof(weights));
        assert(replyMsg.computedChecksum == expectedChecksum);
        assert(agent.getState() == NNSetupState::CONFIGURED);

        NNStartMsg startMsg{};
        NNPacket startPkt{};
        packSetupMessage(startPkt, NNSetupOpcode::START, startMsg, 5);
        agent.onSetupPacket(startPkt);
        assert(agent.isRunning());

        const NNNodeConfig& cfg = agent.getNodeConfig();
        assert(sameAddress(cfg.address, addr));
        assert(cfg.predecessorMask == 0b11);
        assert(cfg.successorLayerId == 2);
        assert(cfg.activationType == NNActivationType::RELU);
        assert(cfg.weightCount == 3);
        assert(cfg.predecessorLayerId == 0);
        assert(!cfg.hasBackupRole);
        for (int i = 0; i < 3; i++) assert(cfg.weights[i] == weights[i]);
        assert(agent.getWindowConfig().ownLayerId == 1);

        printf("TEST 1 PASSED: happy path with no backup role reaches RUNNING with correct config\n");
    }

    // --- TEST 2: happy path WITH a backup role, integration-checked against NNBackupStandby ---
    {
        NNTransportLoopback transport;
        TestConfigStore store;
        NNSetupAgent agent(HW_ID, transport, store);
        agent.begin();

        NNAddress addr{0, 1, 0, 0};
        NNPacket assignPkt{};
        NNAssignAddressMsg assignMsg{HW_ID, encodeAddress(addr), 0};
        packSetupMessage(assignPkt, NNSetupOpcode::ASSIGN_ADDRESS, assignMsg, 1);
        agent.onSetupPacket(assignPkt);
        drain(transport);

        const float weights[2] = {1.5f, -2.5f};
        NNTopologyInfoMsg topoMsg{HW_ID, 0b11, 0, 2, static_cast<uint8_t>(NNActivationType::RELU), 2, 0, 0};
        NNPacket topoPkt{};
        packSetupMessage(topoPkt, NNSetupOpcode::TOPOLOGY_INFO, topoMsg, 2);
        agent.onSetupPacket(topoPkt);
        drain(transport);

        NNAddress backupTarget{1, 1, 0, 0}; // SAME layer as addr, required by NNBackupStandby
        NNBackupRoleInfoMsg roleMsg{HW_ID, encodeAddress(backupTarget), 0b11, 0b111,
                                     static_cast<uint8_t>(NNActivationType::RELU), 2, 50, 0};
        NNPacket rolePkt{};
        packSetupMessage(rolePkt, NNSetupOpcode::BACKUP_ROLE_INFO, roleMsg, 3);
        agent.onSetupPacket(rolePkt);
        NNPacket roleAck = drain(transport);
        assert(unpackSetupMessage<NNAckMsg>(roleAck).ackedOpcode == static_cast<uint8_t>(NNSetupOpcode::BACKUP_ROLE_INFO));

        const float backupWeights[2] = {-1.0f, 2.0f};
        NNWeightsChunkMsg primaryChunk{};
        primaryChunk.hardwareId = HW_ID;
        primaryChunk.chunkIndex = 0;
        primaryChunk.chunkCount = 1;
        primaryChunk.valuesInChunk = 2;
        memcpy(primaryChunk.values, weights, sizeof(weights));
        NNPacket primaryChunkPkt{};
        packSetupMessage(primaryChunkPkt, NNSetupOpcode::WEIGHTS_CHUNK, primaryChunk, 4);
        agent.onSetupPacket(primaryChunkPkt);
        drain(transport);
        assert(agent.getState() == NNSetupState::RECEIVING_CONFIG); // primary done, backup still pending

        NNWeightsChunkMsg backupChunk{};
        backupChunk.hardwareId = HW_ID;
        backupChunk.chunkIndex = 0;
        backupChunk.chunkCount = 1;
        backupChunk.valuesInChunk = 2;
        memcpy(backupChunk.values, backupWeights, sizeof(backupWeights));
        NNPacket backupChunkPkt{};
        packSetupMessage(backupChunkPkt, NNSetupOpcode::BACKUP_WEIGHTS_CHUNK, backupChunk, 5);
        agent.onSetupPacket(backupChunkPkt);
        drain(transport);
        assert(agent.getState() == NNSetupState::VERIFYING);

        NNCommitRequestMsg commitReq{HW_ID};
        NNPacket commitPkt{};
        packSetupMessage(commitPkt, NNSetupOpcode::COMMIT_REQUEST, commitReq, 6);
        agent.onSetupPacket(commitPkt);
        NNPacket reply = drain(transport);
        auto replyMsg = unpackSetupMessage<NNCommitReplyMsg>(reply);
        assert(replyMsg.success == 1);
        assert(replyMsg.hasBackupRole == 1);
        uint8_t expectedBackupChecksum = computeChecksum(reinterpret_cast<const uint8_t*>(backupWeights), sizeof(backupWeights));
        assert(replyMsg.backupChecksum == expectedBackupChecksum);
        assert(agent.getState() == NNSetupState::CONFIGURED);

        const NNNodeConfig& cfg = agent.getNodeConfig();
        assert(cfg.hasBackupRole);
        assert(sameAddress(cfg.backupTargetAddress, backupTarget));
        assert(cfg.backupWeightCount == 2);
        assert(cfg.backupTargetPredecessorLayerId == 0);
        for (int i = 0; i < 2; i++) assert(cfg.backupWeights[i] == backupWeights[i]);

        // Integration check: the resulting config must not trip NNBackupStandby's
        // same-layer assertion -- confirms the setup phase actually produces a
        // usable backup configuration, not just one that looks right on paper.
        NNBackupStandby standby(cfg, testClock);
        assert(standby.isActive());

        printf("TEST 2 PASSED: happy path with a backup role reaches CONFIGURED with "
               "correct backup fields, and the resulting config builds a valid NNBackupStandby\n");
    }

    // --- TEST 3: BACKUP_ROLE_INFO arriving AFTER primary weights already
    //             completed demotes VERIFYING back to RECEIVING_CONFIG until
    //             backup weights land too ---
    {
        NNTransportLoopback transport;
        TestConfigStore store;
        NNSetupAgent agent(HW_ID, transport, store);
        agent.begin();

        NNAddress addr{0, 1, 0, 0};
        NNPacket assignPkt{};
        NNAssignAddressMsg assignMsg{HW_ID, encodeAddress(addr), 0};
        packSetupMessage(assignPkt, NNSetupOpcode::ASSIGN_ADDRESS, assignMsg, 1);
        agent.onSetupPacket(assignPkt);
        drain(transport);

        const float weights[2] = {0.5f, 0.25f};
        NNTopologyInfoMsg topoMsg{HW_ID, 0b11, 0, 2, static_cast<uint8_t>(NNActivationType::RELU), 2, 0, 0};
        NNPacket topoPkt{};
        packSetupMessage(topoPkt, NNSetupOpcode::TOPOLOGY_INFO, topoMsg, 2);
        agent.onSetupPacket(topoPkt);
        drain(transport);

        NNWeightsChunkMsg chunkMsg{};
        chunkMsg.hardwareId = HW_ID;
        chunkMsg.chunkIndex = 0;
        chunkMsg.chunkCount = 1;
        chunkMsg.valuesInChunk = 2;
        memcpy(chunkMsg.values, weights, sizeof(weights));
        NNPacket chunkPkt{};
        packSetupMessage(chunkPkt, NNSetupOpcode::WEIGHTS_CHUNK, chunkMsg, 3);
        agent.onSetupPacket(chunkPkt);
        drain(transport);
        assert(agent.getState() == NNSetupState::VERIFYING); // no backup role yet -> vacuously complete

        NNAddress backupTarget{1, 1, 0, 0};
        NNBackupRoleInfoMsg roleMsg{HW_ID, encodeAddress(backupTarget), 0b11, 0b111,
                                     static_cast<uint8_t>(NNActivationType::RELU), 1, 50, 0};
        NNPacket rolePkt{};
        packSetupMessage(rolePkt, NNSetupOpcode::BACKUP_ROLE_INFO, roleMsg, 4);
        agent.onSetupPacket(rolePkt);
        drain(transport);
        assert(agent.getState() == NNSetupState::RECEIVING_CONFIG); // demoted -- backup weights still missing

        const float backupWeights[1] = {9.5f};
        NNWeightsChunkMsg backupChunk{};
        backupChunk.hardwareId = HW_ID;
        backupChunk.chunkIndex = 0;
        backupChunk.chunkCount = 1;
        backupChunk.valuesInChunk = 1;
        memcpy(backupChunk.values, backupWeights, sizeof(backupWeights));
        NNPacket backupChunkPkt{};
        packSetupMessage(backupChunkPkt, NNSetupOpcode::BACKUP_WEIGHTS_CHUNK, backupChunk, 5);
        agent.onSetupPacket(backupChunkPkt);
        drain(transport);
        assert(agent.getState() == NNSetupState::VERIFYING); // re-promoted now that both are complete

        printf("TEST 3 PASSED: late BACKUP_ROLE_INFO correctly demotes VERIFYING back to "
               "RECEIVING_CONFIG, and re-promotes once backup weights complete\n");
    }

    // --- TEST 4: packets for a different hardwareId are fully ignored ---
    {
        NNTransportLoopback transport;
        TestConfigStore store;
        NNSetupAgent agent(HW_ID, transport, store);
        agent.begin();
        assert(agent.getState() == NNSetupState::UNCONFIGURED);

        NNAddress addr{5, 3, 0, 0};
        NNPacket assignPkt{};
        NNAssignAddressMsg assignMsg{OTHER_HW_ID, encodeAddress(addr), 0};
        packSetupMessage(assignPkt, NNSetupOpcode::ASSIGN_ADDRESS, assignMsg, 1);
        agent.onSetupPacket(assignPkt);

        assert(agent.getState() == NNSetupState::UNCONFIGURED); // untouched
        NNPacket unused{};
        assert(!transport.receive(unused)); // no ACK sent for a foreign hardwareId

        printf("TEST 4 PASSED: packets addressed to a different hardwareId are ignored\n");
    }

    // --- TEST 5: a chunk index past what TOPOLOGY_INFO declared produces a real NACK ---
    {
        NNTransportLoopback transport;
        TestConfigStore store;
        NNSetupAgent agent(HW_ID, transport, store);
        agent.begin();

        NNAddress addr{0, 1, 0, 0};
        NNPacket assignPkt{};
        NNAssignAddressMsg assignMsg{HW_ID, encodeAddress(addr), 0};
        packSetupMessage(assignPkt, NNSetupOpcode::ASSIGN_ADDRESS, assignMsg, 1);
        agent.onSetupPacket(assignPkt);
        drain(transport);

        NNTopologyInfoMsg topoMsg{HW_ID, 0b1, 0, 2, static_cast<uint8_t>(NNActivationType::RELU), 1, 0, 0};
        NNPacket topoPkt{};
        packSetupMessage(topoPkt, NNSetupOpcode::TOPOLOGY_INFO, topoMsg, 2);
        agent.onSetupPacket(topoPkt);
        drain(transport); // expectedChunkCount == 1, so only chunkIndex 0 is valid

        NNWeightsChunkMsg badChunk{};
        badChunk.hardwareId = HW_ID;
        badChunk.chunkIndex = 5;
        badChunk.chunkCount = 1;
        badChunk.valuesInChunk = 1;
        badChunk.values[0] = 1.0f;
        NNPacket badChunkPkt{};
        packSetupMessage(badChunkPkt, NNSetupOpcode::WEIGHTS_CHUNK, badChunk, 3);
        agent.onSetupPacket(badChunkPkt);

        NNPacket nack = drain(transport);
        assert(getSetupOpcode(nack) == NNSetupOpcode::NACK);
        auto nackMsg = unpackSetupMessage<NNNackMsg>(nack);
        assert(nackMsg.reasonCode == NN_SETUP_NACK_BAD_CHUNK_INDEX);
        assert(nackMsg.nackedOpcode == static_cast<uint8_t>(NNSetupOpcode::WEIGHTS_CHUNK));
        assert(agent.getState() == NNSetupState::RECEIVING_CONFIG); // unchanged, not silently dropped into VERIFYING

        printf("TEST 5 PASSED: an out-of-range chunk index produces a NACK, not a silent drop\n");
    }

    // --- TEST 6: persistence round-trip across a simulated reboot, including backup fields ---
    {
        NNTransportLoopback transport;
        TestConfigStore store;
        NNAddress addr{2, 1, 0, 0};
        NNAddress backupTarget{3, 1, 0, 0};
        const float weights[2] = {4.0f, -4.0f};
        const float backupWeights[1] = {7.0f};

        {
            NNSetupAgent agent(HW_ID, transport, store);
            agent.begin();

            NNPacket assignPkt{};
            NNAssignAddressMsg assignMsg{HW_ID, encodeAddress(addr), 0};
            packSetupMessage(assignPkt, NNSetupOpcode::ASSIGN_ADDRESS, assignMsg, 1);
            agent.onSetupPacket(assignPkt);
            drain(transport);

            NNTopologyInfoMsg topoMsg{HW_ID, 0b11, 0, 2, static_cast<uint8_t>(NNActivationType::TANH), 2, 1, 0};
            NNPacket topoPkt{};
            packSetupMessage(topoPkt, NNSetupOpcode::TOPOLOGY_INFO, topoMsg, 2);
            agent.onSetupPacket(topoPkt);
            drain(transport);

            NNBackupRoleInfoMsg roleMsg{HW_ID, encodeAddress(backupTarget), 0b1, 0b1111,
                                         static_cast<uint8_t>(NNActivationType::SIGMOID), 1, 75, 0};
            NNPacket rolePkt{};
            packSetupMessage(rolePkt, NNSetupOpcode::BACKUP_ROLE_INFO, roleMsg, 3);
            agent.onSetupPacket(rolePkt);
            drain(transport);

            NNWeightsChunkMsg chunkMsg{};
            chunkMsg.hardwareId = HW_ID;
            chunkMsg.chunkIndex = 0;
            chunkMsg.chunkCount = 1;
            chunkMsg.valuesInChunk = 2;
            memcpy(chunkMsg.values, weights, sizeof(weights));
            NNPacket chunkPkt{};
            packSetupMessage(chunkPkt, NNSetupOpcode::WEIGHTS_CHUNK, chunkMsg, 4);
            agent.onSetupPacket(chunkPkt);
            drain(transport);

            NNWeightsChunkMsg backupChunkMsg{};
            backupChunkMsg.hardwareId = HW_ID;
            backupChunkMsg.chunkIndex = 0;
            backupChunkMsg.chunkCount = 1;
            backupChunkMsg.valuesInChunk = 1;
            memcpy(backupChunkMsg.values, backupWeights, sizeof(backupWeights));
            NNPacket backupChunkPkt{};
            packSetupMessage(backupChunkPkt, NNSetupOpcode::BACKUP_WEIGHTS_CHUNK, backupChunkMsg, 5);
            agent.onSetupPacket(backupChunkPkt);
            drain(transport);

            NNCommitRequestMsg commitReq{HW_ID};
            NNPacket commitPkt{};
            packSetupMessage(commitPkt, NNSetupOpcode::COMMIT_REQUEST, commitReq, 6);
            agent.onSetupPacket(commitPkt);
            drain(transport);
            assert(agent.getState() == NNSetupState::CONFIGURED);
        } // agent goes out of scope -- simulates power-off

        assert(store.hasSaved);

        // "Reboot": a brand new agent instance, same underlying store.
        NNSetupAgent rebooted(HW_ID, transport, store);
        rebooted.begin();
        assert(rebooted.getState() == NNSetupState::CONFIGURED); // skipped the whole handshake

        const NNNodeConfig& cfg = rebooted.getNodeConfig();
        assert(sameAddress(cfg.address, addr));
        assert(cfg.activationType == NNActivationType::TANH);
        assert(cfg.weightCount == 2);
        assert(cfg.predecessorLayerId == 0);
        for (int i = 0; i < 2; i++) assert(cfg.weights[i] == weights[i]);
        assert(cfg.hasBackupRole);
        assert(sameAddress(cfg.backupTargetAddress, backupTarget));
        assert(cfg.backupTargetActivationType == NNActivationType::SIGMOID);
        assert(cfg.resendGraceMs == 75);
        assert(cfg.backupWeightCount == 1);
        assert(cfg.backupTargetPredecessorLayerId == 0);
        assert(cfg.backupWeights[0] == backupWeights[0]);

        NNStartMsg startMsg{};
        NNPacket startPkt{};
        packSetupMessage(startPkt, NNSetupOpcode::START, startMsg, 7);
        rebooted.onSetupPacket(startPkt);
        assert(rebooted.isRunning());

        printf("TEST 6 PASSED: persisted config (including backup fields) survives a "
               "simulated reboot without repeating the handshake\n");
    }

    // --- TEST 7: a START with the wrong magic is ignored (regression check) ---
    {
        NNTransportLoopback transport;
        TestConfigStore store;
        store.hasSaved = true;
        store.saved.valid = true;
        store.saved.address = encodeAddress({0, 0, 0, 0});
        store.saved.weightCount = 0;

        NNSetupAgent agent(HW_ID, transport, store);
        agent.begin();
        assert(agent.getState() == NNSetupState::CONFIGURED);

        NNStartMsg badStart{};
        badStart.magic = 0x12345678;
        NNPacket badStartPkt{};
        packSetupMessage(badStartPkt, NNSetupOpcode::START, badStart, 1);
        agent.onSetupPacket(badStartPkt);

        assert(agent.getState() == NNSetupState::CONFIGURED); // unchanged
        printf("TEST 7 PASSED: a START packet with the wrong magic is ignored\n");
    }

    printf("\nALL SETUP PROTOCOL TESTS PASSED\n");
    return 0;
}
