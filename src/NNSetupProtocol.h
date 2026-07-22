// ============================================================================
// NNSetupProtocol.h
//
// Setup-phase protocol: a laptop discovers unconfigured devices over the
// same wireless link they'll use at runtime, assigns each one an NNAddress
// + NNNodeConfig, ships its weights over in chunks, optionally assigns a
// backup duty (NNNodeConfig::hasBackupRole, see NNFailover.h) and ships its
// backup weights too, verifies the result, and finally tells everyone to
// go live.
//
// Carried as NNPacketType::CONTROL packets. The opcode lives in
// NNPacketHeader.flags (previously reserved/unused). Every setup message
// is laid out as a whole number of 4-byte words so it can ride through the
// EXISTING serializePacket()/deserializePacket() functions unmodified --
// those only understand "N floats", they don't know or care that the bytes
// aren't really floats.
//
// TRANSPORT SIZE BUDGET: NNTransportUDP / NNTransportUDPMulticast currently
// use 64-byte wire buffers -> 8 header bytes + 56 payload bytes -> 14 floats
// max per packet. Every message below is sized to fit that ceiling exactly,
// so this protocol works whether or not that transport buffer size ever
// gets bumped to match NN_MAX_PAYLOAD_FLOATS (16).
//
// PATCHED: NNTopologyInfoMsg gained a `bias` field and NNBackupRoleInfoMsg
// gained a `backupTargetBias` field, mirroring the same additions to
// NNNodeConfig in NNNode.h. Both messages are still a whole number of
// 4-byte words (24 and 28 bytes respectively) and stay well under the
// 56-byte ceiling above.
//
// ORDERING ASSUMPTIONS: a device expects ASSIGN_ADDRESS before TOPOLOGY_INFO
// (topology setup derives windowConfig.ownLayerId from the just-assigned
// address), and TOPOLOGY_INFO before BACKUP_ROLE_INFO/BACKUP_WEIGHTS_CHUNK
// (those are only accepted once the device has left UNCONFIGURED). The
// laptop-side tool must send them in that order per device. BACKUP_ROLE_INFO
// and the WEIGHTS_CHUNK/BACKUP_WEIGHTS_CHUNK streams are otherwise
// order-tolerant with respect to each other -- a device that finishes its
// primary weights before BACKUP_ROLE_INFO arrives simply waits (see
// NNSetupAgent::maybeAdvanceToVerifying) rather than reporting success early.
// ============================================================================
#pragma once
#include <stdint.h>
#include <cstring>
#include "NNPacket.h"
#include "NNAddress.h"
#include "NNActivation.h"
#include "NNTransport.h"
#include "NNScheduler.h"   // for NNWindowConfig
#include "NNNode.h"        // for NNNodeConfig

// ----------------------------------------------------------------------
// Opcodes -- stored in NNPacketHeader.flags when header.type == CONTROL
// ----------------------------------------------------------------------
enum class NNSetupOpcode : uint8_t {
    HELLO          = 0x01, // device -> laptop: "I exist, unconfigured", carries hardwareId
    ASSIGN_ADDRESS = 0x02, // laptop -> device: assign NNAddress
    TOPOLOGY_INFO  = 0x03, // laptop -> device: predecessorMask / successorLayerId / activation / etc
    WEIGHTS_CHUNK  = 0x04, // laptop -> device: partial weight array
    COMMIT_REQUEST = 0x05, // laptop -> device: "verify what you've assembled and report back"
    COMMIT_REPLY   = 0x06, // device -> laptop: checksum + success/failure of assembled config
    ACK            = 0x07, // device -> laptop: generic ack, references header.sequenceNumber
    NACK           = 0x08, // device -> laptop: reject (bad chunk index, unknown hardwareId, etc.)
    START          = 0x09, // laptop -> all: leave setup mode, begin runtime operation

    // Backup-role provisioning -- OPTIONAL per device. A device that never
    // receives these simply keeps NNNodeConfig::hasBackupRole == false (its
    // default). See NNFailover.h / NNBackupStandby for what these fields feed.
    BACKUP_ROLE_INFO     = 0x0A, // laptop -> device: assign backup duty + its scalar fields
    BACKUP_WEIGHTS_CHUNK = 0x0B, // laptop -> device: partial backupWeights array (reuses NNWeightsChunkMsg's layout -- only the opcode routes it to a different destination array)
};

// 12 bytes of chunk header (hardwareId + 3 index bytes + 1 pad) + 11 floats
// (44 bytes) = 56 bytes payload = exactly the 14-float wire ceiling.
constexpr uint8_t NN_WEIGHTS_CHUNK_MAX_FLOATS = 11;
constexpr uint8_t NN_SETUP_MAX_WEIGHTS        = 64; // device-side storage cap; raise if a node needs more
constexpr uint8_t NN_SETUP_MAX_BACKUP_WEIGHTS = 64; // device-side storage cap for backupWeights

constexpr uint8_t NN_SETUP_NACK_BAD_CHUNK_INDEX = 1; // chunkIndex >= this device's own expected chunk count
constexpr uint8_t NN_SETUP_NACK_WRONG_STATE     = 2; // reserved -- not currently emitted (see NNNackMsg below)

// Every struct below is exactly a whole number of 4-byte words, so it maps
// directly onto NNPacket.payload (float[16]) via memcpy, with payloadCount
// set to sizeof(msg)/4. #pragma pack(1) keeps the C++ layout free of
// compiler-inserted padding so it matches a hand-written Python struct.pack
// format on the laptop side byte-for-byte.
#pragma pack(push, 1)

struct NNHelloMsg {
    uint64_t hardwareId;
    uint32_t _reserved = 0;
};

struct NNAssignAddressMsg {
    uint64_t hardwareId;  // which physical device this is for
    uint16_t address;     // encodeAddress() output
    uint16_t _reserved = 0;
};

struct NNTopologyInfoMsg {
    uint64_t hardwareId;
    uint16_t predecessorMask;
    uint16_t precedingSiblingsMask; // feeds NNScheduler::NNWindowConfig
    uint8_t  successorLayerId;
    uint8_t  activationType;        // NNActivationType
    uint8_t  weightCount;
    uint8_t  transmitSlot;
    uint8_t  predecessorLayerId;    // which layer predecessorMask's node IDs actually live in
                                     // (NNNodeConfig's cross-layer-collision-fix field)
    float    bias;                  // NEW -- mirrors NNNodeConfig::bias (NNNode.h)
    uint8_t  _reserved[3] = {0, 0, 0}; // pads to a whole number of 4-byte words (24 bytes total)
};

// Also used, unmodified, for BACKUP_WEIGHTS_CHUNK -- the opcode on the
// enclosing packet is what routes a given chunk into weightStorage vs.
// backupWeightStorage on the receiving end (see NNSetupAgent below).
struct NNWeightsChunkMsg {
    uint64_t hardwareId;
    uint8_t  chunkIndex;
    uint8_t  chunkCount;
    uint8_t  valuesInChunk;
    uint8_t  _reserved = 0;
    float    values[NN_WEIGHTS_CHUNK_MAX_FLOATS];
};

// OPTIONAL -- only sent to a device that should also stand in as a backup
// for one sibling. See NNNodeConfig (NNNode.h) / NNFailover.h for how these
// fields are actually used once runtime starts.
struct NNBackupRoleInfoMsg {
    uint64_t hardwareId;
    uint16_t backupTargetAddress;         // encodeAddress() of the peer this node backs up
    uint16_t backupTargetPredecessorMask; // THAT peer's own predecessor requirements
    uint16_t layerRosterMask;             // every node ID present in this node's own layer
    uint8_t  backupTargetActivationType;  // NNActivationType
    uint8_t  backupWeightCount;
    uint32_t resendGraceMs;               // fixed-width on the wire, regardless of local `unsigned long` width
    uint8_t  backupTargetPredecessorLayerId; // which layer backupTargetPredecessorMask's node IDs live in
    float    backupTargetBias;            // NEW -- mirrors NNNodeConfig::backupTargetBias (NNNode.h)
    uint8_t  _reserved[3] = {0, 0, 0};       // pads to a whole number of 4-byte words (28 bytes total)
};

struct NNCommitRequestMsg {
    uint64_t hardwareId;
};

struct NNCommitReplyMsg {
    uint64_t hardwareId;
    uint8_t  computedChecksum; // primary weights; 0 if !success
    uint8_t  backupChecksum;   // backup weights; 0 if no backup role or !success
    uint8_t  success;          // 1 = topology + primary weights verified, AND backup weights too if hasBackupRole
    uint8_t  hasBackupRole;    // echoes the device's own state, so the laptop can sanity-check against its manifest
};

struct NNAckMsg {
    uint64_t hardwareId;
    uint8_t  ackedSequenceNumber;
    uint8_t  ackedOpcode;
    uint16_t _reserved = 0;
};

struct NNNackMsg {
    uint64_t hardwareId;
    uint8_t  nackedSequenceNumber;
    uint8_t  nackedOpcode;
    uint8_t  reasonCode;   // NN_SETUP_NACK_* above
    uint8_t  _reserved = 0;
};

struct NNStartMsg {
    uint32_t magic = 0x4E4E5354; // 'NNST' -- sanity check against stray packets
};

#pragma pack(pop)

// Pack/unpack helpers -----------------------------------------------------

template <typename MsgT>
inline void packSetupMessage(NNPacket& pkt, NNSetupOpcode op, const MsgT& msg, uint8_t sequenceNumber) {
    static_assert(sizeof(MsgT) % 4 == 0, "setup messages must be a whole number of 4-byte words");
    static_assert(sizeof(MsgT) <= sizeof(pkt.payload), "setup message too large for one packet");
    pkt = NNPacket{};
    pkt.header.type           = NNPacketType::CONTROL;
    pkt.header.flags          = static_cast<uint8_t>(op);
    pkt.header.sequenceNumber = sequenceNumber;
    pkt.header.payloadCount   = sizeof(MsgT) / 4;
    memcpy(pkt.payload, &msg, sizeof(MsgT));
}

template <typename MsgT>
inline MsgT unpackSetupMessage(const NNPacket& pkt) {
    MsgT msg{};
    constexpr size_t n = sizeof(MsgT);
    memcpy(&msg, pkt.payload, n <= sizeof(pkt.payload) ? n : sizeof(pkt.payload));
    return msg;
}

inline NNSetupOpcode getSetupOpcode(const NNPacket& pkt) {
    return static_cast<NNSetupOpcode>(pkt.header.flags);
}

// ----------------------------------------------------------------------
// Persistence -- implement per-platform (NVS/Preferences on ESP32, EEPROM
// elsewhere). NNVolatileConfigStore below is a no-op default so everything
// compiles and runs (without surviving a reboot) before flash storage is
// wired up.
// ----------------------------------------------------------------------
struct NNPersistedConfig {
    bool     valid = false;
    uint16_t address;
    uint16_t predecessorMask;
    uint16_t precedingSiblingsMask;
    uint8_t  successorLayerId;
    uint8_t  activationType;
    uint8_t  transmitSlot;
    uint8_t  weightCount;
    float    weights[NN_SETUP_MAX_WEIGHTS];
    uint8_t  predecessorLayerId = 0; // which layer predecessorMask's node IDs live in (cross-layer-collision fix)
    float    bias = 0.0f;            // NEW -- mirrors NNNodeConfig::bias (NNNode.h)

    // Backup role -- mirrors NNNodeConfig's own appended backup fields
    // (NNNode.h). hasBackupRole == false is the common case: everything
    // below is then meaningless and never read back out.
    bool     hasBackupRole = false;
    uint16_t backupTargetAddress = 0;          // encoded
    uint16_t backupTargetPredecessorMask = 0;
    uint16_t layerRosterMask = 0;
    uint8_t  backupTargetActivationType = 0;
    uint8_t  backupWeightCount = 0;
    uint32_t resendGraceMs = 0;
    uint8_t  backupTargetPredecessorLayerId = 0; // which layer backupTargetPredecessorMask's node IDs live in
    float    backupTargetBias = 0.0f;            // NEW -- mirrors NNNodeConfig::backupTargetBias (NNNode.h)
    float    backupWeights[NN_SETUP_MAX_BACKUP_WEIGHTS];
};

class NNConfigStore {
public:
    virtual ~NNConfigStore() {}
    virtual bool load(NNPersistedConfig& out) = 0;  // true if a valid config was found
    virtual bool save(const NNPersistedConfig& cfg) = 0;
    virtual void clear() = 0;
};

class NNVolatileConfigStore : public NNConfigStore {
public:
    bool load(NNPersistedConfig& out) override { (void)out; return false; }
    bool save(const NNPersistedConfig& cfg) override { (void)cfg; return true; }
    void clear() override {}
};

// ----------------------------------------------------------------------
// Device-side setup state machine
// ----------------------------------------------------------------------
enum class NNSetupState : uint8_t {
    LOADING,           // checking persisted config at boot
    UNCONFIGURED,       // announcing HELLO, waiting for assignment
    RECEIVING_CONFIG,   // got address + topology, collecting weight chunks
    VERIFYING,          // all chunks in, waiting on a COMMIT_REQUEST round-trip
    CONFIGURED,          // verified and (if a real store is wired up) persisted; waiting for START
    RUNNING,            // setup complete -- caller should switch to normal NNNode/NNScheduler operation
};

class NNSetupAgent {
public:
    NNSetupAgent(uint64_t hardwareId, NNTransport& transportRef, NNConfigStore& storeRef)
        : hwId(hardwareId), transport(transportRef), store(storeRef),
          state(NNSetupState::LOADING), sequenceCounter(0),
          weightCount(0), receivedChunkMask(0), expectedChunkCount(0),
          receivedBackupChunkMask(0), expectedBackupChunkCount(0),
          lastAnnounceMs(0), announceIntervalMs(1000) {}

    // Call once at boot, before the main loop.
    void begin() {
        NNPersistedConfig saved{};
        if (store.load(saved) && saved.valid) {
            applyPersisted(saved);
            state = NNSetupState::CONFIGURED; // still waits for a fresh START after boot
        } else {
            state = NNSetupState::UNCONFIGURED;
        }
    }

    NNSetupState getState() const { return state; }
    bool isRunning() const { return state == NNSetupState::RUNNING; }

    // Valid from CONFIGURED onward -- hand these to NNNode / NNScheduler.
    const NNNodeConfig& getNodeConfig() const { return nodeConfig; }
    const NNWindowConfig& getWindowConfig() const { return windowConfig; }

    // Call for every received packet with header.type == CONTROL.
    void onSetupPacket(const NNPacket& pkt) {
        switch (getSetupOpcode(pkt)) {
            case NNSetupOpcode::ASSIGN_ADDRESS:       handleAssignAddress(pkt); break;
            case NNSetupOpcode::TOPOLOGY_INFO:        handleTopologyInfo(pkt); break;
            case NNSetupOpcode::WEIGHTS_CHUNK:        handleWeightsChunk(pkt); break;
            case NNSetupOpcode::BACKUP_ROLE_INFO:      handleBackupRoleInfo(pkt); break;
            case NNSetupOpcode::BACKUP_WEIGHTS_CHUNK:  handleBackupWeightsChunk(pkt); break;
            case NNSetupOpcode::COMMIT_REQUEST:        handleCommitRequest(pkt); break;
            case NNSetupOpcode::START:                 handleStart(pkt); break;
            default: break; // HELLO / ACK / NACK / COMMIT_REPLY are laptop-bound, not device-bound
        }
    }

    // Call once per loop iteration while state == UNCONFIGURED.
    void tick(uint32_t nowMs) {
        if (state != NNSetupState::UNCONFIGURED) return;
        if (nowMs - lastAnnounceMs >= announceIntervalMs) {
            sendHello();
            lastAnnounceMs = nowMs;
            // Jitter so dozens of devices booting together don't announce in lockstep.
            announceIntervalMs = 700 + static_cast<uint32_t>(hwId % 600);
        }
    }

private:
    uint64_t hwId;
    NNTransport& transport;
    NNConfigStore& store;
    NNSetupState state;
    uint8_t sequenceCounter;

    NNNodeConfig nodeConfig{};
    NNWindowConfig windowConfig{};
    float weightStorage[NN_SETUP_MAX_WEIGHTS];
    uint8_t weightCount;
    uint64_t receivedChunkMask;   // bit per chunk index received (supports up to 64 chunks)
    uint8_t expectedChunkCount;

    // Backup role -- unused (stays all-zero/false) for a device with no
    // backup duty. nodeConfig.weights/backupWeights below both point into
    // THIS object's own RAM arrays, not flash -- see the note on
    // NNNodeConfig::weights in NNNode.h, which predates network-provisioned
    // devices and doesn't reflect this code path.
    float backupWeightStorage[NN_SETUP_MAX_BACKUP_WEIGHTS];
    uint64_t receivedBackupChunkMask;
    uint8_t expectedBackupChunkCount;

    uint32_t lastAnnounceMs;
    uint32_t announceIntervalMs;

    void sendHello() {
        NNPacket pkt{};
        NNHelloMsg msg{hwId};
        packSetupMessage(pkt, NNSetupOpcode::HELLO, msg, sequenceCounter++);
        transport.send(pkt);
    }

    void sendAck(NNSetupOpcode forOpcode, uint8_t seq) {
        NNPacket pkt{};
        NNAckMsg msg{hwId, seq, static_cast<uint8_t>(forOpcode), 0};
        packSetupMessage(pkt, NNSetupOpcode::ACK, msg, sequenceCounter++);
        transport.send(pkt);
    }

    void sendNack(NNSetupOpcode forOpcode, uint8_t seq, uint8_t reasonCode) {
        NNPacket pkt{};
        NNNackMsg msg{hwId, seq, static_cast<uint8_t>(forOpcode), reasonCode, 0};
        packSetupMessage(pkt, NNSetupOpcode::NACK, msg, sequenceCounter++);
        transport.send(pkt);
    }

    // A device with no backup role never has anything to wait for here --
    // vacuously complete so it never blocks reaching VERIFYING.
    bool primaryComplete() const {
        uint64_t expectedMask = expectedChunkCount >= 64
            ? ~uint64_t(0)
            : ((uint64_t(1) << expectedChunkCount) - 1);
        return (receivedChunkMask & expectedMask) == expectedMask;
    }

    bool backupComplete() const {
        if (!nodeConfig.hasBackupRole) return true;
        uint64_t expectedMask = expectedBackupChunkCount >= 64
            ? ~uint64_t(0)
            : ((uint64_t(1) << expectedBackupChunkCount) - 1);
        return (receivedBackupChunkMask & expectedMask) == expectedMask;
    }

    // Called after every chunk-affecting handler (primary or backup). Also
    // handles the case where BACKUP_ROLE_INFO shows up AFTER primary weights
    // already finished (state already VERIFYING) -- backupComplete() will be
    // false until backup chunks land, so this simply won't re-promote until
    // then; see handleBackupRoleInfo's demotion back to RECEIVING_CONFIG.
    void maybeAdvanceToVerifying() {
        if (state != NNSetupState::RECEIVING_CONFIG) return;
        if (!primaryComplete() || !backupComplete()) return;
        nodeConfig.weights = weightStorage;
        if (nodeConfig.hasBackupRole) nodeConfig.backupWeights = backupWeightStorage;
        state = NNSetupState::VERIFYING;
    }

    void handleAssignAddress(const NNPacket& pkt) {
        auto msg = unpackSetupMessage<NNAssignAddressMsg>(pkt);
        if (msg.hardwareId != hwId) return; // not addressed to us
        nodeConfig.address = decodeAddress(msg.address);
        sendAck(NNSetupOpcode::ASSIGN_ADDRESS, pkt.header.sequenceNumber);
    }

    void handleTopologyInfo(const NNPacket& pkt) {
        auto msg = unpackSetupMessage<NNTopologyInfoMsg>(pkt);
        if (msg.hardwareId != hwId) return;

        nodeConfig.predecessorMask   = msg.predecessorMask;
        nodeConfig.successorLayerId  = msg.successorLayerId;
        nodeConfig.transmitSlot      = msg.transmitSlot;
        nodeConfig.activationType    = static_cast<NNActivationType>(msg.activationType);
        nodeConfig.weightCount       = msg.weightCount;
        nodeConfig.predecessorLayerId = msg.predecessorLayerId;
        nodeConfig.bias               = msg.bias;   // NEW

        windowConfig.ownLayerId            = nodeConfig.address.layerId; // requires ASSIGN_ADDRESS already handled
        windowConfig.precedingSiblingsMask = msg.precedingSiblingsMask;

        expectedChunkCount = (msg.weightCount + NN_WEIGHTS_CHUNK_MAX_FLOATS - 1) / NN_WEIGHTS_CHUNK_MAX_FLOATS;
        receivedChunkMask = 0;
        weightCount = 0;
        state = NNSetupState::RECEIVING_CONFIG;
        sendAck(NNSetupOpcode::TOPOLOGY_INFO, pkt.header.sequenceNumber);
    }

    void handleWeightsChunk(const NNPacket& pkt) {
        auto msg = unpackSetupMessage<NNWeightsChunkMsg>(pkt);
        if (msg.hardwareId != hwId) return;
        if (state != NNSetupState::RECEIVING_CONFIG && state != NNSetupState::VERIFYING) return;
        if (msg.chunkIndex >= expectedChunkCount) {
            sendNack(NNSetupOpcode::WEIGHTS_CHUNK, pkt.header.sequenceNumber, NN_SETUP_NACK_BAD_CHUNK_INDEX);
            return;
        }

        uint16_t offset = static_cast<uint16_t>(msg.chunkIndex) * NN_WEIGHTS_CHUNK_MAX_FLOATS;
        for (uint8_t i = 0; i < msg.valuesInChunk && (offset + i) < NN_SETUP_MAX_WEIGHTS; i++) {
            weightStorage[offset + i] = msg.values[i];
        }
        receivedChunkMask |= (uint64_t(1) << msg.chunkIndex);
        sendAck(NNSetupOpcode::WEIGHTS_CHUNK, pkt.header.sequenceNumber);
        maybeAdvanceToVerifying();
    }

    // OPTIONAL -- only arrives for a device the laptop tool has assigned a
    // backup duty to. If this shows up after primary weights already
    // completed (state already VERIFYING), demote back to RECEIVING_CONFIG
    // until the backup weight chunks land too -- a device must never report
    // COMMIT success while hasBackupRole is true but backupWeights is
    // incomplete (NNBackupStandby would then hold a stale/incomplete array).
    void handleBackupRoleInfo(const NNPacket& pkt) {
        auto msg = unpackSetupMessage<NNBackupRoleInfoMsg>(pkt);
        if (msg.hardwareId != hwId) return;
        if (state != NNSetupState::RECEIVING_CONFIG && state != NNSetupState::VERIFYING) return;

        nodeConfig.hasBackupRole                       = true;
        nodeConfig.backupTargetAddress                 = decodeAddress(msg.backupTargetAddress);
        nodeConfig.backupTargetPredecessorMask         = msg.backupTargetPredecessorMask;
        nodeConfig.backupTargetActivationType           = static_cast<NNActivationType>(msg.backupTargetActivationType);
        nodeConfig.backupWeightCount                    = msg.backupWeightCount;
        nodeConfig.resendGraceMs                        = static_cast<unsigned long>(msg.resendGraceMs);
        nodeConfig.layerRosterMask                      = msg.layerRosterMask;
        nodeConfig.backupTargetPredecessorLayerId       = msg.backupTargetPredecessorLayerId;
        nodeConfig.backupTargetBias                     = msg.backupTargetBias;   // NEW

        expectedBackupChunkCount = (msg.backupWeightCount + NN_WEIGHTS_CHUNK_MAX_FLOATS - 1) / NN_WEIGHTS_CHUNK_MAX_FLOATS;
        receivedBackupChunkMask = 0;

        state = NNSetupState::RECEIVING_CONFIG; // demote if we'd already reached VERIFYING
        sendAck(NNSetupOpcode::BACKUP_ROLE_INFO, pkt.header.sequenceNumber);
        maybeAdvanceToVerifying();
    }

    void handleBackupWeightsChunk(const NNPacket& pkt) {
        auto msg = unpackSetupMessage<NNWeightsChunkMsg>(pkt);
        if (msg.hardwareId != hwId) return;
        if (!nodeConfig.hasBackupRole) return; // stray traffic for a device with no backup duty
        if (state != NNSetupState::RECEIVING_CONFIG && state != NNSetupState::VERIFYING) return;
        if (msg.chunkIndex >= expectedBackupChunkCount) {
            sendNack(NNSetupOpcode::BACKUP_WEIGHTS_CHUNK, pkt.header.sequenceNumber, NN_SETUP_NACK_BAD_CHUNK_INDEX);
            return;
        }

        uint16_t offset = static_cast<uint16_t>(msg.chunkIndex) * NN_WEIGHTS_CHUNK_MAX_FLOATS;
        for (uint8_t i = 0; i < msg.valuesInChunk && (offset + i) < NN_SETUP_MAX_BACKUP_WEIGHTS; i++) {
            backupWeightStorage[offset + i] = msg.values[i];
        }
        receivedBackupChunkMask |= (uint64_t(1) << msg.chunkIndex);
        sendAck(NNSetupOpcode::BACKUP_WEIGHTS_CHUNK, pkt.header.sequenceNumber);
        maybeAdvanceToVerifying();
    }

    void handleCommitRequest(const NNPacket& pkt) {
        auto msg = unpackSetupMessage<NNCommitRequestMsg>(pkt);
        if (msg.hardwareId != hwId) return;

        // state == VERIFYING already implies primaryComplete() && backupComplete()
        // (see maybeAdvanceToVerifying) -- no separate backup check needed here.
        bool ok = (state == NNSetupState::VERIFYING);
        uint8_t checksum = ok
            ? computeChecksum(reinterpret_cast<const uint8_t*>(weightStorage), nodeConfig.weightCount * sizeof(float))
            : 0;
        uint8_t backupChecksum = (ok && nodeConfig.hasBackupRole)
            ? computeChecksum(reinterpret_cast<const uint8_t*>(backupWeightStorage), nodeConfig.backupWeightCount * sizeof(float))
            : 0;

        if (ok) {
            NNPersistedConfig toSave{};
            toSave.valid                  = true;
            toSave.address                = encodeAddress(nodeConfig.address);
            toSave.predecessorMask        = nodeConfig.predecessorMask;
            toSave.precedingSiblingsMask  = windowConfig.precedingSiblingsMask;
            toSave.successorLayerId       = nodeConfig.successorLayerId;
            toSave.activationType         = static_cast<uint8_t>(nodeConfig.activationType);
            toSave.transmitSlot           = nodeConfig.transmitSlot;
            toSave.weightCount            = nodeConfig.weightCount;
            memcpy(toSave.weights, weightStorage, sizeof(float) * nodeConfig.weightCount);
            toSave.predecessorLayerId     = nodeConfig.predecessorLayerId;
            toSave.bias                   = nodeConfig.bias;   // NEW

            toSave.hasBackupRole = nodeConfig.hasBackupRole;
            if (nodeConfig.hasBackupRole) {
                toSave.backupTargetAddress         = encodeAddress(nodeConfig.backupTargetAddress);
                toSave.backupTargetPredecessorMask = nodeConfig.backupTargetPredecessorMask;
                toSave.layerRosterMask             = nodeConfig.layerRosterMask;
                toSave.backupTargetActivationType  = static_cast<uint8_t>(nodeConfig.backupTargetActivationType);
                toSave.backupWeightCount           = nodeConfig.backupWeightCount;
                toSave.resendGraceMs               = static_cast<uint32_t>(nodeConfig.resendGraceMs);
                toSave.backupTargetPredecessorLayerId = nodeConfig.backupTargetPredecessorLayerId;
                toSave.backupTargetBias            = nodeConfig.backupTargetBias;   // NEW
                memcpy(toSave.backupWeights, backupWeightStorage, sizeof(float) * nodeConfig.backupWeightCount);
            }

            store.save(toSave);
            state = NNSetupState::CONFIGURED;
        }

        NNPacket reply{};
        NNCommitReplyMsg replyMsg{hwId, checksum, backupChecksum, static_cast<uint8_t>(ok ? 1 : 0),
                                   static_cast<uint8_t>(nodeConfig.hasBackupRole ? 1 : 0)};
        packSetupMessage(reply, NNSetupOpcode::COMMIT_REPLY, replyMsg, sequenceCounter++);
        transport.send(reply);
    }

    void handleStart(const NNPacket& pkt) {
        auto msg = unpackSetupMessage<NNStartMsg>(pkt);
        if (msg.magic != 0x4E4E5354) return;
        if (state == NNSetupState::CONFIGURED) {
            state = NNSetupState::RUNNING;
        }
    }

    void applyPersisted(const NNPersistedConfig& saved) {
        nodeConfig.address          = decodeAddress(saved.address);
        nodeConfig.predecessorMask  = saved.predecessorMask;
        nodeConfig.successorLayerId = saved.successorLayerId;
        nodeConfig.activationType   = static_cast<NNActivationType>(saved.activationType);
        nodeConfig.transmitSlot     = saved.transmitSlot;
        nodeConfig.weightCount      = saved.weightCount;
        memcpy(weightStorage, saved.weights, sizeof(float) * saved.weightCount);
        nodeConfig.weights = weightStorage;
        nodeConfig.predecessorLayerId = saved.predecessorLayerId;
        nodeConfig.bias                = saved.bias;   // NEW

        windowConfig.ownLayerId            = nodeConfig.address.layerId;
        windowConfig.precedingSiblingsMask = saved.precedingSiblingsMask;

        nodeConfig.hasBackupRole = saved.hasBackupRole;
        if (saved.hasBackupRole) {
            nodeConfig.backupTargetAddress         = decodeAddress(saved.backupTargetAddress);
            nodeConfig.backupTargetPredecessorMask = saved.backupTargetPredecessorMask;
            nodeConfig.layerRosterMask             = saved.layerRosterMask;
            nodeConfig.backupTargetActivationType  = static_cast<NNActivationType>(saved.backupTargetActivationType);
            nodeConfig.backupWeightCount           = saved.backupWeightCount;
            nodeConfig.resendGraceMs               = static_cast<unsigned long>(saved.resendGraceMs);
            nodeConfig.backupTargetPredecessorLayerId = saved.backupTargetPredecessorLayerId;
            nodeConfig.backupTargetBias            = saved.backupTargetBias;   // NEW
            memcpy(backupWeightStorage, saved.backupWeights, sizeof(float) * saved.backupWeightCount);
            nodeConfig.backupWeights = backupWeightStorage;
        }
    }
};
