#pragma once
#include"NNAddress.h"
#include"NNPacket.h"
#include"NNBuffer.h"
#include"NNActivation.h"

// Everything a single physical device needs to know about ITSELF and its IMMEDIATE neighbors. Deliberately contains NOTHING about the rest of the network.
struct NNNodeConfig {
    NNAddress address;               
    uint16_t predecessorMask;        // which node IDs in the previous layer feed this node
    uint8_t  successorLayerId;       // where this node's output is broadcast to (target layer)
    uint8_t  transmitSlot;           // this node's assigned time slot within its layer 
    NNActivationType activationType;
    const float* weights;            // pointer into flash — never copied into RAM
    const float* backupWeights;      // a COPY of backupTargetAddress's own weights (see below) —
                                       // meaningful only if hasBackupRole is true
    uint8_t  weightCount;

    // --- Backup role (NEW, appended fields): this node ALSO stands ready
    // to compute on behalf of ONE other node, if that node fails to
    // transmit in time. Leave hasBackupRole = false (the default) for a
    // node configured with no backup duty. ---
    bool      hasBackupRole = false;
    NNAddress backupTargetAddress{};                // the peer this node backs up
    uint16_t  backupTargetPredecessorMask = 0;       // THAT peer's own predecessor requirements
                                                       // (may differ from this node's own predecessorMask)
    NNActivationType backupTargetActivationType = NNActivationType::LINEAR;
    uint8_t   backupWeightCount = 0;                  // count for backupWeights above
    unsigned long resendGraceMs = 0;                  // the ONLY clock-based timing left: after
                                                        // failure is DETECTED (event-driven, see
                                                        // layerRosterMask below and NNFailover.h),
                                                        // how long to wait for a resend reply before
                                                        // falling back to backup-weight substitution
    uint16_t  layerRosterMask = 0;                     // every node ID present in THIS node's own
                                                        // layer (siblings + self) -- lets a backup
                                                        // detect "the round finished and my target
                                                        // never went" purely from observed traffic,
                                                        // with no clock needed for detection itself

    // --- Cross-layer collision fix (NEW): NNNode::onPacketReceived matches
    // predecessors by node ID ONLY -- on a single shared broadcast medium,
    // without this field, a node ID reused in a DIFFERENT layer could be
    // silently mistaken for a real predecessor. Explicit, not computed
    // from address.layerId - 1, for the same reason successorLayerId is
    // explicit rather than assumed to be address.layerId + 1: a real
    // per-device config should never rely on an unstated adjacency
    // assumption holding true elsewhere in the system. ---
    uint8_t   predecessorLayerId = 0;                  // which layer this node's OWN predecessors
                                                        // (config.predecessorMask) actually live in
    uint8_t   backupTargetPredecessorLayerId = 0;       // same idea, for the backup target's
                                                        // predecessors (backupTargetPredecessorMask) --
                                                        // meaningful only if hasBackupRole is true
};

class NNNode {
public:
    explicit NNNode(const NNNodeConfig& cfg) : config(cfg), hasExecuted(false), outputValue(0.0f) {}

    // Called whenever a packet arrives.
    // FIX: now filters by the sender's layer BEFORE matching by node ID.
    // Previously matched by node ID alone -- on a single shared broadcast
    // medium, a node ID reused in a different layer (e.g. a Layer-0 input
    // node sharing an ID with a Layer-1 node) could be silently mistaken
    // for a real predecessor, corrupting the weighted sum with the wrong
    // sender's value. See NNNodeConfig::predecessorLayerId above.
    void onPacketReceived(const NNPacket& pkt) {
        NNAddress src = decodeAddress(pkt.header.sourceAddress);
        if (src.layerId != config.predecessorLayerId) return;  // NEW: reject cross-layer collision
        inputBuffer.storeInput(src.nodeId, pkt.payload, pkt.header.payloadCount);
    }

    bool readyToExecute() const {
        return !hasExecuted && inputBuffer.isComplete(config.predecessorMask);
    }

    // Computes this node's output: a weighted sum of all received predecessor outputs, passed through the selected activation function
    void execute() {
        float sum = 0.0f;
        uint8_t weightIndex = 0;
        for (uint8_t senderId = 0; senderId < NN_MAX_PREDECESSORS; senderId++) {
            if (!(config.predecessorMask & (uint16_t(1) << senderId))) continue;
            uint8_t count;
            const float* vals = inputBuffer.getInput(senderId, count);
            for (uint8_t i = 0; i < count && weightIndex < config.weightCount; i++, weightIndex++) {
                sum += vals[i] * config.weights[weightIndex];
            }
        }
        outputValue = applyActivation(config.activationType, sum);
        hasExecuted = true;
    }

    // Packages this node's computed output into an NNPacket.
    NNPacket buildOutputPacket() const {
        NNPacket pkt{};
        pkt.header.sourceAddress  = encodeAddress(config.address);
        pkt.header.targetLayerId  = config.successorLayerId;
        pkt.header.type           = NNPacketType::DATA;
        pkt.header.payloadCount   = 1;
        pkt.payload[0] = outputValue;
        return pkt;
    }

    // Must be called between inference passes, alongside inputBuffer.reset(), to avoid data leakage 
    void resetForNextPass() {
        inputBuffer.reset();
        hasExecuted = false;
        outputValue = 0.0f;
    }

private:
    NNNodeConfig config;
    NNInputBuffer inputBuffer;
    bool hasExecuted;
    float outputValue;
};