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
    const float* weights;            // borrowed, not owned — points at flash for a statically
                                       // configured node, or at RAM for a network-provisioned one
    const float* backupWeights;      // a COPY of backupTargetAddress's own weights (see below) —
                                       // meaningful only if hasBackupRole is true
    uint8_t  weightCount;

    float bias = 0.0f;              // Bias: added to the weighted sum before activation

    // Backup role: this node ALSO stands ready to compute on behalf of ONE other node, if that node fails to transmit in time. 
    bool      hasBackupRole = false;
    NNAddress backupTargetAddress{};                // the peer this node backs up
    uint16_t  backupTargetPredecessorMask = 0;       // That peer's own predecessor requirements
    NNActivationType backupTargetActivationType = NNActivationType::LINEAR;
    uint8_t   backupWeightCount = 0;                  // count for backupWeights above
 
    
    float     backupTargetBias = 0.0f;              // backupTargetBias: the peer's own bias value
 
    unsigned long resendGraceMs = 0;                  // how long to wait for a resend reply before falling back 
                                                    // to backup-weight substitution
    uint16_t  layerRosterMask = 0;                     // every node ID present in THIS node's own layer

    uint8_t   predecessorLayerId = 0;                  // which layer this node's OWN predecessors live in

    uint8_t   backupTargetPredecessorLayerId = 0;       
};
 
class NNNode {
public:
    explicit NNNode(const NNNodeConfig& cfg) : config(cfg), hasExecuted(false), outputValue(0.0f) {}
 
    // Called whenever a packet arrives. Filters by the sender's layer before matching by node ID
    void onPacketReceived(const NNPacket& pkt) {
        NNAddress src = decodeAddress(pkt.header.sourceAddress);
        if (src.layerId != config.predecessorLayerId) return;  // reject cross-layer collision
        inputBuffer.storeInput(src.nodeId, pkt.payload, pkt.header.payloadCount);
    }
 
    bool readyToExecute() const {
        return !hasExecuted && inputBuffer.isComplete(config.predecessorMask);
    }
 
    // Computes this node's output: a weighted sum of all received predecessor outputs
    // (starting from this node's own bias), passed through the selected activation function.
    void execute() {
        float sum = config.bias;
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
 
    // For nodes with predecessorMask == 0 (input-layer) sets outputValue directly 
    // and marks the node executed, bypassing the weighted-sum path entirely.
    void seedOutput(float value) {
        outputValue = value;
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
 
    // Must be called between inference passes to avoid data leakage 
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
 