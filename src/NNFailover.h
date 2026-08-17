// symmetric peer-to-peer backup failover, with a bounded resend-request 
// retry before falling back to backup weights, and teardown dispatch when
// even that fails.
//


#pragma once
#include <cassert>
#include "NNAddress.h"
#include "NNPacket.h"
#include "NNBuffer.h"
#include "NNActivation.h"
#include "NNNode.h"
 

constexpr uint8_t NN_FLAG_FAILOVER_SUBSTITUTE = 0x01;
// Set on a failover REPORT whose target could not be recovered at all 
constexpr uint8_t NN_FLAG_FAILOVER_TEARDOWN   = 0x02;

// Diagnostics multicast group: reported TO, never joined by any node.
constexpr uint8_t NN_DIAG_LAYER_GROUP = 254;

// Builds the packet that tells the laptop a failover happened, naming BOTH the
// node that failed and the node that stood in for it.
// The substitute packet itself cannot carry this. It deliberately acts as the target so it names the failed node but
// never the backup and appending the backup as a second payload float would corrupt the next layer's weighted sum
inline void buildFailoverReport(const NNAddress& backer, const NNAddress& failed,
                                uint8_t flags, NNPacket& outPkt) {
    outPkt = NNPacket{};
    outPkt.header.sourceAddress = encodeAddress(backer);   // who took over
    outPkt.header.targetLayerId = NN_DIAG_LAYER_GROUP;
    outPkt.header.type          = NNPacketType::CONTROL;
    outPkt.header.flags         = flags;
    outPkt.header.payloadCount  = 1;
    outPkt.payload[0] = static_cast<float>(encodeAddress(failed));  // who failed
}

class NNBackupStandby {
public:
    using ClockFn = unsigned long (*)();
 
 
    NNBackupStandby(const NNNodeConfig& ownerCfg, ClockFn clockFn)
        : cfg(ownerCfg), getTimeMs(clockFn) {
        if (cfg.hasBackupRole) {
            #ifndef NDEBUG
            assert(cfg.backupTargetAddress.layerId == cfg.address.layerId &&
                   "NNBackupStandby: backupTargetAddress must be in the SAME layer "
                   "as this node's own address -- backup is a per-layer, "
                   "sibling-to-sibling relationship, not cross-layer.");
            #endif
        }
    }
 
    bool isActive() const { return cfg.hasBackupRole; }
 
    // Called for EVERY packet observed on the medium.
    void onPacketObserved(const NNPacket& pkt) {
        if (!cfg.hasBackupRole) return;
 
        NNAddress src = decodeAddress(pkt.header.sourceAddress);
 
        if (isBackupTargetAddress(src)) {
            targetObserved = true;  
            return;
        }
 
        // Track every sibling's transmission in this layer, not just the ones this node's own predecessorMask cares about 
        if (src.layerId == cfg.address.layerId) {
            layerObservedMask |= (uint16_t(1) << src.nodeId);
        }
 
        
        if (src.layerId == cfg.backupTargetPredecessorLayerId &&
            (cfg.backupTargetPredecessorMask & (uint16_t(1) << src.nodeId))) {
            inputBuffer.storeInput(src.nodeId, pkt.payload, pkt.header.payloadCount);
        }
    }
 
  
    void tick() {
        if (!cfg.hasBackupRole) return;
        if (targetObserved || resolved) return;
 
        if (!resendRequested) {
            uint16_t targetBit = uint16_t(1) << cfg.backupTargetAddress.nodeId;
            uint16_t ownBit = uint16_t(1) << cfg.address.nodeId;
            uint16_t othersMask = cfg.layerRosterMask & ~targetBit & ~ownBit;
 
            if ((layerObservedMask & othersMask) != othersMask) {
                return;  // round not finished yet — some OTHER sibling hasn't gone either
            }
 
            // Everyone else in the layer has gone; the target specifically has not.
            queueResendRequest();
            resendRequested = true;
            waitStartTime = getTimeMs();  // the ONLY point this class reads the clock
            waitStarted = true;
            return;
        }
 
        // resendRequested == true: now in the post-resend grace period —
        // the one place this class is genuinely clock-based, per above.
        if (getTimeMs() - waitStartTime < cfg.resendGraceMs) return;
 
        if (inputBuffer.isComplete(cfg.backupTargetPredecessorMask)) {
            computeSubstituteOutput();
        } else {
            buildTeardownPacket();
        }
        resolved = true;
    }
 
    // Returns true and fills outPkt exactly once per queued output — first
    // the resend request (step 2), and LATER, in a subsequent tick() cycle,
    // the substitute (step 4) or teardown (step 5). Never more than one
    // pending at a time.
    bool hasOutputReady(NNPacket& outPkt) {
        if (!outputPending) return false;
        outPkt = outgoingPacket;
        outputPending = false;
        return true;
    }
 
    bool didRequestResend() const { return resendRequested; }
    bool didSubstitute() const { return hasSubstituted; }
    bool didTearDown() const { return hasTornDown; }

  
    bool hasDiagnosticReady(NNPacket& outPkt) {
        if (!diagnosticPending) return false;
        outPkt = diagnosticPacket;
        diagnosticPending = false;
        return true;
    }
 
    void resetForNextPass() {
        inputBuffer.reset();
        targetObserved = false;
        layerObservedMask = 0;
        resendRequested = false;
        hasSubstituted = false;
        hasTornDown = false;
        resolved = false;
        outputPending = false;
        diagnosticPending = false;
        waitStartTime = 0;
        waitStarted = false;
    }
 
private:
    bool isBackupTargetAddress(const NNAddress& addr) const {
        return addr.nodeId == cfg.backupTargetAddress.nodeId &&
               addr.layerId == cfg.backupTargetAddress.layerId &&
               addr.clusterId == cfg.backupTargetAddress.clusterId;
    }
 
  
    void queueDiagnostic(uint8_t flags) {
        buildFailoverReport(cfg.address, cfg.backupTargetAddress, flags, diagnosticPacket);
        diagnosticPending = true;
    }

    void queueResendRequest() {
        outgoingPacket = NNPacket{};
        outgoingPacket.header.sourceAddress = encodeAddress(cfg.address);  // who is asking
        outgoingPacket.header.type = NNPacketType::CONTROL;
        outgoingPacket.header.payloadCount = 1;
      
        outgoingPacket.payload[0] = static_cast<float>(encodeAddress(cfg.backupTargetAddress));
        outputPending = true;
    }
 
    void computeSubstituteOutput() {
 
        float sum = cfg.backupTargetBias;
        uint8_t weightIndex = 0;
        for (uint8_t senderId = 0; senderId < NN_MAX_PREDECESSORS; senderId++) {
            if (!(cfg.backupTargetPredecessorMask & (uint16_t(1) << senderId))) continue;
            uint8_t count;
            const float* vals = inputBuffer.getInput(senderId, count);
            for (uint8_t i = 0; i < count && weightIndex < cfg.backupWeightCount; i++, weightIndex++) {
                sum += vals[i] * cfg.backupWeights[weightIndex];
            }
        }
        float output = applyActivation(cfg.backupTargetActivationType, sum);
 
        outgoingPacket = NNPacket{};
        // IDENTITY SUBSTITUTION: the backup TARGET's address, not this node's own.
        outgoingPacket.header.sourceAddress = encodeAddress(cfg.backupTargetAddress);
        outgoingPacket.header.type = NNPacketType::DATA;
        outgoingPacket.header.flags = NN_FLAG_FAILOVER_SUBSTITUTE;
        outgoingPacket.header.payloadCount = 1;
        outgoingPacket.payload[0] = output;
 
        hasSubstituted = true;
        outputPending = true;
        queueDiagnostic(NN_FLAG_FAILOVER_SUBSTITUTE);
    }
 
    void buildTeardownPacket() {
        outgoingPacket = NNPacket{};
        outgoingPacket.header.sourceAddress = encodeAddress(cfg.address);  

        outgoingPacket.header.type = NNPacketType::TEARDOWN;
        outgoingPacket.header.payloadCount = 0;
        hasTornDown = true;
        outputPending = true;
        queueDiagnostic(NN_FLAG_FAILOVER_TEARDOWN);
    }
 
    const NNNodeConfig& cfg;
    ClockFn getTimeMs;
    NNInputBuffer inputBuffer;
    bool targetObserved = false;
    uint16_t layerObservedMask = 0;  // every sibling observed transmitting this pass
    bool resendRequested = false;
    bool hasSubstituted = false;
    bool hasTornDown = false;
    bool resolved = false;
    bool outputPending = false;
    bool diagnosticPending = false;
    unsigned long waitStartTime = 0;
    bool waitStarted = false;
    NNPacket outgoingPacket{};
    NNPacket diagnosticPacket{};
};
 
// The receiving  side of a resend request. A node that is alive but simply hasn't transmitted 
// yet uses this to recognize its own address in an incoming resend request and re-offer its already-computed output
class NNResendResponder {
public:
    explicit NNResendResponder(const NNAddress& ownAddress) : ownAddress(ownAddress) {}
 

    void onPacketObserved(const NNPacket& pkt, bool haveOutputThisPass, float outputValue) {
        if (pkt.header.type != NNPacketType::CONTROL) return;
        if (pkt.header.payloadCount < 1) return;
 
        uint16_t requestedAddr = static_cast<uint16_t>(pkt.payload[0]);
        NNAddress requested = decodeAddress(requestedAddr);
        if (requested.nodeId != ownAddress.nodeId ||
            requested.layerId != ownAddress.layerId ||
            requested.clusterId != ownAddress.clusterId) {
            return;  
        }
 
        if (!haveOutputThisPass) return;  
 
        pendingResendValue = outputValue;
        resendPending = true;
    }
 
    bool hasResendReady(NNPacket& outPkt) {
        if (!resendPending) return false;
        outPkt = NNPacket{};
        outPkt.header.sourceAddress = encodeAddress(ownAddress);
        outPkt.header.type = NNPacketType::DATA;
        outPkt.header.payloadCount = 1;
        outPkt.payload[0] = pendingResendValue;
        resendPending = false;
        return true;
    }
 
    void resetForNextPass() { resendPending = false; }
 
private:
    NNAddress ownAddress;
    bool resendPending = false;
    float pendingResendValue = 0.0f;
};
 

class NNDuplicateSuppressor {
public:
    // Returns false if this packet should be DROPPED 
    bool shouldAccept(const NNPacket& pkt) {
        NNAddress src = decodeAddress(pkt.header.sourceAddress);
        bool isSubstitute = (pkt.header.flags & NN_FLAG_FAILOVER_SUBSTITUTE) != 0;
 
        if (lockedMask & (uint16_t(1) << src.nodeId)) {
            return false;  // slot already claimed by a substitute this pass
        }
        if (isSubstitute) {
            lockedMask |= (uint16_t(1) << src.nodeId);  // lock it now
        }
        return true;
    }
 
    void reset() { lockedMask = 0; }
 
private:
    uint16_t lockedMask = 0;
};
 
// Desktop-only backup-config validation
struct NNBackupConfigValidationResult {
    static constexpr uint8_t MAX_ISSUES = 8;
 
    bool isValid = true;
    const char* issues[MAX_ISSUES] = {};
    uint8_t issueCount = 0;
 
    void addIssue(const char* msg) {
        isValid = false;
        if (issueCount < MAX_ISSUES) {
            issues[issueCount++] = msg;
        }
        
    }
};
 

inline NNBackupConfigValidationResult validateBackupConfig(
    const NNNodeConfig& backupCfg, const NNNodeConfig& targetRealCfg) {
 
    NNBackupConfigValidationResult result;
 
    if (!backupCfg.hasBackupRole) {
        result.addIssue("backupCfg.hasBackupRole is false -- nothing to validate");
        return result;  // every other field is meaningless if this is false
    }
 
    if (backupCfg.backupTargetAddress.nodeId != targetRealCfg.address.nodeId ||
        backupCfg.backupTargetAddress.layerId != targetRealCfg.address.layerId ||
        backupCfg.backupTargetAddress.clusterId != targetRealCfg.address.clusterId) {
        result.addIssue("backupTargetAddress does not match targetRealCfg.address");
    }
 
    if (backupCfg.backupTargetAddress.layerId != backupCfg.address.layerId) {
        result.addIssue("backupTargetAddress is not in the SAME layer as this node's own "
                         "address -- backup must be a per-layer, sibling-to-sibling relationship");
    }
 
    if (backupCfg.backupTargetActivationType != targetRealCfg.activationType) {
        result.addIssue("backupTargetActivationType does not match the target's REAL activationType");
    }
 
    if (backupCfg.backupTargetPredecessorMask != targetRealCfg.predecessorMask) {
        result.addIssue("backupTargetPredecessorMask does not match the target's REAL predecessorMask");
    }
 
    if (backupCfg.backupWeightCount != targetRealCfg.weightCount) {
        result.addIssue("backupWeightCount does not match the target's REAL weightCount");
    }
 
    // Bias must be mirrored too, for the same reason as every other
    // backupTarget* field above -- see NNNodeConfig::backupTargetBias.
    if (backupCfg.backupTargetBias != targetRealCfg.bias) {
        result.addIssue("backupTargetBias does not match the target's REAL bias");
    }
 
    return result;
}
