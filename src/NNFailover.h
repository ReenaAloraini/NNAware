// symmetric peer-to-peer backup failover, with a bounded resend-request 
// retry before falling back to backup weights, and teardown dispatch when
// even that fails.
//
// DESIGN: there is no separate "backup node" role. EVERY node's own
// NNNodeConfig (NNNode.h) optionally carries a designated backup duty for
// exactly ONE peer (hasBackupRole/backupTarget*/backupWeights fields).
// This distributes fault-tolerance responsibility across every node in a
// layer instead of concentrating it in one place — a dedicated backup
// node would itself be a new single point of failure — and each node
// stores only one extra weight set (for the specific peer it backs up),
// not the whole layer's weight matrix.
//

//
// NNResendResponder is the RECEIVING side of step 2/3: a node that is
// alive but simply hasn't transmitted yet recognizes its own address in an
// incoming resend request and re-offers its already-computed output for
// retransmission, if it has one. Without this, a resend request is a
// message nobody answers — this is what makes retransmission (as opposed
// to only backup-weight substitution) an actual recovery path for
// transient packet loss, not just node death.
//
// NNDuplicateSuppressor enforces FIRST-ARRIVAL-WINS for a slot once a
// substitute has claimed it — the opposite of NNInputBuffer's existing
// latest-wins retransmission tolerance: ordinary
// retransmission should take the newest value, but a failover slot must
// not flip later if the original arrives late. NNInputBuffer itself is
// UNCHANGED.

#pragma once
#include <cassert>
#include "NNAddress.h"
#include "NNPacket.h"
#include "NNBuffer.h"
#include "NNActivation.h"
#include "NNNode.h"
 
// Reuses NNPacketHeader::flags, previously entirely reserved/unused.
constexpr uint8_t NN_FLAG_FAILOVER_SUBSTITUTE = 0x01;
 
class NNBackupStandby {
public:
    using ClockFn = unsigned long (*)();
 
    // NNBackupStandby(ownerCfg, clockFn): ownerCfg.backupTargetAddress MUST
    // be a SIBLING -- same layerId as ownerCfg.address. This is a per-layer
    // mechanism by design: nodes within a layer are already continuously
    // observing every sibling's broadcast (the same traffic NNScheduler's
    // own sibling-turn-ordering relies on), so a backup relationship
    // crossing layers would need an entirely different detection mechanism
    // and isn't what this class implements. Enforced with an assertion
    // rather than silently accepting a misconfigured cross-layer pairing.
    NNBackupStandby(const NNNodeConfig& ownerCfg, ClockFn clockFn)
        : cfg(ownerCfg), getTimeMs(clockFn) {
        if (cfg.hasBackupRole) {
            // In a release/embedded build without <cassert> wired up, this
            // is a silent no-op if NDEBUG is defined -- callers should
            // still validate configuration at startup in production.
            // The check exists here specifically so a desktop test build
            // catches a same-layer violation immediately, not in the field.
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
            targetObserved = true;  // the target spoke for itself (original or a
                                      // genuine response to our resend request) —
                                      // our job this pass is done
            return;
        }
 
        // Track EVERY sibling's transmission in this layer, not just the
        // ones this node's own predecessorMask cares about — needed to
        // know when "the round finished" for failure detection below.
        // Mirrors NNScheduler::onPacketObserved's own layer-filtered
        // tracking, kept separate here since NNBackupStandby doesn't have
        // access to NNScheduler's private observedSiblingsMask.
        if (src.layerId == cfg.address.layerId) {
            layerObservedMask |= (uint16_t(1) << src.nodeId);
        }
 
        // FIX (mirrors the identical fix in NNNode::onPacketReceived):
        // filter by the sender's layer BEFORE matching by node ID. Without
        // this, a node ID reused in a different layer than
        // backupTargetPredecessorLayerId could be mistaken for one of the
        // backup target's real predecessors, corrupting the substitute
        // computation with the wrong sender's value.
        if (src.layerId == cfg.backupTargetPredecessorLayerId &&
            (cfg.backupTargetPredecessorMask & (uint16_t(1) << src.nodeId))) {
            inputBuffer.storeInput(src.nodeId, pkt.payload, pkt.header.payloadCount);
        }
    }
 
    // Call once per loop() iteration, same cadence as NNScheduler::tick().
    //
    // FAILURE DETECTION IS EVENT-DRIVEN, NOT CLOCK-BASED: this node
    // already knows every node ID expected in its own layer
    // (cfg.layerRosterMask). Once every OTHER sibling (excluding the
    // target and this node itself) has been observed transmitting, and
    // the target STILL hasn't, that's a complete, timing-free failure
    // signal — "the round finished and my target never went." No clock is
    // consulted for this part at all.
    //
    // A clock is used ONLY after that point: once failure is detected and
    // a resend request is queued, there's no further "everyone
    // transmitted" signal left to wait on (the round already completed),
    // so the grace period for a reply is necessarily time-bounded.
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
 
    void resetForNextPass() {
        inputBuffer.reset();
        targetObserved = false;
        layerObservedMask = 0;
        resendRequested = false;
        hasSubstituted = false;
        hasTornDown = false;
        resolved = false;
        outputPending = false;
        waitStartTime = 0;
        waitStarted = false;
    }
 
private:
    bool isBackupTargetAddress(const NNAddress& addr) const {
        return addr.nodeId == cfg.backupTargetAddress.nodeId &&
               addr.layerId == cfg.backupTargetAddress.layerId &&
               addr.clusterId == cfg.backupTargetAddress.clusterId;
    }
 
    void queueResendRequest() {
        outgoingPacket = NNPacket{};
        outgoingPacket.header.sourceAddress = encodeAddress(cfg.address);  // who is asking
        outgoingPacket.header.type = NNPacketType::CONTROL;
        outgoingPacket.header.payloadCount = 1;
        // Encodes WHICH node's retransmission is being requested. Exact
        // representation: encodeAddress() produces a value well within
        // float32's exact-integer range (<= 2^16 << 2^24).
        outgoingPacket.payload[0] = static_cast<float>(encodeAddress(cfg.backupTargetAddress));
        outputPending = true;
    }
 
    void computeSubstituteOutput() {
        // Same senderId-ascending weight-consumption order as
        // NNNode::execute() — backupWeights must be provided in that same
        // order for cfg.backupTargetPredecessorMask.
        // PATCHED: starts from cfg.backupTargetBias, mirroring the target's
        // own execute() starting from config.bias -- otherwise a substitute
        // silently drops the target's bias term (see NNNodeConfig above).
        float sum = cfg.backupTargetBias;   // was: float sum = 0.0f;
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
    }
 
    void buildTeardownPacket() {
        outgoingPacket = NNPacket{};
        outgoingPacket.header.sourceAddress = encodeAddress(cfg.address);  // reported as OUR OWN
                                                                             // address — teardown is a
                                                                             // diagnostic broadcast, not
                                                                             // an identity substitution
        outgoingPacket.header.type = NNPacketType::TEARDOWN;
        outgoingPacket.header.payloadCount = 0;
        hasTornDown = true;
        outputPending = true;
    }
 
    const NNNodeConfig& cfg;
    ClockFn getTimeMs;
    NNInputBuffer inputBuffer;   // reused, real, UNCHANGED class from NNBuffer.h
    bool targetObserved = false;
    uint16_t layerObservedMask = 0;  // every sibling observed transmitting this pass
    bool resendRequested = false;
    bool hasSubstituted = false;
    bool hasTornDown = false;
    bool resolved = false;
    bool outputPending = false;
    unsigned long waitStartTime = 0;
    bool waitStarted = false;
    NNPacket outgoingPacket{};
};
 
// The RECEIVING side of a resend request. A node that is alive but simply
// hasn't transmitted yet (its packet was lost, or it's mid-slot) uses this
// to recognize its own address in an incoming resend request and re-offer
// its already-computed output. Deliberately does NOT re-execute anything —
// it only re-sends a value the node already computed this pass, supplied
// externally by the caller (see haveOutputThisPass below), since NNNode's
// own execution state is private and this class has no need to reach into
// it — the orchestrating loop already knows whether it has a fresh output
// to offer.
class NNResendResponder {
public:
    explicit NNResendResponder(const NNAddress& ownAddress) : ownAddress(ownAddress) {}
 
    // Call for every observed packet, alongside whatever output value this
    // node has actually computed this pass (if any) and whether it's valid
    // yet. Only remembers to respond if BOTH the request is genuinely
    // addressed to this node AND a real output already exists to resend —
    // a node with nothing computed yet has nothing useful to offer.
    void onPacketObserved(const NNPacket& pkt, bool haveOutputThisPass, float outputValue) {
        if (pkt.header.type != NNPacketType::CONTROL) return;
        if (pkt.header.payloadCount < 1) return;
 
        uint16_t requestedAddr = static_cast<uint16_t>(pkt.payload[0]);
        NNAddress requested = decodeAddress(requestedAddr);
        if (requested.nodeId != ownAddress.nodeId ||
            requested.layerId != ownAddress.layerId ||
            requested.clusterId != ownAddress.clusterId) {
            return;  // not addressed to us
        }
 
        if (!haveOutputThisPass) return;  // nothing to resend yet
 
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
 
// Applied by any RECEIVING node before handing an incoming packet to its
// own NNNode::onPacketReceived() / NNScheduler::onPacketObserved(). Keeps
// NNInputBuffer itself completely unmodified — a separate, composable
// filter, not a change to the synchronization gate's own logic.
class NNDuplicateSuppressor {
public:
    // Returns false if this packet should be DROPPED (a late-arriving
    // "real" packet for a slot a substitute already claimed this pass).
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
 
// --- Desktop-only backup-config validation (see top-of-file note) ---------
 
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
        // If MAX_ISSUES is exceeded, isValid still correctly reports
        // false — only the LISTING of every individual issue is capped,
        // never the overall validity verdict.
    }
};
 
// Confirms a backup node's copy of its target's identity/activation/
// predecessor/weight-count/bias information genuinely matches the target's
// REAL, independently-authored NNNodeConfig. Returns every mismatch
// found, not just the first — useful for a desktop tool reporting a full
// diagnosis at once, and for testing each failure mode independently.
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
 
    // NEW: bias must be mirrored too, for the same reason as every other
    // backupTarget* field above -- see NNNodeConfig::backupTargetBias.
    if (backupCfg.backupTargetBias != targetRealCfg.bias) {
        result.addIssue("backupTargetBias does not match the target's REAL bias");
    }
 
    return result;
}
