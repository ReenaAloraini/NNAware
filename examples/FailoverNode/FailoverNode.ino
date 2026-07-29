// Device-side firmware: everything examples/SetupAndRun/SetupANdRUn.ino does
// (setup phase over UDP broadcast, then runtime inference over UDP multicast),
// PLUS the fault-tolerance layer from src/NNFailover.h -- backup weights,
// resend requests, substitute outputs and teardown.
//
// Flash this SAME sketch onto every physical device (edit NN_HARDWARE_ID to a
// DIFFERENT value per board first), then provision from the NNAware Mapper app.
// Which board carries backup duty is decided entirely by what the laptop sends
// over the wire (BACKUP_ROLE_INFO + BACKUP_WEIGHTS_CHUNK) -- there is no
// per-board firmware difference.
//
// WHAT A BACKUP ROLE DOES (see NNFailover.h's own header comment): every node
// optionally stands ready to compute on behalf of exactly ONE SIBLING in its own
// layer. If that sibling goes silent, this node first asks it to retransmit;
// if it still doesn't answer within resendGraceMs, this node computes the
// missing output itself from a mirrored copy of the sibling's weights and bias,
// and broadcasts it under the SIBLING'S OWN identity so the next layer never
// knows the difference.
//
// IMPORTANT CAVEATS:
//
// 1. This has NOT been compiled or tested against real devices -- there is no
//    Arduino toolchain available in this project. Review it before trusting it.
//
// 2. The setup protocol still has no "start a new inference pass" message, so
//    this sketch decides for itself when a pass is over -- see closePass().
//    Unlike SetupAndRun/RunningNode, it does NOT reset immediately after
//    transmitting: a backup node transmits its own output long before its
//    detection window closes, and resetting there would discard the sibling
//    observations the standby still needs. Sketch policy, not a library
//    requirement.
//
// 3. Two behaviours here exist purely to make EVERY backup shape work, rather
//    than restricting which sibling may back up which. Both are sketch-level
//    policy built on public library API; neither patches the library. They are
//    marked [F1] and [F2] below and explained at their implementation sites.
#include "NNSetupProtocol.h"
#include "NNTransportUDP.h"
#include "NNTransportUDPMulticast.h"
#include "NNNode.h"
#include "NNScheduler.h"
#include "NNFailover.h"

// ---------------------------------------------------------------------
// EDIT THESE before flashing:
// ---------------------------------------------------------------------
const char* WIFI_SSID     = "";
const char* WIFI_PASSWORD = "";
const char* BROADCAST_ADDR = "192.168.3.255";  // set for your 192.168.3.x/24 network -- re-check with
                                                // `ipconfig /all` if your subnet mask isn't 255.255.255.0
const uint16_t SETUP_PORT   = 4210;            // must match the app's "Setup port" setting
// Deliberately DIFFERENT from SETUP_PORT: this device has two separate WiFiUDP
// sockets alive at once, and keeping them on separate ports also guarantees
// setup-phase CONTROL traffic can never reach NNResendResponder, which reads
// payload[0] of any CONTROL packet as an address (see runRuntimeStep).
const uint16_t RUNTIME_PORT = 4211;            // must match the app's "Runtime port" setting

// MUST be unique per physical device -- this is how the laptop tool and this
// device recognize which manifest entry belongs to which board.
const uint64_t NN_HARDWARE_ID = 0x0000000000000001ULL;
// ---------------------------------------------------------------------

// DATA packets have no ACK/retry layer and WiFi multicast frames get no
// MAC-level ACK+retry, so a single send can simply be lost. Resend the
// identical packet a few times. Safe to duplicate: NNInputBuffer::storeInput()
// and NNScheduler::onPacketObserved() are both idempotent for a repeated
// identical packet.
//
// Unlike RunningNode.ino these resends are NON-BLOCKING. RunningNode justifies
// its delay(150) with "with no backup role configured in this sketch there's
// nothing else it needs to do" -- which is false here. ~300ms of blocking would
// swallow most of a 300ms resendGraceMs budget and drop incoming resend
// requests on the floor.
const uint8_t  NN_DATA_RETRANSMITS       = 3;
const uint16_t NN_DATA_RETRANSMIT_GAP_MS = 150;

// [F2] How long a node may sit in WAITING_FOR_TURN before it stops waiting for
// a silent preceding sibling and transmits anyway. See maybeReleaseStalledTurn().
const unsigned long NN_TURN_STALL_MS = 1500;
// Backstop for closing a pass that never resolved. MUST stay comfortably larger
// than NN_TURN_STALL_MS so stalled siblings get their chance to transmit first.
const unsigned long NN_PASS_IDLE_MS  = 5000;

NNTransportUDP setupTransport(WIFI_SSID, WIFI_PASSWORD, BROADCAST_ADDR, SETUP_PORT);
NNVolatileConfigStore store;  // no persistence yet -- see NNSetupProtocol.h's own note on this
NNSetupAgent agent(NN_HARDWARE_ID, setupTransport, store);

NNSetupState lastPrintedState = NNSetupState::LOADING;

// Constructed only once setup finishes -- we don't know this device's layer or
// role (and therefore which multicast group to join, or whether it has backup
// duty at all) until then.
NNTransportUDPMulticast* runtimeTransport = nullptr;
NNNode*                  runtimeNode      = nullptr;
NNScheduler*             runtimeScheduler = nullptr;
// nullptr means "no backup duty" -- checked at every call site. Better than
// constructing one and testing isActive(): NNBackupStandby owns a private
// NNInputBuffer (16 x 16 floats = 1KB of RAM) that a node without backup duty
// should never pay for.
NNBackupStandby*         standby          = nullptr;
NNResendResponder*       responder        = nullptr;
NNDuplicateSuppressor    suppressor;   // by value, no heap

// --- per-pass state owned by this sketch, not by the library ---------------
// NNNode exposes no getter for its computed output, and NNResendResponder needs
// that value handed to it, so we cache it at the one moment it is observable.
// ownOutputSent doubles as "myOutputValue is valid", which is exactly what
// NNResendResponder wants for its haveOutputThisPass argument.
bool  ownOutputSent      = false;
float myOutputValue      = 0.0f;
// NNBackupStandby has no targetObserved() getter, so track it here for logging
// and for deciding when a pass is resolved.
bool  targetSeenThisPass = false;
// Set once per pass so the duplicate-suppressor drop is reported once, not once
// per dropped frame -- a substitute arrives NN_DATA_RETRANSMITS times, and
// Serial.print blocks inside the receive-drain loop.
bool  loggedSuppression  = false;
// A predecessorMask==0 node's value is fixed at provisioning time, so it fires
// exactly once and must never be reset (see RunningNode.ino's own reasoning).
bool  passManaged        = true;

unsigned long lastActivityMs   = 0;
unsigned long turnStallSinceMs = 0;
bool          turnStallArmed   = false;

struct RetxSlot {
    NNPacket      pkt;
    uint8_t       left;
    unsigned long dueMs;
};
RetxSlot retxSlots[2];  // 0 = this node's own output, 1 = a substitute

// NNBackupStandby::ClockFn
unsigned long clockMs() { return millis(); }

const char* stateName(NNSetupState s) {
    switch (s) {
        case NNSetupState::LOADING:           return "LOADING";
        case NNSetupState::UNCONFIGURED:      return "UNCONFIGURED (announcing HELLO)";
        case NNSetupState::RECEIVING_CONFIG:  return "RECEIVING_CONFIG";
        case NNSetupState::VERIFYING:         return "VERIFYING";
        case NNSetupState::CONFIGURED:        return "CONFIGURED (waiting for START)";
        case NNSetupState::RUNNING:           return "RUNNING";
    }
    return "UNKNOWN";
}

// The bits NNBackupStandby::tick() waits on before it will declare the round
// finished: every node in this layer EXCEPT the backup target and this node
// itself. Mirrors NNFailover.h:127-129 exactly. Printed at startup because it
// is the single number that tells you whether a given backup pairing can
// detect a failure at all.
uint16_t backupOthersMask(const NNNodeConfig& cfg) {
    return cfg.layerRosterMask
         & ~(uint16_t(1) << cfg.backupTargetAddress.nodeId)
         & ~(uint16_t(1) << cfg.address.nodeId);
}

bool isBackupTarget(const NNAddress& src) {
    const NNNodeConfig& cfg = agent.getNodeConfig();
    return cfg.hasBackupRole
        && src.nodeId    == cfg.backupTargetAddress.nodeId
        && src.layerId   == cfg.backupTargetAddress.layerId
        && src.clusterId == cfg.backupTargetAddress.clusterId;
}


void printNodeConfig() {
    const NNNodeConfig& cfg = agent.getNodeConfig();
    Serial.println("=== SETUP COMPLETE -- NNNodeConfig ===");
    Serial.print("  address: node="); Serial.print(cfg.address.nodeId);
    Serial.print(" layer="); Serial.print(cfg.address.layerId);
    Serial.print(" cluster="); Serial.println(cfg.address.clusterId);
    Serial.print("  predecessorMask: 0b"); Serial.println(cfg.predecessorMask, BIN);
    Serial.print("  predecessorLayerId: "); Serial.println(cfg.predecessorLayerId);
    Serial.print("  successorLayerId: "); Serial.println(cfg.successorLayerId);
    Serial.print("  transmitSlot: "); Serial.println(cfg.transmitSlot);
    Serial.print("  activationType: "); Serial.println(static_cast<int>(cfg.activationType));
    Serial.print("  weightCount: "); Serial.println(cfg.weightCount);
    for (uint8_t i = 0; i < cfg.weightCount; i++) {
        Serial.print("    weights["); Serial.print(i); Serial.print("] = ");
        Serial.println(cfg.weights[i], 6);
    }
    Serial.print("  bias: "); Serial.println(cfg.bias, 6);
    Serial.print("  hasBackupRole: "); Serial.println(cfg.hasBackupRole ? "true" : "false");
    if (cfg.hasBackupRole) {
        Serial.print("    backupTargetAddress: node="); Serial.print(cfg.backupTargetAddress.nodeId);
        Serial.print(" layer="); Serial.println(cfg.backupTargetAddress.layerId);
        Serial.print("    backupTargetPredecessorMask: 0b");
        Serial.println(cfg.backupTargetPredecessorMask, BIN);
        Serial.print("    backupTargetPredecessorLayerId: ");
        Serial.println(cfg.backupTargetPredecessorLayerId);
        Serial.print("    backupTargetActivationType: ");
        Serial.println(static_cast<int>(cfg.backupTargetActivationType));
        Serial.print("    backupTargetBias: "); Serial.println(cfg.backupTargetBias, 6);
        Serial.print("    backupWeightCount: "); Serial.println(cfg.backupWeightCount);
        for (uint8_t i = 0; i < cfg.backupWeightCount; i++) {
            Serial.print("      backupWeights["); Serial.print(i); Serial.print("] = ");
            Serial.println(cfg.backupWeights[i], 6);
        }
        Serial.print("    layerRosterMask: 0b"); Serial.println(cfg.layerRosterMask, BIN);
        Serial.print("    resendGraceMs: "); Serial.println(cfg.resendGraceMs);
        Serial.print("    (derived) othersMask: 0b"); Serial.println(backupOthersMask(cfg), BIN);
    }
    Serial.println("=======================================");
}

// ---------------------------------------------------------------------------
// Sending
// ---------------------------------------------------------------------------

// EVERY packet builder in NNFailover.h -- queueResendRequest(),
// computeSubstituteOutput(), buildTeardownPacket() and
// NNResendResponder::hasResendReady() -- starts from a fresh NNPacket{} and
// never assigns header.targetLayerId, so it stays 0. NNNode::buildOutputPacket()
// is the ONLY builder in the whole library that sets it.
// NNTransportUDPMulticast::send() routes solely on that field
// (layerGroup(targetLayerId) == 239.1.0.<targetLayerId>), so an un-fixed
// failover packet is multicast to 239.1.0.0 -- a group no compute node ever
// joins -- and is silently lost. Every send in this sketch therefore goes
// through here with an explicit destination layer.
void sendOnce(NNPacket& pkt, uint8_t targetLayerId, const char* what) {
    pkt.header.targetLayerId = targetLayerId;
    runtimeTransport->send(pkt);
    Serial.print("[Failover] sent "); Serial.print(what);
    Serial.print(" -> layer group "); Serial.println(targetLayerId);
}

void sendAndRetransmit(NNPacket& pkt, uint8_t targetLayerId, uint8_t slot, const char* what) {
    sendOnce(pkt, targetLayerId, what);   // also stamps header.targetLayerId
    retxSlots[slot].pkt   = pkt;
    retxSlots[slot].left  = NN_DATA_RETRANSMITS - 1;
    retxSlots[slot].dueMs = millis() + NN_DATA_RETRANSMIT_GAP_MS;
}

void serviceRetransmits() {
    unsigned long now = millis();
    for (uint8_t i = 0; i < 2; i++) {
        if (retxSlots[i].left == 0) continue;
        // Signed comparison so millis() rollover (~49 days) can't strand a slot.
        if ((long)(now - retxSlots[i].dueMs) < 0) continue;
        runtimeTransport->send(retxSlots[i].pkt);
        retxSlots[i].left--;
        retxSlots[i].dueMs = now + NN_DATA_RETRANSMIT_GAP_MS;
    }
}

// ---------------------------------------------------------------------------
// [F2] Turn-stall release
// ---------------------------------------------------------------------------
// Devices transmit in node-id order: precedingSiblingsMask == (1 << nodeId) - 1
// (generate_manifest.py), and NNScheduler holds a node in WAITING_FOR_TURN
// until every preceding sibling has been observed. So if node T dies, EVERY
// node with a higher id waits on it forever -- they never transmit, which in
// turn means T's backup never sees "the round finished" and never substitutes.
// The whole layer deadlocks silently: no request, no substitute, no teardown.
//
// The fix is to stop waiting. NNScheduler::onPacketObserved() reads ONLY the
// source address and ORs one bit into observedSiblingsMask -- it never touches
// NNNode's input buffer -- so an address-only marker releases this node's turn
// without fabricating any value or corrupting anyone's weighted sum.
//
// Transmitting out of order is safe: turn-taking exists to stagger access to a
// shared medium, and NNInputBuffer is keyed by sender id, so arrival order is
// irrelevant to correctness. Late-and-correct beats deadlocked.
void releaseTurn() {
    const NNNodeConfig&   cfg = agent.getNodeConfig();
    const NNWindowConfig& win = agent.getWindowConfig();

    for (uint8_t nid = 0; nid < NN_MAX_PREDECESSORS; nid++) {
        if (!(win.precedingSiblingsMask & (uint16_t(1) << nid))) continue;
        NNPacket marker{};
        NNAddress src;
        src.nodeId    = nid;
        src.layerId   = cfg.address.layerId;
        src.clusterId = cfg.address.clusterId;
        src.reserved  = 0;
        marker.header.sourceAddress = encodeAddress(src);
        marker.header.type          = NNPacketType::DATA;
        marker.header.payloadCount  = 0;   // ordering only -- carries no value
        runtimeScheduler->onPacketObserved(marker);   // NOT runtimeNode->onPacketReceived()
    }
    Serial.println("[Failover] turn stalled too long -- releasing and transmitting out of order.");
}

void maybeReleaseStalledTurn() {
    if (runtimeScheduler->getState() != NNNodeState::WAITING_FOR_TURN) {
        turnStallArmed = false;
        return;
    }
    if (!turnStallArmed) {
        turnStallArmed   = true;
        turnStallSinceMs = millis();
        return;
    }
    if ((long)(millis() - turnStallSinceMs) < (long)NN_TURN_STALL_MS) return;

    releaseTurn();
    turnStallArmed = false;
}

// ---------------------------------------------------------------------------
// Pass lifecycle
// ---------------------------------------------------------------------------
// Only called from maybeClosePass(), which has already checked passManaged.
bool passResolved() {
    if (!ownOutputSent) return false;
    if (standby == nullptr) return true;
    return targetSeenThisPass || standby->didSubstitute() || standby->didTearDown();
}

void closePass(const char* reason) {
    runtimeScheduler->resetForNextPass();   // also resets the NNNode's input buffer
    if (standby) standby->resetForNextPass();
    responder->resetForNextPass();
    suppressor.reset();                     // NOTE: reset(), NOT resetForNextPass()

    ownOutputSent      = false;
    myOutputValue      = 0.0f;
    targetSeenThisPass = false;
    turnStallArmed     = false;
    loggedSuppression  = false;
    for (uint8_t i = 0; i < 2; i++) retxSlots[i].left = 0;
    lastActivityMs = millis();

    Serial.print("[Failover] pass closed ("); Serial.print(reason);
    Serial.println(") -- ready for the next Run.");
}

void maybeClosePass() {
    if (!passManaged) return;
    if (passResolved()) { closePass("resolved"); return; }

    // Backstop: something never arrived. Close anyway so the next Run isn't
    // stuck behind a half-finished pass.
    if ((long)(millis() - lastActivityMs) >= (long)NN_PASS_IDLE_MS) {
        closePass(ownOutputSent ? "idle timeout" : "stalled -- never transmitted");
    }
}

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------
void startRuntime() {
    const NNNodeConfig&   cfg = agent.getNodeConfig();
    const NNWindowConfig& win = agent.getWindowConfig();

    // Siblings broadcast their output to the NEXT layer's group (their
    // successorLayerId), never to their own -- so a node that needs to watch its
    // siblings has to join that group too, observation-only.
    //
    // SetupAndRun/RunningNode gate this on precedingSiblingsMask alone. That is
    // wrong for a backup node in transmit slot 0: its mask is 0, so it would
    // never join, never observe its target, and NNBackupStandby's targetObserved
    // would stay false -- making it substitute for a perfectly healthy sibling
    // on every single pass. A backup role needs the group regardless of slot.
    bool needSiblings = (win.precedingSiblingsMask != 0) || cfg.hasBackupRole;
    uint8_t siblingGroup = needSiblings ? cfg.successorLayerId : NN_NO_SIBLING_GROUP;

    Serial.print("[Failover] joining multicast group for layer ");
    Serial.println(cfg.address.layerId);
    if (siblingGroup != NN_NO_SIBLING_GROUP) {
        Serial.print("[Failover] also joining sibling-observation group for layer ");
        Serial.println(siblingGroup);
    }

    runtimeTransport = new NNTransportUDPMulticast(WIFI_SSID, WIFI_PASSWORD, cfg.address.layerId,
                                                   RUNTIME_PORT, siblingGroup);
    if (!runtimeTransport->begin()) {
        Serial.println("[Failover] multicast join FAILED -- runtime will not receive inputs.");
        return;
    }

    runtimeNode      = new NNNode(cfg);                    // NNNode COPIES cfg
    runtimeScheduler = new NNScheduler(*runtimeNode, win);
    responder        = new NNResendResponder(cfg.address);  // takes NNAddress by value

    if (cfg.hasBackupRole) {
        // NNBackupStandby's constructor asserts the target is a same-layer
        // sibling. Arduino builds normally leave NDEBUG undefined, so a bad
        // config would abort() the board with no explanation. device_manifest.py
        // already rejects this laptop-side, so this can only fire on a
        // hand-edited manifest -- which is exactly the case worth surviving.
        if (cfg.backupTargetAddress.layerId != cfg.address.layerId) {
            Serial.println("[Failover] REFUSING backup role: target is not a sibling in this layer.");
        } else if (cfg.layerRosterMask == 0) {
            Serial.println("[Failover] REFUSING backup role: layerRosterMask is 0.");
        } else {
            // MUST bind to agent.getNodeConfig(): NNBackupStandby stores the
            // config BY REFERENCE (unlike NNNode, which copies). A local
            // NNNodeConfig here would dangle the moment this function returns.
            standby = new NNBackupStandby(agent.getNodeConfig(), clockMs);

            Serial.print("[Failover] backup duty for L");
            Serial.print(cfg.backupTargetAddress.layerId); Serial.print("_N");
            Serial.print(cfg.backupTargetAddress.nodeId);
            Serial.print("  othersMask=0b"); Serial.print(backupOthersMask(cfg), BIN);
            Serial.print("  resendGraceMs="); Serial.println(cfg.resendGraceMs);
        }
    }

    if (cfg.predecessorMask == 0) {
        // A real physical input-layer node: its value was pushed at provisioning
        // time (INPUT_VALUE, deliberately separate from bias), so it fires once
        // and is never reset -- resetting would clear the seed and, since
        // readyToExecute() is trivially true for an empty predecessorMask, it
        // would then emit activation(bias) forever.
        passManaged = false;
        if (agent.hasSeedInputValue()) {
            Serial.print("[Failover] predecessorMask==0 -- seeding input value ");
            Serial.println(agent.getInputValue(), 6);
            runtimeNode->seedOutput(agent.getInputValue());
            runtimeScheduler->notifySeeded();
        } else {
            Serial.println("[Failover] predecessorMask==0 but no INPUT_VALUE was sent -- "
                           "this node has nothing to emit.");
        }
    }

    lastActivityMs = millis();
    Serial.println("[Failover] runtime ready -- waiting for inputs.");
}

void runRuntimeStep() {
    if (runtimeScheduler == nullptr) return;
    const NNNodeConfig& cfg = agent.getNodeConfig();

    runtimeTransport->poll();

    NNPacket pkt{};
    while (runtimeTransport->receive(pkt)) {
        NNAddress src = decodeAddress(pkt.header.sourceAddress);
        lastActivityMs = millis();

        // NNDuplicateSuppressor locks a slot by src.nodeId ALONE, ignoring
        // layerId. This socket carries both this node's predecessor layer and
        // (when joined) the sibling-observation group, so an unfiltered
        // suppressor would let a substitute for sibling nodeId k lock out a
        // genuine predecessor packet from a different layer that reuses id k.
        // Only gate the traffic it exists to protect: real predecessor inputs.
        if (src.layerId == cfg.predecessorLayerId && !suppressor.shouldAccept(pkt)) {
            // Logged once per pass, not per frame: a substitute is itself sent
            // NN_DATA_RETRANSMITS times, so its own copies land here too, and
            // Serial.print blocks this drain loop during the recovery window.
            if (!loggedSuppression) {
                loggedSuppression = true;
                Serial.println("[Failover] a substitute claimed a slot -- further packets "
                               "for it are being dropped this pass.");
            }
            continue;
        }

        if (pkt.header.type == NNPacketType::DATA) {
            runtimeNode->onPacketReceived(pkt);        // self-filters on predecessorLayerId
            runtimeScheduler->onPacketObserved(pkt);   // self-filters on ownLayerId
            if (standby) standby->onPacketObserved(pkt);

            if (standby && isBackupTarget(src) && !targetSeenThisPass) {
                targetSeenThisPass = true;
                if (standby->didRequestResend()) {
                    Serial.println("[Failover] target ANSWERED the resend request -- recovered by "
                                   "retransmission, no substitute needed.");
                } else {
                    Serial.println("[Failover] target transmitted normally -- standing down this pass.");
                }
            }
        } else if (pkt.header.type == NNPacketType::CONTROL) {
            // This is the RUNTIME socket. Setup-phase CONTROL packets carry raw
            // struct bytes reinterpreted as floats and would be garbage here --
            // they can't reach us because setup lives on a different port.
            responder->onPacketObserved(pkt, ownOutputSent, myOutputValue);
        } else if (pkt.header.type == NNPacketType::TEARDOWN) {
            Serial.print("[Failover] TEARDOWN observed from L"); Serial.print(src.layerId);
            Serial.print("_N"); Serial.println(src.nodeId);
        }
    }

    runtimeScheduler->tick();

    // [F1] Only tick the standby once this node's OWN predecessor inputs are
    // complete -- i.e. the round has genuinely started.
    //
    // NNBackupStandby::tick() treats "every bit in othersMask observed" as
    // "the round finished". In a 2-node layer othersMask is 0, so that test
    // passes on the very first tick, before any input exists. resendGraceMs
    // later its input buffer is still empty, so it takes the no-recovery branch
    // and broadcasts a TEARDOWN seconds after START -- before the user has even
    // clicked Run.
    //
    // Leaving WAITING_FOR_INPUT means readyToExecute() returned true, so this
    // node's predecessor set is complete. Siblings share predecessors, so the
    // standby's own mirrored input buffer is complete at that same moment --
    // which turns that spurious teardown into a correct SUBSTITUTE and leaves
    // TEARDOWN meaning what it should: the inputs really are missing.
    if (standby && runtimeScheduler->getState() != NNNodeState::WAITING_FOR_INPUT) {
        standby->tick();
    }

    maybeReleaseStalledTurn();   // [F2]

    NNPacket outPkt{};

    // --- 1. this node's own output ---
    if (runtimeScheduler->hasOutputReady(outPkt)) {
        // The ONLY moment this node's output value is observable -- NNNode has
        // no getter, and NNResendResponder needs it to answer a resend request.
        myOutputValue = outPkt.payload[0];
        ownOutputSent = true;
        sendAndRetransmit(outPkt, cfg.successorLayerId, /*slot=*/0, "own output");
        // Deliberately NOT resetting here -- see caveat 2 at the top of the file.
    }

    // --- 2. backup standby: resend request, then substitute OR teardown ---
    // All three come out of the same channel, one at a time; classify with the
    // did*() flags, which are set by the builders before the packet is handed
    // over. Check the terminal outcomes first -- the else branch is the request.
    if (standby && standby->hasOutputReady(outPkt)) {
        if (standby->didSubstitute()) {
            Serial.print("[Failover] grace expired, target still SILENT -> SUBSTITUTE = ");
            Serial.println(outPkt.payload[0], 6);
            sendAndRetransmit(outPkt, cfg.successorLayerId, /*slot=*/1, "SUBSTITUTE");
            // If THIS node was itself blocked in WAITING_FOR_TURN on the dead
            // target, the substitute carries exactly the address it is waiting
            // for -- but a board can't be relied on to receive its own multicast
            // back, so feed it to our own scheduler directly.
            runtimeScheduler->onPacketObserved(outPkt);
            targetSeenThisPass = true;
        } else if (standby->didTearDown()) {
            Serial.println("[Failover] grace expired AND the target's inputs never arrived -> TEARDOWN");
            sendOnce(outPkt, cfg.successorLayerId, "TEARDOWN");
            targetSeenThisPass = true;
        } else {
            Serial.println("[Failover] round complete but target SILENT -> asking it to retransmit");
            // The target listens on its OWN layer's group, which every node
            // joins unconditionally -- not on successorLayerId.
            sendOnce(outPkt, cfg.address.layerId, "resend REQUEST");
        }
    }

    // --- 3. we are the node someone asked to retransmit ---
    if (responder->hasResendReady(outPkt)) {
        Serial.print("[Failover] answering a resend request for myself -> ");
        Serial.println(outPkt.payload[0], 6);
        sendOnce(outPkt, cfg.successorLayerId, "resend REPLY");
    }

    serviceRetransmits();
    maybeClosePass();
}

// ---------------------------------------------------------------------------

void setup() {
    Serial.begin(115200);
    delay(1500);

    Serial.print("[Failover] hardwareId = 0x");
    Serial.println((uint32_t)(NN_HARDWARE_ID >> 32), HEX);  // Wio Terminal's Serial has no 64-bit HEX print
    Serial.println((uint32_t)(NN_HARDWARE_ID & 0xFFFFFFFF), HEX);

    if (!setupTransport.begin()) {
        Serial.println("[Failover] WiFi connect failed -- halting.");
        while (true) delay(1000);
    }

    agent.begin();
    Serial.print("[Failover] initial state: ");
    Serial.println(stateName(agent.getState()));
    lastPrintedState = agent.getState();
}

void loop() {
    if (!agent.isRunning()) {
        agent.tick(millis());

        NNPacket pkt{};
        while (setupTransport.receive(pkt)) {
            if (pkt.header.type == NNPacketType::CONTROL) {
                agent.onSetupPacket(pkt);
            }
        }

        NNSetupState current = agent.getState();
        if (current != lastPrintedState) {
            Serial.print("[Failover] state -> ");
            Serial.println(stateName(current));
            lastPrintedState = current;

            if (current == NNSetupState::CONFIGURED) printNodeConfig();
            if (current == NNSetupState::RUNNING) {
                Serial.println("[Failover] RUNNING -- setup phase finished successfully.");
                startRuntime();
            }
        }
        return;
    }

    // Keep draining the setup socket so its receive buffer can't fill up, but
    // DISCARD everything. SetupAndRun.ino keeps feeding agent.onSetupPacket()
    // forever, which is unsafe here: handleAssignAddress() checks only the
    // hardwareId and has no state guard, so a stray or replayed ASSIGN_ADDRESS
    // would rewrite nodeConfig.address at any time -- and because
    // NNBackupStandby holds that same config BY REFERENCE, the change would
    // land live inside a running standby.
    NNPacket discard{};
    while (setupTransport.receive(discard)) { /* intentionally dropped */ }

    runRuntimeStep();
}
