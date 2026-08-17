// Device-side firmware: the full NNAware runtime, setup phase over UDP
// broadcast, runtime inference over UDP multicast, and the fault-tolerance
// layer from src/NNFailover.h (backup weights, resend requests, substitute
// outputs and teardown).
//
//Flash this sketch onto each physical device so that the framework can recognize the devices
// you have available, and change NN_HARDWARE_ID for each device

#include "NNSetupProtocol.h"
#include "NNTransportUDP.h"
#include "NNTransportUDPMulticast.h"
#include "NNNode.h"
#include "NNScheduler.h"
#include "NNFailover.h"


// EDIT THESE before flashing:

const char* WIFI_SSID     = "";
const char* WIFI_PASSWORD = "";
const char* BROADCAST_ADDR = "192.168.3.255";  
const uint16_t SETUP_PORT   = 4210;            // must match the app's "Setup port" setting
const uint16_t RUNTIME_PORT = 4211;            // must match the app's "Runtime port" setting

// MUST be unique per physical device 
const uint64_t NN_HARDWARE_ID = 0x0000000000000001ULL;
// ---------------------------------------------------------------------


const uint8_t  NN_DATA_RETRANSMITS       = 3;
const uint16_t NN_DATA_RETRANSMIT_GAP_MS = 150;

//How long a node may sit in WAITING_FOR_TURN before it stops waiting for a silent preceding sibling and transmits anyway
const unsigned long NN_TURN_STALL_MS = 1500;

// Backstop for closing a pass that never resolved.
const unsigned long NN_PASS_IDLE_MS  = 5000;

NNTransportUDP setupTransport(WIFI_SSID, WIFI_PASSWORD, BROADCAST_ADDR, SETUP_PORT);
NNVolatileConfigStore store;  
NNSetupAgent agent(NN_HARDWARE_ID, setupTransport, store);

NNSetupState lastPrintedState = NNSetupState::LOADING;


NNTransportUDPMulticast* runtimeTransport = nullptr;
NNNode*                  runtimeNode      = nullptr;
NNScheduler*             runtimeScheduler = nullptr;

NNBackupStandby*         standby          = nullptr;
NNResendResponder*       responder        = nullptr;
NNDuplicateSuppressor    suppressor;   // by value, no heap


bool  ownOutputSent      = false;
float myOutputValue      = 0.0f;

bool  targetSeenThisPass = false;

bool  loggedSuppression  = false;

bool  passManaged        = true;

unsigned long lastActivityMs   = 0;
unsigned long turnStallSinceMs = 0;
bool          turnStallArmed   = false;

struct RetxSlot {
    NNPacket      pkt;
    uint8_t       left;
    unsigned long dueMs;
};

enum RetxSlotId : uint8_t {
    RETX_OWN_OUTPUT = 0,  // this node's own computed output
    RETX_SUBSTITUTE,      // an output computed on a failed sibling's behalf
    RETX_REPORT,          // the failover report for the laptop
    RETX_SLOT_COUNT,
};
RetxSlot retxSlots[RETX_SLOT_COUNT];

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


void sendOnce(NNPacket& pkt, uint8_t targetLayerId, const char* what) {
    pkt.header.targetLayerId = targetLayerId;
    runtimeTransport->send(pkt);
    if (what == nullptr) return;
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
    for (RetxSlot& slot : retxSlots) {
        if (slot.left == 0) continue;
        // Signed comparison so millis() rollover (~49 days) can't strand a slot.
        if ((long)(now - slot.dueMs) < 0) continue;
        runtimeTransport->send(slot.pkt);
        slot.left--;
        slot.dueMs = now + NN_DATA_RETRANSMIT_GAP_MS;
    }
}


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



bool passResolved() {
    if (!ownOutputSent) return false;
    if (standby == nullptr) return true;
    return targetSeenThisPass || standby->didSubstitute() || standby->didTearDown();
}

void closePass(const char* reason) {
    runtimeScheduler->resetForNextPass();   
    if (standby) standby->resetForNextPass();
    responder->resetForNextPass();
    suppressor.reset();                     

    ownOutputSent      = false;
    myOutputValue      = 0.0f;
    targetSeenThisPass = false;
    turnStallArmed     = false;
    loggedSuppression  = false;
    for (RetxSlot& slot : retxSlots) slot.left = 0;
    lastActivityMs = millis();

    Serial.print("[Failover] pass closed ("); Serial.print(reason);
    Serial.println(") -- ready for the next Run.");
}

void maybeClosePass() {
    if (!passManaged) return;
    if (passResolved()) { closePass("resolved"); return; }

  
    if ((long)(millis() - lastActivityMs) >= (long)NN_PASS_IDLE_MS) {
        closePass(ownOutputSent ? "idle timeout" : "stalled -- never transmitted");
    }
}


// Runtime
void startRuntime() {
    const NNNodeConfig&   cfg = agent.getNodeConfig();
    const NNWindowConfig& win = agent.getWindowConfig();


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
      
        if (cfg.backupTargetAddress.layerId != cfg.address.layerId) {
            Serial.println("[Failover] REFUSING backup role: target is not a sibling in this layer.");
        } else if (cfg.layerRosterMask == 0) {
            Serial.println("[Failover] REFUSING backup role: layerRosterMask is 0.");
        } else {

            standby = new NNBackupStandby(agent.getNodeConfig(), clockMs);

            Serial.print("[Failover] backup duty for L");
            Serial.print(cfg.backupTargetAddress.layerId); Serial.print("_N");
            Serial.print(cfg.backupTargetAddress.nodeId);
            Serial.print("  othersMask=0b"); Serial.print(backupOthersMask(cfg), BIN);
            Serial.print("  resendGraceMs="); Serial.println(cfg.resendGraceMs);
        }
    }

    if (cfg.predecessorMask == 0) {
  
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

     
        if (src.layerId == cfg.predecessorLayerId && !suppressor.shouldAccept(pkt)) {
           
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
            
            responder->onPacketObserved(pkt, ownOutputSent, myOutputValue);
        } else if (pkt.header.type == NNPacketType::TEARDOWN) {
            Serial.print("[Failover] TEARDOWN observed from L"); Serial.print(src.layerId);
            Serial.print("_N"); Serial.println(src.nodeId);
        }
    }

    runtimeScheduler->tick();

 
    if (standby && runtimeScheduler->getState() != NNNodeState::WAITING_FOR_INPUT) {
        standby->tick();
    }

    maybeReleaseStalledTurn();  

    NNPacket outPkt{};

    //this node's own output
    if (runtimeScheduler->hasOutputReady(outPkt)) {
      
        myOutputValue = outPkt.payload[0];
        ownOutputSent = true;
        sendAndRetransmit(outPkt, cfg.successorLayerId, RETX_OWN_OUTPUT, "own output");
       
    }

    // the failover REPORT for the laptop 
    if (standby && standby->hasDiagnosticReady(outPkt)) {
        sendAndRetransmit(outPkt, NN_DIAG_LAYER_GROUP, RETX_REPORT, /*what=*/nullptr);
    }

    // backup standby: resend request, then substitute OR teardown 
    if (standby && standby->hasOutputReady(outPkt)) {
        if (standby->didSubstitute()) {
            Serial.print("[Failover] grace expired, target still SILENT -> SUBSTITUTE = ");
            Serial.println(outPkt.payload[0], 6);
            sendAndRetransmit(outPkt, cfg.successorLayerId, RETX_SUBSTITUTE, "SUBSTITUTE");
            Serial.println("[Failover] failover report sent to the laptop.");
            // If THIS node was itself blocked in WAITING_FOR_TURN on the dead
            // target, the substitute carries exactly the address it is waiting
            // for -- but a board can't be relied on to receive its own multicast
            // back, so feed it to our own scheduler directly.
            runtimeScheduler->onPacketObserved(outPkt);
            targetSeenThisPass = true;
        } else if (standby->didTearDown()) {
            Serial.println("[Failover] grace expired AND the target's inputs never arrived -> TEARDOWN");
            // TEARDOWN tells the next LAYER to abort; the report already sent above
            // tells the laptop WHICH node was lost, so the UI can name it instead of
            // showing an unexplained timeout.
            sendOnce(outPkt, cfg.successorLayerId, "TEARDOWN");
            Serial.println("[Failover] failover report sent to the laptop.");
            targetSeenThisPass = true;
        } else {
            Serial.println("[Failover] round complete but target SILENT -> asking it to retransmit");
            // The target listens on its OWN layer's group, which every node
            // joins unconditionally -- not on successorLayerId.
            sendOnce(outPkt, cfg.address.layerId, "resend REQUEST");
        }
    }

   
    if (responder->hasResendReady(outPkt)) {
        Serial.print("[Failover] answering a resend request for myself -> ");
        Serial.println(outPkt.payload[0], 6);
        sendOnce(outPkt, cfg.successorLayerId, "resend REPLY");
    }

    serviceRetransmits();
    maybeClosePass();
}



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

    
    NNPacket discard{};
    while (setupTransport.receive(discard)) { /* intentionally dropped */ }

    runRuntimeStep();
}
