// Device-side firmware: setup phase (src/NNSetupProtocol.h) EXACTLY as in
// SetupTest.ino, extended to continue into real runtime inference once
// setup finishes -- NNNode/NNScheduler driven over NNTransportUDPMulticast,
// per the project's choice of UDP broadcast for setup, UDP multicast for
// running. Flash this onto each Wio Terminal / Seeeduino device, edit
// NN_HARDWARE_ID to a DIFFERENT value per physical device first, then run
// tools/nn_setup/setup_tool.py (or the NNAware Mapper app's Setup step)
// from a laptop on the same WiFi network to provision it over the air.
//
// IMPORTANT CAVEATS (read before relying on this against real hardware --
// this has not been compiled or tested against real devices; there is no
// toolchain available to do that from where this was written):
//
// 1. The setup protocol has no "start a new inference pass" message. Once
//    a device transmits its own output, NNNode's hasExecuted flag blocks it
//    from ever running again unless something resets it. Nothing in
//    NNNode.h/NNScheduler.h specifies WHEN that reset should happen -- this
//    sketch resets each device immediately after it transmits, so it's
//    ready for a fresh round the next time the app sends new inputs. This
//    is this sketch's own design choice, not a library requirement.
//
// 2. Sibling turn-taking (precedingSiblingsMask) over UDP multicast: fixed
//    by a patch to NNTransportUDPMulticast.h -- see that file's own comment
//    for the full explanation. Short version: this sketch now passes each
//    device's successorLayerId as a SECOND group to join, purely so it can
//    observe its own siblings' transmissions (which go to the NEXT layer's
//    group, not their own). Only actually joins that second group when
//    precedingSiblingsMask != 0, so a single-neuron layer (like the
//    AND-gate example) behaves exactly as before -- one socket, no change.
#include "NNSetupProtocol.h"
#include "NNTransportUDP.h"
#include "NNTransportUDPMulticast.h"
#include "NNScheduler.h"

// ---------------------------------------------------------------------
// EDIT THESE before flashing:
// ---------------------------------------------------------------------
const char* WIFI_SSID     = "";
const char* WIFI_PASSWORD = "";
const char* BROADCAST_ADDR = "192.168.3.255";  // set for your 192.168.3.x/24 network -- re-check with
                                                // `ipconfig /all` if your subnet mask isn't 255.255.255.0
const uint16_t SETUP_PORT   = 4210;            // must match the app's "Setup port" setting
// Deliberately DIFFERENT from SETUP_PORT: this device has two separate
// WiFiUDP sockets alive at once (setupTransport + runtimeTransport), and
// whether this board's WiFi stack tolerates two sockets sharing one port
// hasn't been tested against real hardware -- using a different port
// sidesteps the question entirely rather than risking it.
const uint16_t RUNTIME_PORT = 4211;            // must match the app's "Runtime port" setting

// MUST be unique per physical device -- this is how the laptop tool and
// this device recognize which manifest entry belongs to which board.
// There's no automatic per-chip ID available here (unlike ESP32's
// ESP.getEfuseMac()), so just pick a different constant for each device
// you flash, e.g. 0x0000000000000001ULL, 0x0000000000000002ULL, ...
const uint64_t NN_HARDWARE_ID = 0x0000000000000001ULL;
// ---------------------------------------------------------------------

NNTransportUDP setupTransport(WIFI_SSID, WIFI_PASSWORD, BROADCAST_ADDR, SETUP_PORT);
NNVolatileConfigStore store;  // no persistence yet -- see NNSetupProtocol.h's own note on this
NNSetupAgent agent(NN_HARDWARE_ID, setupTransport, store);

NNSetupState lastPrintedState = NNSetupState::LOADING;

// Constructed only once setup finishes -- we don't know this device's
// layer/role (and therefore which multicast group to join) until then.
NNTransportUDPMulticast* runtimeTransport = nullptr;
NNNode* runtimeNode = nullptr;
NNScheduler* runtimeScheduler = nullptr;

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

void printNodeConfig() {
    const NNNodeConfig& cfg = agent.getNodeConfig();
    Serial.println("=== SETUP COMPLETE -- NNNodeConfig ===");
    Serial.print("  address: node="); Serial.print(cfg.address.nodeId);
    Serial.print(" layer="); Serial.print(cfg.address.layerId);
    Serial.print(" cluster="); Serial.println(cfg.address.clusterId);
    Serial.print("  predecessorMask: 0b"); Serial.println(cfg.predecessorMask, BIN);
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
        Serial.print("    backupWeightCount: "); Serial.println(cfg.backupWeightCount);
        for (uint8_t i = 0; i < cfg.backupWeightCount; i++) {
            Serial.print("      backupWeights["); Serial.print(i); Serial.print("] = ");
            Serial.println(cfg.backupWeights[i], 6);
        }
        Serial.print("    resendGraceMs: "); Serial.println(cfg.resendGraceMs);
    }
    Serial.println("=======================================");
}

// Called once, the first time we observe RUNNING -- builds the real
// NNNode/NNScheduler runtime from the config the setup phase just
// assembled, and joins this device's own layer's multicast group.
void startRuntime() {
    const NNNodeConfig& cfg = agent.getNodeConfig();
    const NNWindowConfig& windowCfg = agent.getWindowConfig();

    // Only join the sibling-observation group if this device actually has
    // siblings to wait for -- see NNTransportUDPMulticast.h's own comment.
    uint8_t siblingGroup = (windowCfg.precedingSiblingsMask != 0)
        ? cfg.successorLayerId
        : NN_NO_SIBLING_GROUP;

    Serial.print("[SetupAndRunTest] joining multicast group for layer ");
    Serial.println(cfg.address.layerId);
    if (siblingGroup != NN_NO_SIBLING_GROUP) {
        Serial.print("[SetupAndRunTest] also joining sibling-observation group for layer ");
        Serial.println(siblingGroup);
    }

    runtimeTransport = new NNTransportUDPMulticast(WIFI_SSID, WIFI_PASSWORD, cfg.address.layerId,
                                                     RUNTIME_PORT, siblingGroup);
    if (!runtimeTransport->begin()) {
        Serial.println("[SetupAndRunTest] multicast join FAILED -- runtime will not receive inputs.");
        return;
    }

    runtimeNode = new NNNode(cfg);
    runtimeScheduler = new NNScheduler(*runtimeNode, windowCfg);
    Serial.println("[SetupAndRunTest] runtime ready -- waiting for inputs.");
}

// Drives one iteration of the real inference pipeline: observe traffic,
// advance the state machine, transmit + self-reset once this device's
// output is ready. Safe to call every loop() iteration once runtime exists.
void runRuntimeStep() {
    if (runtimeScheduler == nullptr) return;

    runtimeTransport->poll();
    NNPacket pkt{};
    while (runtimeTransport->receive(pkt)) {
        if (pkt.header.type == NNPacketType::DATA) {
            runtimeNode->onPacketReceived(pkt);
            runtimeScheduler->onPacketObserved(pkt);
        }
    }

    runtimeScheduler->tick();

    NNPacket outPkt{};
    if (runtimeScheduler->hasOutputReady(outPkt)) {
        runtimeTransport->send(outPkt);
        Serial.print("[SetupAndRunTest] TRANSMITTED output = ");
        Serial.println(outPkt.payload[0], 6);
        // See caveat (1) at the top of this file -- self-reset so this
        // device is ready for the next inference pass.
        runtimeScheduler->resetForNextPass();
        Serial.println("[SetupAndRunTest] reset -- ready for the next run.");
    }
}

void setup() {
    Serial.begin(115200);
    delay(1500);

    Serial.print("[SetupAndRunTest] hardwareId = 0x");
    Serial.println((uint32_t)(NN_HARDWARE_ID >> 32), HEX);  // Wio Terminal's Serial has no 64-bit HEX print
    Serial.println((uint32_t)(NN_HARDWARE_ID & 0xFFFFFFFF), HEX);

    if (!setupTransport.begin()) {
        Serial.println("[SetupAndRunTest] WiFi connect failed -- halting.");
        while (true) delay(1000);
    }

    agent.begin();
    Serial.print("[SetupAndRunTest] initial state: ");
    Serial.println(stateName(agent.getState()));
    lastPrintedState = agent.getState();
}

void loop() {
    agent.tick(millis());

    NNPacket pkt{};
    while (setupTransport.receive(pkt)) {
        if (pkt.header.type == NNPacketType::CONTROL) {
            agent.onSetupPacket(pkt);
        }
    }

    NNSetupState current = agent.getState();
    if (current != lastPrintedState) {
        Serial.print("[SetupAndRunTest] state -> ");
        Serial.println(stateName(current));
        lastPrintedState = current;

        if (current == NNSetupState::CONFIGURED) {
            printNodeConfig();
        }
        if (current == NNSetupState::RUNNING) {
            Serial.println("[SetupAndRunTest] RUNNING -- setup phase finished successfully.");
            startRuntime();
        }
    }

    runRuntimeStep();
}
