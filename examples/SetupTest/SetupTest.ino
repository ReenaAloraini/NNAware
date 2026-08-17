
#include "NNSetupProtocol.h"
#include "NNTransportUDP.h"


// EDIT THESE before flashing:
const char* WIFI_SSID     = "";
const char* WIFI_PASSWORD = "";
const char* BROADCAST_ADDR = "192.168.1.255";  
const uint16_t SETUP_PORT  = 4210;             


const uint64_t NN_HARDWARE_ID = 0x1122334455667788ULL;


NNTransportUDP transport(WIFI_SSID, WIFI_PASSWORD, BROADCAST_ADDR, SETUP_PORT);
NNVolatileConfigStore store;  
NNSetupAgent agent(NN_HARDWARE_ID, transport, store);

NNSetupState lastPrintedState = NNSetupState::LOADING;

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
    Serial.println("From here, hand cfg / agent.getWindowConfig() to your normal");
    Serial.println("NNNode / NNScheduler runtime code -- this example stops here.");
}

void setup() {
    Serial.begin(115200);
    delay(1500);

    Serial.print("[SetupTest] hardwareId = 0x");
    Serial.println((uint32_t)(NN_HARDWARE_ID >> 32), HEX);  // Wio Terminal's Serial has no 64-bit HEX print
    Serial.println((uint32_t)(NN_HARDWARE_ID & 0xFFFFFFFF), HEX);

    if (!transport.begin()) {
        Serial.println("[SetupTest] WiFi connect failed -- halting.");
        while (true) delay(1000);
    }

    agent.begin();
    Serial.print("[SetupTest] initial state: ");
    Serial.println(stateName(agent.getState()));
    lastPrintedState = agent.getState();
}

void loop() {
    agent.tick(millis());

    NNPacket pkt{};
    while (transport.receive(pkt)) {
        if (pkt.header.type == NNPacketType::CONTROL) {
            agent.onSetupPacket(pkt);
        }
    }

    NNSetupState current = agent.getState();
    if (current != lastPrintedState) {
        Serial.print("[SetupTest] state -> ");
        Serial.println(stateName(current));
        lastPrintedState = current;

        if (current == NNSetupState::CONFIGURED) {
            printNodeConfig();
        }
        if (current == NNSetupState::RUNNING) {
            Serial.println("[SetupTest] RUNNING -- setup phase finished successfully.");
        }
    }
}
