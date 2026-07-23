// Device-side firmware that runs the setup phase AND the running phase, back
// to back, on the same board -- picking up exactly where examples/SetupTest.ino
// stops ("hand cfg to your normal NNNode / NNScheduler runtime code -- this
// example stops here"). Flash this SAME sketch onto every physical device in
// the network (edit NN_HARDWARE_ID to a DIFFERENT value per board first) --
// genericity comes entirely from what the laptop provisions over the wire via
// tools/nn_setup/setup_tool.py, not from per-board firmware differences. A
// device with predecessorMask==0 (a real physical input-layer node) is the
// one place this sketch branches: everyone else runs the identical path.
//
// Setup phase: plain UDP broadcast (NNTransportUDP), exactly like SetupTest.ino
// -- run tools/nn_setup/setup_tool.py from a laptop on the same WiFi network.
// Running phase: UDP multicast (NNTransportUDPMulticast), joined directly to
// this device's own real layer group (239.1.0.<layerId>) the moment setup
// finishes -- the laptop is NOT part of the running-phase network at all.
#include "NNSetupProtocol.h"
#include "NNTransportUDP.h"
#include "NNTransportUDPMulticast.h"
#include "NNNode.h"
#include "NNScheduler.h"

// ---------------------------------------------------------------------
// EDIT THESE before flashing:
// ---------------------------------------------------------------------
const char* WIFI_SSID     = "your-wifi-ssid";
const char* WIFI_PASSWORD = "your-wifi-password";
const char* BROADCAST_ADDR = "192.168.1.255";  // your subnet's broadcast address
const uint16_t SETUP_PORT  = 4210;             // must match setup_tool.py's --port
const uint16_t RUN_PORT    = 4211;             // running-phase multicast port

// MUST be unique per physical device -- this is how the laptop tool and this
// device recognize which manifest entry belongs to which board. Pick a
// different constant for each device you flash, e.g.
// 0x0000000000000001ULL, 0x0000000000000002ULL, ...
const uint64_t NN_HARDWARE_ID = 0x0000000000000001ULL;
// ---------------------------------------------------------------------

// DATA packets have no ACK/retry layer (NNPacket.h's NNPacketType::ACK is
// explicitly "reserved for a future reliability layer", never implemented),
// and WiFi multicast/broadcast frames get no MAC-layer ACK+retry the way
// unicast frames do -- a single send has a real chance of just being lost,
// and (separately) a freshly-RUNNING sibling may still be mid-join on its
// multicast group when the first send goes out. Resend the identical packet
// a few times with a short gap to cover both. Safe to duplicate:
// NNInputBuffer::storeInput() / NNScheduler::onPacketObserved() are both
// idempotent for a repeated identical packet.
const uint8_t NN_DATA_RETRANSMITS = 3;          // matches setup_tool.py's own START-x3 precedent
const uint16_t NN_DATA_RETRANSMIT_GAP_MS = 150;

NNTransportUDP setupTransport(WIFI_SSID, WIFI_PASSWORD, BROADCAST_ADDR, SETUP_PORT);
NNVolatileConfigStore store;  // no persistence yet -- see NNSetupProtocol.h's own note on this
NNSetupAgent agent(NN_HARDWARE_ID, setupTransport, store);

NNTransportUDPMulticast* runTransport = nullptr;
NNNode* node = nullptr;
NNScheduler* scheduler = nullptr;

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

// Called exactly once, the moment agent.getState() first reports RUNNING --
// switches from the setup-phase broadcast transport to this device's own
// real-layer multicast group and constructs the running-phase objects.
void enterRunningPhase() {
    const NNNodeConfig& cfg = agent.getNodeConfig();

    Serial.print("[RunningNode] entering RUNNING -- layer=");
    Serial.print(cfg.address.layerId);
    Serial.print(" node=");
    Serial.println(cfg.address.nodeId);

    runTransport = new NNTransportUDPMulticast(WIFI_SSID, WIFI_PASSWORD, cfg.address.layerId, RUN_PORT);
    if (!runTransport->begin()) {
        Serial.println("[RunningNode] FAILED to join running-phase multicast group -- halting.");
        while (true) delay(1000);
    }

    node = new NNNode(cfg);
    scheduler = new NNScheduler(*node, agent.getWindowConfig());

    // The one place this sketch branches per device: a predecessorMask==0
    // node (a real physical input-layer device) has nothing to wait for --
    // seed it from the value the laptop pushed during setup (INPUT_VALUE,
    // NOT bias -- see NNSetupProtocol.h) and hand the scheduler's state
    // machine straight into normal turn-taking. Every other device just
    // falls through to the identical receive/tick/transmit loop below.
    if (cfg.predecessorMask == 0) {
        Serial.print("[RunningNode] predecessorMask==0 -- seeding input value ");
        Serial.println(agent.getInputValue(), 6);
        node->seedOutput(agent.getInputValue());
        scheduler->notifySeeded();
    }
}

void setup() {
    Serial.begin(115200);
    delay(1500);

    Serial.print("[RunningNode] hardwareId = 0x");
    Serial.println((uint32_t)(NN_HARDWARE_ID >> 32), HEX);  // Wio Terminal's Serial has no 64-bit HEX print
    Serial.println((uint32_t)(NN_HARDWARE_ID & 0xFFFFFFFF), HEX);

    if (!setupTransport.begin()) {
        Serial.println("[RunningNode] WiFi connect failed -- halting.");
        while (true) delay(1000);
    }

    agent.begin();
    Serial.print("[RunningNode] initial state: ");
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
            Serial.print("[RunningNode] state -> ");
            Serial.println(stateName(current));
            lastPrintedState = current;

            if (current == NNSetupState::RUNNING) {
                enterRunningPhase();
            }
        }
        return;
    }

    // Running phase -- identical loop body for every device.
    NNPacket pkt{};
    while (runTransport->receive(pkt)) {
        if (pkt.header.type == NNPacketType::DATA) {
            node->onPacketReceived(pkt);
            scheduler->onPacketObserved(pkt);
        }
    }

    scheduler->tick();

    NNPacket outPkt{};
    if (scheduler->hasOutputReady(outPkt)) {
        // See NN_DATA_RETRANSMITS' comment above -- blocking here briefly is
        // fine: this node has already computed+transmitted for this pass, and
        // with no backup role configured in this sketch there's nothing else
        // it needs to do (e.g. observe siblings) while the resends go out.
        for (uint8_t i = 0; i < NN_DATA_RETRANSMITS; i++) {
            runTransport->send(outPkt);
            if (i + 1 < NN_DATA_RETRANSMITS) delay(NN_DATA_RETRANSMIT_GAP_MS);
        }
        Serial.print("[RunningNode] transmitted output (x");
        Serial.print(NN_DATA_RETRANSMITS);
        Serial.print("): ");
        Serial.println(outPkt.payload[0], 6);
    }
}
