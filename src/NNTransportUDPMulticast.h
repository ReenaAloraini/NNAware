// NNTransportUDPMulticast.h
// ---------------------------------------------------------------
// A multicast variant of NNTransportUDP.
//
// Layer-addressed multicast: every NN layer L owns the group
// 239.1.0.L. A node JOINS its own layer's group (so it receives that
// layer's inputs) and SENDS to the group of pkt.header.targetLayerId.
//
// Because only that layer's nodes joined the group, the network does
// the filtering for us -- a node never even sees traffic addressed to
// another layer, so no software discard check is needed for real inputs.
//
// PATCHED: sibling turn-taking (NNNodeConfig's precedingSiblingsMask, via
// NNScheduler::onPacketObserved()) needs a device to see its OWN SIBLINGS'
// transmissions too -- but siblings in the SAME layer send their own
// outputs to the NEXT layer's group (their successorLayerId), never to
// their own layer's group. Under the original single-group design, a
// device could never observe its siblings at all, so any node with a
// nonzero precedingSiblingsMask would sit in WAITING_FOR_TURN forever.
// Fixed by optionally joining a SECOND group -- this device's own
// successorLayerId's group -- purely to OBSERVE (not consume as real
// input) its own and its siblings' transmissions. Safe even if a stray
// packet from that second group reaches onPacketReceived(): that
// function's own predecessorLayerId check already rejects anything not
// actually from this node's real predecessor layer (see NNNode.h), so no
// extra filtering is needed here. A device with no siblings to wait for
// (precedingSiblingsMask == 0 -- e.g. the only neuron in its layer) can
// simply omit the second group by passing NN_NO_SIBLING_GROUP (the
// default), avoiding the extra socket entirely.
//
// Implements the same NNTransport interface as NNTransportUDP and
// NNTransportLoopback, so it is a drop-in replacement.
// ---------------------------------------------------------------
#pragma once
#include "NNTransport.h"
#include "NNPacket.h"

#if defined(ARDUINO_ARCH_ESP32) && !defined(NN_USE_RPC_WIFI)
  #include <WiFi.h>
#else
  #include <rpcWiFi.h>
#endif
#include <WiFiUdp.h>

constexpr uint8_t NN_NO_SIBLING_GROUP = 255;  // pass this (the default) to skip joining a second group

class NNTransportUDPMulticast : public NNTransport {
public:
    // joinLayerId: the layer whose group this node listens on (its OWN layer) --
    // where its real predecessor inputs arrive.
    // siblingGroupLayerId: OPTIONAL second group to ALSO join, purely to observe
    // sibling transmissions for NNScheduler's turn-taking -- pass this node's own
    // successorLayerId (siblings send their output there, not to their own
    // layer's group). Leave at the default (NN_NO_SIBLING_GROUP) if this node has
    // no siblings to wait for (precedingSiblingsMask == 0).
    NNTransportUDPMulticast(const char* ssid, const char* password,
                            uint8_t joinLayerId, uint16_t port,
                            uint8_t siblingGroupLayerId = NN_NO_SIBLING_GROUP)
        : ssid(ssid), password(password),
          joinLayerId(joinLayerId), port(port),
          siblingGroupLayerId(siblingGroupLayerId) {}

    // One multicast group per layer: 239.1.0.<layer>
    static IPAddress layerGroup(uint8_t layer) {
        return IPAddress(239, 1, 0, layer);
    }

    bool hasSiblingGroup() const { return siblingGroupLayerId != NN_NO_SIBLING_GROUP; }

    bool begin() override {
        WiFi.mode(WIFI_STA);
        WiFi.disconnect();
        delay(500);

        Serial.print("[NNTransportUDPMulticast] connecting to \"");
        Serial.print(ssid);
        Serial.println("\"...");
        WiFi.begin(ssid, password);

        unsigned long start = millis();
        while (WiFi.status() != WL_CONNECTED) {
            delay(500);
            Serial.print(".");
            if (millis() - start > 15000) {
                Serial.println();
                Serial.println("[NNTransportUDPMulticast] WiFi connect TIMEOUT -- check "
                               "ssid/password and that the router is in range.");
                return false;
            }
        }
        Serial.println();
        Serial.print("[NNTransportUDPMulticast] connected. IP: ");
        Serial.println(WiFi.localIP());

        // Join this node's own layer group so its real inputs are delivered here.
        IPAddress group = layerGroup(joinLayerId);
        uint8_t ok = udp.beginMulticast(group, port);
        Serial.print("[NNTransportUDPMulticast] joined group ");
        Serial.print(group);
        Serial.println(ok ? "  [ok]" : "  [JOIN FAILED]");

        // Optionally join a second group, purely to observe siblings.
        bool siblingOk = true;
        if (hasSiblingGroup()) {
            IPAddress sibGroup = layerGroup(siblingGroupLayerId);
            siblingOk = udpSiblings.beginMulticast(sibGroup, port) != 0;
            Serial.print("[NNTransportUDPMulticast] joined sibling-observation group ");
            Serial.print(sibGroup);
            Serial.println(siblingOk ? "  [ok]" : "  [JOIN FAILED]");
        }

        return (ok != 0) && siblingOk;
    }

    // Sends to the group of the packet's targetLayerId -- the destination
    // group address is what encodes the target layer on the wire.
    bool send(const NNPacket& pkt) override {
        uint8_t buffer[64];
        uint16_t len = serializePacket(pkt, buffer, sizeof(buffer));
        if (len == 0) return false;

        udp.beginPacket(layerGroup(pkt.header.targetLayerId), port);
        udp.write(buffer, len);
        bool ok = udp.endPacket();   // returns 1 on success per WiFiUDP convention

        if (!ok) {
            Serial.println("[NNTransportUDPMulticast] send() FAILED at endPacket()");
        }
        return ok;
    }

    // Checks the own-layer group first (real inputs), then the sibling-
    // observation group if one was joined -- both feed the SAME receive()
    // interface, so callers still route every packet through
    // onPacketReceived() AND onPacketObserved() exactly as before.
    bool receive(NNPacket& outPkt) override {
        return tryReceiveFrom(udp, outPkt)
            || (hasSiblingGroup() && tryReceiveFrom(udpSiblings, outPkt));
    }

    void poll() override {}

private:
    // One non-blocking drain attempt against a single socket. Shared by both
    // groups so the buffer size and the read/deserialize validation live in
    // exactly one place.
    static bool tryReceiveFrom(WiFiUDP& sock, NNPacket& outPkt) {
        if (sock.parsePacket() <= 0) return false;   // non-blocking
        uint8_t buffer[64];
        int n = sock.read(buffer, sizeof(buffer));
        return n > 0 && deserializePacket(buffer, static_cast<uint16_t>(n), outPkt);
    }

    const char* ssid;
    const char* password;
    uint8_t     joinLayerId;
    uint16_t    port;
    uint8_t     siblingGroupLayerId;
    WiFiUDP     udp;
    WiFiUDP     udpSiblings;
};
