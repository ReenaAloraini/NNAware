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
// another layer, so no software discard check is needed.
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

class NNTransportUDPMulticast : public NNTransport {
public:
    // joinLayerId: the layer whose group this node listens on (its OWN layer)
    NNTransportUDPMulticast(const char* ssid, const char* password,
                            uint8_t joinLayerId, uint16_t port)
        : ssid(ssid), password(password),
          joinLayerId(joinLayerId), port(port) {}

    // One multicast group per layer: 239.1.0.<layer>
    static IPAddress layerGroup(uint8_t layer) {
        return IPAddress(239, 1, 0, layer);
    }

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

        // Join this node's own layer group so its inputs are delivered here.
        IPAddress group = layerGroup(joinLayerId);
        uint8_t ok = udp.beginMulticast(group, port);

        Serial.print("[NNTransportUDPMulticast] joined group ");
        Serial.print(group);
        Serial.println(ok ? "  [ok]" : "  [JOIN FAILED]");
        return ok != 0;
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

    bool receive(NNPacket& outPkt) override {
        int packetSize = udp.parsePacket();   // non-blocking
        if (packetSize <= 0) return false;

        uint8_t buffer[64];
        int n = udp.read(buffer, sizeof(buffer));
        if (n <= 0) return false;

        return deserializePacket(buffer, static_cast<uint16_t>(n), outPkt);
    }

    void poll() override {}

private:
    const char* ssid;
    const char* password;
    uint8_t     joinLayerId;
    uint16_t    port;
    WiFiUDP     udp;
};
