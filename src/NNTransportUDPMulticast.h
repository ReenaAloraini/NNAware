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

        IPAddress group = layerGroup(joinLayerId);
        uint8_t ok = udp.beginMulticast(group, port);
        Serial.print("[NNTransportUDPMulticast] joined group ");
        Serial.print(group);
        Serial.println(ok ? "  [ok]" : "  [JOIN FAILED]");


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

   
    bool send(const NNPacket& pkt) override {
        uint8_t buffer[64];
        uint16_t len = serializePacket(pkt, buffer, sizeof(buffer));
        if (len == 0) return false;

        udp.beginPacket(layerGroup(pkt.header.targetLayerId), port);
        udp.write(buffer, len);
        bool ok = udp.endPacket();   

        if (!ok) {
            Serial.println("[NNTransportUDPMulticast] send() FAILED at endPacket()");
        }
        return ok;
    }

    bool receive(NNPacket& outPkt) override {
        return tryReceiveFrom(udp, outPkt)
            || (hasSiblingGroup() && tryReceiveFrom(udpSiblings, outPkt));
    }

    void poll() override {}

private:
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
