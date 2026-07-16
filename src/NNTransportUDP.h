#pragma once
#include "NNTransport.h"
#include "NNPacket.h"

#include <rpcWiFi.h>
#include <WiFiUdp.h>

class NNTransportUDP : public NNTransport {
public:
    NNTransportUDP(const char* ssid, const char* password,
                   const char* broadcastAddr, uint16_t port)
        : ssid(ssid), password(password), broadcastAddr(broadcastAddr), port(port) {}

    bool begin() override {
        WiFi.mode(WIFI_STA);
        WiFi.disconnect();
        delay(500);

        Serial.print("[NNTransportUDP] connecting to \"");
        Serial.print(ssid);
        Serial.println("\"...");
        WiFi.begin(ssid, password);

        unsigned long start = millis();
        while (WiFi.status() != WL_CONNECTED) {
            delay(500);
            Serial.print(".");
            if (millis() - start > 15000) {
                Serial.println();
                Serial.println("[NNTransportUDP] WiFi connect TIMEOUT — check ssid/password"
                                " and that the router is in range.");
                return false;
            }
        }
        Serial.println();
        Serial.print("[NNTransportUDP] connected. IP: ");
        Serial.println(WiFi.localIP());

        udp.begin(port);
        return true;
    }

    bool send(const NNPacket& pkt) override {
        uint8_t buffer[64];
        uint16_t len = serializePacket(pkt, buffer, sizeof(buffer));
        if (len == 0) return false;

        udp.beginPacket(broadcastAddr, port);
        udp.write(buffer, len);
        bool ok = udp.endPacket();  // returns 1 on success per WiFiUDP convention

        if (!ok) {
            Serial.println("[NNTransportUDP] send() FAILED at endPacket()");
        }
        return ok;
    }

    bool receive(NNPacket& outPkt) override {
        int packetSize = udp.parsePacket();  // non-blocking, returns 0 immediately if nothing waiting
        if (packetSize <= 0) return false;

        uint8_t buffer[64];
        int n = udp.read(buffer, sizeof(buffer));
        if (n <= 0) return false;

        return deserializePacket(buffer, static_cast<uint16_t>(n), outPkt);
        
    }

    void poll() override {
    }

private:
    const char* ssid;
    const char* password;
    const char* broadcastAddr;
    uint16_t port;
    WiFiUDP udp;
};
