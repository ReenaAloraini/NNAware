#pragma once
#include "NNTransport.h"
#include "NNBuffer.h"
#include "NNPacket.h"
#include "NNAddress.h"

#include <rpcWiFi.h>
#include <PubSubClient.h>

constexpr unsigned long NN_MQTT_RECONNECT_INTERVAL_MS = 5000;

class NNTransportMQTT : public NNTransport {
public:
    NNTransportMQTT(const char* ssid, const char* password,
                     const char* brokerAddress, uint16_t brokerPort,
                     const char* clientId)
        : ssid(ssid), password(password), brokerAddress(brokerAddress),
          brokerPort(brokerPort), clientId(clientId), mqttClient(wifiClient) {
        instance = this;  
    }

    bool begin() override {
        WiFi.mode(WIFI_STA);
        WiFi.disconnect();
        delay(500);

        Serial.print("[NNTransportMQTT] connecting to WiFi \"");
        Serial.print(ssid);
        Serial.println("\"...");
        WiFi.begin(ssid, password);

        unsigned long start = millis();
        while (WiFi.status() != WL_CONNECTED) {
            delay(500);
            Serial.print(".");
            if (millis() - start > 15000) {
                Serial.println();
                Serial.println("[NNTransportMQTT] WiFi connect TIMEOUT.");
                return false;
            }
        }
        Serial.println();
        Serial.print("[NNTransportMQTT] WiFi connected. IP: ");
        Serial.println(WiFi.localIP());

        mqttClient.setServer(brokerAddress, brokerPort);
        mqttClient.setCallback(mqttCallback);

        return connectToBroker();
    }

    bool send(const NNPacket& pkt) override {
        if (!mqttClient.connected()) return false;

        uint8_t buffer[64];
        uint16_t len = serializePacket(pkt, buffer, sizeof(buffer));
        if (len == 0) return false;

        char topic[48];
        NNAddress src = decodeAddress(pkt.header.sourceAddress);
        snprintf(topic, sizeof(topic), "nnaware/test/node/%u", src.nodeId);

        
        bool ok = mqttClient.publish(topic, buffer, len, true);
        if (!ok) {
            Serial.println("[NNTransportMQTT] publish() FAILED");
        }
        return ok;
    }

    bool receive(NNPacket& outPkt) override {
        return incomingQueue.pop(outPkt);
    }

    void poll() override {
        if (!mqttClient.connected()) {
            unsigned long now = millis();
            if (now - lastReconnectAttempt >= NN_MQTT_RECONNECT_INTERVAL_MS) {
                lastReconnectAttempt = now;
                Serial.println("[NNTransportMQTT] reconnecting to broker...");
                connectToBroker();
            }
            return;
        }
        mqttClient.loop();  
    }

private:
    const char* ssid;
    const char* password;
    const char* brokerAddress;
    uint16_t brokerPort;
    const char* clientId;
    WiFiClient wifiClient;
    PubSubClient mqttClient;
    NNPacketQueue incomingQueue;  
    unsigned long lastReconnectAttempt = 0;

    static NNTransportMQTT* instance;

    bool connectToBroker() {
        Serial.print("[NNTransportMQTT] connecting to broker \"");
        Serial.print(brokerAddress);
        Serial.print("\" as \"");
        Serial.print(clientId);
        Serial.print("\"... ");

        if (mqttClient.connect(clientId)) {
            Serial.println("connected.");
            mqttClient.subscribe("nnaware/test/node/+");
            return true;
        } else {
            Serial.print("FAILED, state=");
            Serial.println(mqttClient.state());  // negative = connection issue, positive = broker rejected
            return false;
        }
    }

   
    static void mqttCallback(char* topic, byte* payload, unsigned int length) {
        if (instance == nullptr) return;

        NNPacket pkt{};
        if (deserializePacket(payload, static_cast<uint16_t>(length), pkt)) {
            instance->incomingQueue.push(pkt);
        }
        
    }
};

NNTransportMQTT* NNTransportMQTT::instance = nullptr;
