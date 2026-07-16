// experimenting with this module was not successful, will keep it in the library in case we decided to debug again 
#pragma once
#include "NNTransport.h"
#include "NNBuffer.h"
#include "NNPacket.h"

#include <rpcBLEDevice.h>
#include <BLEServer.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>



constexpr uint16_t NN_BLE_MFG_DATA_CAPACITY = 27;
constexpr uint32_t NN_BLE_SCAN_DURATION_SEC = 3;
constexpr unsigned long NN_BLE_SCAN_RESTART_INTERVAL_MS = 5000;

class NNTransportBLE : public NNTransport {
public:
    explicit NNTransportBLE(const char* deviceName, unsigned long scanPhaseOffsetMs = 0)
        : deviceName(deviceName), scanPhaseOffsetMs(scanPhaseOffsetMs) {
        instance = this; 
    }

    bool begin() override {
        BLEDevice::init(deviceName);
        Serial.print("[NNTransportBLE] initialized as \"");
        Serial.print(deviceName);
        Serial.println("\" — search for this exact name in nRF Connect if you use it.");

        BLEDevice::createServer();

        // Advertising side 
        advertising = BLEDevice::getAdvertising();
        advertising->setScanResponse(false);

        // Scanning side 
        scanner = BLEDevice::getScan();
        scanner->setAdvertisedDeviceCallbacks(new NNAdvertisedDeviceCallback());
        scanner->setInterval(1349);   
        scanner->setWindow(449);
        scanner->setActiveScan(true);

        lastScanRestart = millis() - NN_BLE_SCAN_RESTART_INTERVAL_MS + scanPhaseOffsetMs;

        return true;
    }

    bool send(const NNPacket& pkt) override {
        uint8_t buffer[64];
        uint16_t len = serializePacket(pkt, buffer, sizeof(buffer));
        if (len == 0) return false;

        if (len > NN_BLE_MFG_DATA_CAPACITY) {
            Serial.println("[NNTransportBLE] send() REFUSED — packet exceeds the confirmed 27-byte cap.");
            return false;
        }

        BLEAdvertisementData advData;
        advData.setFlags(0x06);  

        std::string raw;
        raw += static_cast<char>(len + 1);  
        raw += static_cast<char>(0xFF);     
        raw += std::string(reinterpret_cast<char*>(buffer), len);
        advData.addData(raw);

        Serial.print("[NNTransportBLE] send() embedding ");
        Serial.print(len);
        Serial.print(" bytes: ");
        for (uint16_t i = 0; i < len; i++) {
            if (buffer[i] < 0x10) Serial.print("0");
            Serial.print(buffer[i], HEX);
            Serial.print(" ");
        }
        Serial.println();

        advertising->stop();
        advertising->setAdvertisementData(advData);
        advertising->start();

        return true;
    }

    bool receive(NNPacket& outPkt) override {
        return incomingQueue.pop(outPkt);
    }

    void poll() override {
        unsigned long now = millis();
        if (now - lastScanRestart < NN_BLE_SCAN_RESTART_INTERVAL_MS) return;
        lastScanRestart = now;

        devicesSeenThisWindow = 0;
        flaggedMfgDataSeenThisWindow = 0;
        successfulDecodesThisWindow = 0;

        Serial.print("[NNTransportBLE] scan starting at t=");
        Serial.println(now);
        scanner->start(NN_BLE_SCAN_DURATION_SEC, false);  
        Serial.print("[NNTransportBLE] scan returned at t=");
        Serial.println(millis());

        Serial.print("[NNTransportBLE] this window: saw ");
        Serial.print(devicesSeenThisWindow);
        Serial.print(" device(s) total, ");
        Serial.print(flaggedMfgDataSeenThisWindow);
        Serial.print(" flagged haveManufacturerData, ");
        Serial.print(successfulDecodesThisWindow);
        Serial.println(" actually decoded successfully.");
    }

private:
    const char* deviceName;
    unsigned long scanPhaseOffsetMs;
    BLEAdvertising* advertising = nullptr;
    BLEScan* scanner = nullptr;
    NNPacketQueue incomingQueue;  
    unsigned long lastScanRestart = 0;
    uint16_t devicesSeenThisWindow = 0;
    uint16_t flaggedMfgDataSeenThisWindow = 0;
    uint16_t successfulDecodesThisWindow = 0;

    static NNTransportBLE* instance;

    class NNAdvertisedDeviceCallback : public BLEAdvertisedDeviceCallbacks {
        void onResult(BLEAdvertisedDevice advertisedDevice) override {
            if (instance == nullptr) return;
            instance->devicesSeenThisWindow++;  

            if (advertisedDevice.haveManufacturerData()) {
                instance->flaggedMfgDataSeenThisWindow++;
            }

            uint8_t* mfgData = advertisedDevice.getManufacturerData();
            if (mfgData == nullptr) return;

            NNPacket pkt{};
            if (deserializePacket(mfgData, NN_BLE_MFG_DATA_CAPACITY, pkt)) {
                instance->successfulDecodesThisWindow++;
                instance->incomingQueue.push(pkt);

                Serial.print("[NNTransportBLE] DECODED PACKET from ");
                Serial.print(advertisedDevice.getAddress().toString().c_str());
                Serial.print(": ");
                for (uint16_t i = 0; i < NN_BLE_MFG_DATA_CAPACITY; i++) {
                    if (mfgData[i] < 0x10) Serial.print("0");
                    Serial.print(mfgData[i], HEX);
                    Serial.print(" ");
                }
                Serial.println();
            }
           
        }
    };
};

NNTransportBLE* NNTransportBLE::instance = nullptr;
