#include"NNAddress.h"
#include"NNPacket.h"

int testsPassed = 0;
int testsFailed = 0;

#define NN_TEST_ASSERT(cond,msg)do{\
if(cond){Serial.print("PASS: ");Serial.println(msg);testsPassed++;}\
else{Serial.print("FAIL: ");Serial.println(msg);testsFailed++;}\
}while(0)

void printSummary() {
    Serial.println("----------------------------------");
    Serial.print("Tests passed: "); Serial.println(testsPassed);
    Serial.print("Tests failed: "); Serial.println(testsFailed);
    Serial.println(testsFailed == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
}

void setup() {
    Serial.begin(115200);
    delay(1500);

    Serial.println("=== Milestone 2: NNPacket serialize/deserialize ===");

    NNPacket original{};
    original.header.sourceAddress = encodeAddress({1, 1, 0, 0});
    original.header.targetLayerId = 2;
    original.header.type          = NNPacketType::DATA;
    original.header.sequenceNumber = 5;
    original.header.payloadCount  = 3;
    original.payload[0] = 1.0f;
    original.payload[1] = 2.0f;
    original.payload[2] = 3.0f;

    uint8_t buffer[64];
    uint16_t written = serializePacket(original, buffer, sizeof(buffer));
    NN_TEST_ASSERT(written > 0, "serializePacket returns nonzero length");

    NNPacket recovered{};
    bool ok = deserializePacket(buffer, written, recovered);
    NN_TEST_ASSERT(ok, "deserializePacket succeeds on an unmodified buffer");
    NN_TEST_ASSERT(recovered.header.sourceAddress == original.header.sourceAddress,
                   "recovered sourceAddress matches original");
    NN_TEST_ASSERT(recovered.header.payloadCount == 3, "recovered payloadCount == 3");
    NN_TEST_ASSERT(recovered.payload[1] == 2.0f, "recovered payload[1] == 2.0");

    // Corruption test — flip a bit inside the payload region and confirm it's caught
    buffer[sizeof(NNPacketHeader) + 1] ^= 0xFF;
    NNPacket corrupted{};
    bool shouldFail = deserializePacket(buffer, written, corrupted);
    NN_TEST_ASSERT(!shouldFail, "deserializePacket correctly rejects a corrupted buffer");

    printSummary();
}

void loop() { }