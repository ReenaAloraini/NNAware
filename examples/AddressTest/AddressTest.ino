#include"NNAddress.h"

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
    delay(1500);  // give you time to open the Serial Monitor before output starts

    Serial.println("=== Milestone 1: NNAddress encode/decode ===");

    // Round-trip test
    NNAddress a{3, 2, 1, 0};   // Node 3, Layer 2, Cluster 1, Reserved 0
    uint16_t packed = encodeAddress(a);
    NN_TEST_ASSERT(packed == 0x3210, "encodeAddress(3,2,1,0) == 0x3210");

    NNAddress b = decodeAddress(0x3210);
    NN_TEST_ASSERT(b.nodeId == 3, "decodeAddress nodeId == 3");
    NN_TEST_ASSERT(b.layerId == 2, "decodeAddress layerId == 2");
    NN_TEST_ASSERT(b.clusterId == 1, "decodeAddress clusterId == 1");
    NN_TEST_ASSERT(b.reserved == 0, "decodeAddress reserved == 0");

    // Overflow/masking protection test
    NNAddress overflowed{200, 2, 1, 0};  // nodeId far out of 4-bit range
    uint16_t packedOverflow = encodeAddress(overflowed);
    NN_TEST_ASSERT(((packedOverflow >> 12) & 0x0F) == (200 & 0x0F),
                   "overflowed nodeId is masked, not corrupting other fields");

    // Manual exercise answer check (Milestone 1, Exercise 1)
    NNAddress exercise = decodeAddress(0x4A50);
    NN_TEST_ASSERT(exercise.nodeId == 4 && exercise.layerId == 10 &&
                   exercise.clusterId == 5 && exercise.reserved == 0,
                   "0x4A50 decodes to Node=4, Layer=10(A), Cluster=5, Reserved=0");

    printSummary();
}

void loop() { /* nothing — this is a one-shot test sketch */ }