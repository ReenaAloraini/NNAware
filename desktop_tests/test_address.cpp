#include <cassert>
#include <iostream>
#include "NNAddress.h"

int main() {
    NNAddress a{3, 2, 1, 0};              // Node 3, Layer 2, Cluster 1, Reserved 0
    uint16_t packed = encodeAddress(a);
    assert(packed == 0x3210);             // manually verified by hand first!

    NNAddress b = decodeAddress(0x3210);
    assert(b.nodeId == 3 && b.layerId == 2 && b.clusterId == 1 && b.reserved == 0);

    // Edge case: overflow protection
    NNAddress overflowed{200, 2, 1, 0};    // nodeId way out of 4-bit range
    uint16_t packedOverflow = encodeAddress(overflowed);
    assert(((packedOverflow >> 12) & 0x0F) == (200 & 0x0F)); // masked, not corrupted
    std::cout << "All address tests passed!\n";
    
    return 0;
}