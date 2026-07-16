#include<cassert>
#include"NNBuffer.h"

int main() {
    // Queue test
    NNPacketQueue q;
    NNPacket p1{}, p2{};
    assert(q.push(p1));
    assert(q.push(p2));
    assert(q.size() == 2);
    NNPacket out;
    assert(q.pop(out));
    assert(q.size() == 1);

    // Input buffer test — simulating the Layer1(A,B,C) -> Layer2(D) example
    NNInputBuffer buf;
    uint16_t expected = 0b0000000000000111; // nodes 0 (A), 1 (B), 2 (C)
    float a[1] = {0.5f}, b[1] = {0.25f}, c[1] = {0.1f};

    assert(!buf.isComplete(expected));   // nothing received yet
    buf.storeInput(0, a, 1);
    assert(!buf.isComplete(expected));   // only A so far
    buf.storeInput(1, b, 1);
    assert(!buf.isComplete(expected));   // A and B so far
    buf.storeInput(2, c, 1);
    assert(buf.isComplete(expected));    // now complete — matches the project brief's example exactly

    buf.reset();
    assert(!buf.isComplete(expected));   // must be false again after reset

    return 0;
}