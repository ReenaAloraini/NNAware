// assigning an address to each node; Address = {nodeID, layerID, clusterID, reserved}
#pragma once
#include <stdint.h>

// A decoded, human-friendly view of a 16-bit node address.
// This struct never touches the network directly — it's a pure data model.
struct NNAddress {
    uint8_t nodeId;     // 0-15  : which physical node within its layer/cluster
    uint8_t layerId;    // 0-15  : which NN layer this node belongs to
    uint8_t clusterId;  // 0-15  : which cluster within the layer (for parallel groups)
    uint8_t reserved;   // 0-15  : unused for now — reserved for future protocol versions
};

// Packs a decoded NNAddress into the 16-bit wire format used in every NNPacket header.
inline uint16_t encodeAddress(const NNAddress& addr) {
    return  (uint16_t(addr.nodeId    & 0x0F) << 12) |
            (uint16_t(addr.layerId   & 0x0F) << 8)  |
            (uint16_t(addr.clusterId & 0x0F) << 4)  |
            (uint16_t(addr.reserved  & 0x0F));
}

// Unpacks a raw 16-bit wire value into a human-readable NNAddress (hex)
inline NNAddress decodeAddress(uint16_t raw) {
    NNAddress addr;
    addr.nodeId    = (raw >> 12) & 0x0F;
    addr.layerId   = (raw >> 8)  & 0x0F;
    addr.clusterId = (raw >> 4)  & 0x0F;
    addr.reserved  =  raw        & 0x0F;
    return addr;
}