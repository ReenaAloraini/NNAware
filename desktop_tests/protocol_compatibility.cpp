// A thin CLI/stdin-stdout wrapper around the NNAware library
// functions -- serializePacket(), deserializePacket(), computeChecksum(),
// encodeAddress(), decodeAddress() -- all called directly from the actual
// headers, not reimplemented. This file contains zero protocol logic of
// its own; its only job is JSON in, real library call, JSON out.

// Usage:
//   ./protocol_compatibility <command>
//   Reads one JSON object from stdin, writes one JSON object to stdout, exits.

// C:\Users\Reena\OneDrive\Desktop\NN_Aware_Project\NNAware\desktop_tests\protocol_compatibility.cpp
// Commands: serialize | deserialize | checksum | encode-address | decode-address

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <cstdint>

#include <nlohmann/json.hpp>

#include "NNAddress.h"
#include "NNPacket.h"

using json = nlohmann::json;

//hex <-> bytes helpers 

static std::string bytesToHex(const uint8_t* data, size_t len) {
    std::ostringstream oss;
    for (size_t i = 0; i < len; i++) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(data[i]);
    }
    return oss.str();
}

static std::vector<uint8_t> hexToBytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i + 1 < hex.size() + 1 && i + 2 <= hex.size(); i += 2) {
        bytes.push_back(static_cast<uint8_t>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    }
    return bytes;
}

//command handlers 

static json handleEncodeAddress(const json& in) {
    NNAddress addr;
    addr.nodeId    = in.at("node_id").get<uint8_t>();
    addr.layerId   = in.at("layer_id").get<uint8_t>();
    addr.clusterId = in.value("cluster_id", 0);
    addr.reserved  = in.value("reserved", 0);

    uint16_t encoded = encodeAddress(addr);  // library call

    json out;
    out["ok"] = true;
    out["encoded"] = encoded;
    return out;
}

static json handleDecodeAddress(const json& in) {
    uint16_t encoded = in.at("encoded").get<uint16_t>();
    NNAddress addr = decodeAddress(encoded);  // REAL library call

    json out;
    out["ok"] = true;
    out["node_id"] = addr.nodeId;
    out["layer_id"] = addr.layerId;
    out["cluster_id"] = addr.clusterId;
    out["reserved"] = addr.reserved;
    return out;
}

static json handleChecksum(const json& in) {
    std::vector<uint8_t> bytes = hexToBytes(in.at("bytes_hex").get<std::string>());
    uint8_t cs = computeChecksum(bytes.data(), static_cast<uint16_t>(bytes.size()));  // REAL library call

    json out;
    out["ok"] = true;
    out["checksum"] = cs;
    return out;
}

static json handleSerialize(const json& in) {
    NNPacket pkt{};
    pkt.header.sourceAddress  = in.at("source_address").get<uint16_t>();
    pkt.header.targetLayerId  = in.at("target_layer_id").get<uint8_t>();
    pkt.header.type           = static_cast<NNPacketType>(in.at("type").get<int>());
    pkt.header.sequenceNumber = in.at("sequence").get<uint8_t>();
    pkt.header.flags          = in.value("flags", 0);

    auto values = in.at("values").get<std::vector<float>>();
    pkt.header.payloadCount = static_cast<uint8_t>(values.size());
    for (size_t i = 0; i < values.size() && i < NN_MAX_PAYLOAD_FLOATS; i++) {
        pkt.payload[i] = values[i];
    }

    uint8_t buffer[512];
    uint16_t written = serializePacket(pkt, buffer, sizeof(buffer));  // REAL library call

    json out;
    out["ok"] = (written > 0);
    if (written > 0) {
        out["bytes_hex"] = bytesToHex(buffer, written);
        out["length"] = written;
    } else {
        out["error"] = "serializePacket() returned 0 (buffer too small, or payloadCount exceeds NN_MAX_PAYLOAD_FLOATS)";
    }
    return out;
}

static json handleDeserialize(const json& in) {
    std::vector<uint8_t> bytes = hexToBytes(in.at("bytes_hex").get<std::string>());

    NNPacket pkt{};
    bool ok = deserializePacket(bytes.data(), static_cast<uint16_t>(bytes.size()), pkt);  // REAL library call

    json out;
    out["ok"] = ok;
    if (ok) {
        out["source_address"] = pkt.header.sourceAddress;
        out["target_layer_id"] = pkt.header.targetLayerId;
        out["type"] = static_cast<int>(pkt.header.type);
        out["sequence"] = pkt.header.sequenceNumber;
        out["payload_count"] = pkt.header.payloadCount;
        out["flags"] = pkt.header.flags;
        out["checksum"] = pkt.header.checksum;
        out["values"] = std::vector<float>(pkt.payload, pkt.payload + pkt.header.payloadCount);
    } else {
        out["error"] = "deserializePacket() rejected the input (checksum, length, or payloadCount validation failed)";
    }
    return out;
}

//entry point 

int main(int argc, char** argv) {
    static_assert(sizeof(NNPacketHeader) == 8,
                  "NNPacketHeader size drifted from the confirmed 8-byte layout -- "
                  "every offset assumption in the Python test suite depends on this.");

    if (argc < 2) {
        std::cerr << "Usage: protocol_compatibility <command>\n"
                     "Commands: serialize | deserialize | checksum | encode-address | decode-address\n"
                     "Reads one JSON object from stdin, writes one JSON object to stdout.\n";
        return 1;
    }
    const std::string command = argv[1];

    std::ostringstream stdinBuffer;
    stdinBuffer << std::cin.rdbuf();

    json in;
    try {
        in = json::parse(stdinBuffer.str());
    } catch (const std::exception& e) {
        json err;
        err["ok"] = false;
        err["error"] = std::string("JSON parse error: ") + e.what();
        std::cout << err.dump() << std::endl;
        return 1;
    }

    json out;
    try {
        if (command == "encode-address")      out = handleEncodeAddress(in);
        else if (command == "decode-address") out = handleDecodeAddress(in);
        else if (command == "checksum")       out = handleChecksum(in);
        else if (command == "serialize")      out = handleSerialize(in);
        else if (command == "deserialize")    out = handleDeserialize(in);
        else {
            out["ok"] = false;
            out["error"] = "unknown command: " + command;
            std::cout << out.dump() << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        out["ok"] = false;
        out["error"] = std::string("exception: ") + e.what();
        std::cout << out.dump() << std::endl;
        return 1;
    }

    std::cout << out.dump() << std::endl;
    return 0;
}
