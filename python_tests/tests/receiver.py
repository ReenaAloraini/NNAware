import argparse
import socket
import struct


# WIRE FORMAT 
#   sourceAddress (uint16), targetLayerId (uint8), type (uint8),
#   sequenceNumber (uint8), payloadCount (uint8), flags (uint8), checksum (uint8)
HEADER_FMT = ">HBBBBBB"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 8, matching the C++ struct

PACKET_TYPE_NAMES = {0: "DATA", 1: "CONTROL", 2: "ACK"}  # matches NNPacketType


NN_MAX_PAYLOAD_FLOATS = 16  # must match NNPacket.h's constant exactly
CHECKSUM_INDEX = 7          # checksum's fixed byte offset in this wire format
                             


def decode_address(encoded: int) -> tuple[int, int, int, int]:
    """Mirrors NNAddress.h's decodeAddress() exactly."""
    node_id = (encoded >> 12) & 0x0F
    layer_id = (encoded >> 8) & 0x0F
    cluster_id = (encoded >> 4) & 0x0F
    reserved = encoded & 0x0F
    return node_id, layer_id, cluster_id, reserved


def compute_checksum(buffer: bytes) -> int:
    """
    Mirrors NNPacket.h's computeChecksum(data, length) exactly: an additive
    sum of every byte in the given buffer, truncated to 8 bits
    """
    return sum(buffer) & 0xFF


def deserialize_packet(data: bytes):
    """
    Returns (ok: bool, decoded: dict). Mirrors NNPacket::deserializePacket's
    exact validation order
    """
    if len(data) < HEADER_SIZE:
        return False, {"error": f"buffer too short for header ({len(data)} < {HEADER_SIZE})"}

    received_checksum = data[CHECKSUM_INDEX]
    checksum_input = bytearray(data)
    checksum_input[CHECKSUM_INDEX] = 0
    computed_checksum = compute_checksum(checksum_input)

    if computed_checksum != received_checksum:
        return False, {"error": f"checksum mismatch (got {received_checksum}, expected {computed_checksum})"}

    source_address, target_layer_id, ptype, sequence, payload_count, flags, checksum = \
        struct.unpack(HEADER_FMT, data[:HEADER_SIZE])

    if payload_count > NN_MAX_PAYLOAD_FLOATS:
        return False, {"error": f"payloadCount ({payload_count}) exceeds "
                                 f"NN_MAX_PAYLOAD_FLOATS ({NN_MAX_PAYLOAD_FLOATS})"}

    expected_total = HEADER_SIZE + payload_count * 4  # 4 bytes per float
    if len(data) != expected_total:
        return False, {"error": f"length mismatch: received {len(data)} bytes, expected "
                                 f"exactly {expected_total} for payloadCount={payload_count}"}

    payload_bytes = data[HEADER_SIZE:expected_total]
    values = struct.unpack(f">{payload_count}f", payload_bytes) if payload_count > 0 else ()
    node_id, layer_id, cluster_id, _ = decode_address(source_address)

    return True, {
        "node_id": node_id, "layer_id": layer_id, "cluster_id": cluster_id,
        "target_layer_id": target_layer_id,
        "type": PACKET_TYPE_NAMES.get(ptype, f"UNKNOWN({ptype})"),
        "sequence": sequence, "payload_count": payload_count, "values": values,
    }


def main():
    parser = argparse.ArgumentParser(description="NNAwareBLE-compatible UDP receiver (no hardware needed)")
    parser.add_argument("--port", type=int, default=4210)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", args.port))

    print(f"[receiver] listening on UDP port {args.port} ...")
    received_count = 0
    rejected_count = 0

    try:
        while True:
            data, addr = sock.recvfrom(1024)
            ok, decoded = deserialize_packet(data)
            if ok:
                received_count += 1
                print(f"[receiver] OK from {addr[0]}: nodeId={decoded['node_id']} "
                      f"layer={decoded['layer_id']} type={decoded['type']} "
                      f"seq={decoded['sequence']} values={decoded['values']}")
            else:
                rejected_count += 1
                print(f"[receiver] REJECTED from {addr[0]} ({len(data)} bytes, "
                      f"hex={data.hex()}): {decoded['error']}")

            if (received_count + rejected_count) % 10 == 0:
                print(f"[receiver] --- summary: {received_count} accepted, "
                      f"{rejected_count} rejected ---")
    except KeyboardInterrupt:
        print(f"\n[receiver] stopped. Final: {received_count} accepted, {rejected_count} rejected.")


if __name__ == "__main__":
    main()