import argparse
import socket
import struct
import time
 
# ---------------------------------------------------------------------------
# WIRE FORMAT — must match NNPacket.h EXACTLY. '>' = big-endian, no padding.
#   sourceAddress (uint16), targetLayerId (uint8), type (uint8),
#   sequenceNumber (uint8), payloadCount (uint8), flags (uint8), checksum (uint8)
HEADER_FMT = ">HBBBBBB"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # should be 8, matching the C++ struct
 
PACKET_TYPE_DATA = 0  # DATA
 
 
def encode_address(node_id: int, layer_id: int, cluster_id: int = 0, reserved: int = 0) -> int:
    """Mirrors NNAddress.h's encodeAddress() exactly — same bit-shift layout."""
    return ((node_id & 0x0F) << 12) | ((layer_id & 0x0F) << 8) | ((cluster_id & 0x0F) << 4) | (reserved & 0x0F)
 
 
def compute_checksum(header_bytes_without_checksum: bytes, payload_bytes: bytes) -> int:

    total = sum(header_bytes_without_checksum) + sum(payload_bytes)
    return total & 0xFF
 
 
def build_packet(node_id: int, layer_id: int, target_layer_id: int,
                  sequence: int, values: list[float]) -> bytes:
    source_address = encode_address(node_id, layer_id)
    payload_count = len(values)
    payload_bytes = struct.pack(f">{payload_count}f", *values)
 
    # Pack header with checksum=0 first, to compute the checksum over it
    header_without_checksum = struct.pack(
        ">HBBBBB", source_address, target_layer_id, PACKET_TYPE_DATA,
        sequence & 0xFF, payload_count, 0  # 0 = flags placeholder, confirm meaning in your header
    )
    checksum = compute_checksum(header_without_checksum, payload_bytes)
 
    header = struct.pack(
        HEADER_FMT, source_address, target_layer_id, PACKET_TYPE_DATA,
        sequence & 0xFF, payload_count, 0, checksum
    )
    return header + payload_bytes
 
 
def main():
    parser = argparse.ArgumentParser(description="NNAwareBLE-compatible UDP transmitter (no hardware needed)")
    parser.add_argument("--node-id", type=int, required=True, help="This node's ID (0-15)")
    parser.add_argument("--layer-id", type=int, required=True, help="This node's layer (0-15)")
    parser.add_argument("--target-layer-id", type=int, default=15, help="Destination layer marker")
    parser.add_argument("--broadcast-addr", type=str, default="255.255.255.255",
                         help="Use 255.255.255.255 for same-machine testing, or your real subnet "
                              "broadcast address (e.g. 192.168.1.255) to reach real hardware")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between sends")
    args = parser.parse_args()
 
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
 
    counter = 0.0
    sequence = 0
    print(f"[transmitter] node_id={args.node_id} layer_id={args.layer_id} "
          f"-> {args.broadcast_addr}:{args.port}")
 
    try:
        while True:
            packet = build_packet(args.node_id, args.layer_id, args.target_layer_id,
                                   sequence, [counter])
            sock.sendto(packet, (args.broadcast_addr, args.port))
            print(f"[transmitter] SENT seq={sequence} value={counter:.2f} "
                  f"({len(packet)} bytes): {packet.hex()}")
            counter += 1.0
            sequence += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[transmitter] stopped.")
 
 
if __name__ == "__main__":
    main()
 