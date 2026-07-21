"""
Wire-format helpers for NNAware setup-phase packets.

Mirrors NNPacket.h's serializePacket()/deserializePacket() and
NNAddress.h's encodeAddress()/decodeAddress() exactly. Deliberately
duplicated here (rather than imported) -- see python_tests/tests/receiver.py
for the pytest-side twin of this logic -- because this is a standalone
operational tool meant to run against real hardware, not part of the
pytest tree, and importing across into python_tests/ would be the wrong
dependency direction.
"""
import struct

# sourceAddress (uint16), targetLayerId (uint8), type (uint8),
# sequenceNumber (uint8), payloadCount (uint8), flags (uint8), checksum (uint8)
HEADER_FMT = ">HBBBBBB"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 8, matches NNPacketHeader
CHECKSUM_INDEX = 7

PACKET_TYPE_CONTROL = 1  # NNPacketType::CONTROL, from NNPacket.h


def encode_address(node_id: int, layer_id: int, cluster_id: int = 0, reserved: int = 0) -> int:
    """Mirrors NNAddress.h's encodeAddress()."""
    return ((node_id & 0x0F) << 12) | ((layer_id & 0x0F) << 8) | ((cluster_id & 0x0F) << 4) | (reserved & 0x0F)


def decode_address(encoded: int):
    """Mirrors NNAddress.h's decodeAddress(). Returns (nodeId, layerId, clusterId, reserved)."""
    return (encoded >> 12) & 0x0F, (encoded >> 8) & 0x0F, (encoded >> 4) & 0x0F, encoded & 0x0F


def compute_checksum(buffer: bytes) -> int:
    """Mirrors NNPacket.h's computeChecksum(): additive sum of bytes, truncated to 8 bits."""
    return sum(buffer) & 0xFF


def reverse_words(data: bytes) -> bytes:
    """
    Mirrors the word-level transform serializePacket()/deserializePacket()
    apply to every 4-byte chunk of the payload.

    NNPacket.h's wire format does NOT serialize payload words as IEEE-754
    floats semantically -- it reverses the 4 bytes of each word (byte-
    reversing a little-endian value produces its big-endian encoding,
    which is exactly why one reversal pass is both correct and self-
    inverse). Every setup message struct is packed little-endian here
    (matching the C++ #pragma pack(1) in-memory layout) and is always a
    whole number of 4-byte words (enforced by a static_assert in
    packSetupMessage), so this can operate blindly in 4-byte strides --
    an 8-byte hardwareId becomes two INDEPENDENTLY reversed 4-byte halves,
    not one 8-byte reversal.
    """
    if len(data) % 4 != 0:
        raise ValueError(f"data length {len(data)} is not a whole number of 4-byte words")
    return b"".join(data[i:i + 4][::-1] for i in range(0, len(data), 4))


def build_packet(opcode: int, sequence: int, msg_bytes: bytes,
                  source_address: int = 0, target_layer_id: int = 0) -> bytes:
    """
    Wraps an already little-endian-packed setup message struct (as produced
    by setup_messages.pack_*) into a full wire-format NNPacket with
    type=CONTROL and flags=opcode. Mirrors packSetupMessage() +
    serializePacket() together.
    """
    if len(msg_bytes) % 4 != 0:
        raise ValueError("setup messages must be a whole number of 4-byte words")
    payload_count = len(msg_bytes) // 4
    payload_wire = reverse_words(msg_bytes)

    # 6 header fields excluding checksum -- matches serializePacket()'s own
    # order: it computes the checksum over the buffer AFTER writing every
    # other byte, with the checksum byte itself held at zero.
    header_without_checksum = struct.pack(
        ">HBBBBB", source_address, target_layer_id, PACKET_TYPE_CONTROL,
        sequence & 0xFF, payload_count, opcode & 0xFF,
    )
    checksum = compute_checksum(header_without_checksum + b"\x00" + payload_wire)

    header = struct.pack(
        HEADER_FMT, source_address, target_layer_id, PACKET_TYPE_CONTROL,
        sequence & 0xFF, payload_count, opcode & 0xFF, checksum,
    )
    return header + payload_wire


def parse_packet(data: bytes):
    """
    Returns (ok, opcode, sequence, msg_bytes) for a CONTROL packet, or
    (False, None, None, None) on any validation failure. Mirrors
    deserializePacket()'s exact validation order, then un-reverses the
    payload back into the little-endian struct bytes a setup_messages
    unpack_* function expects.
    """
    if len(data) < HEADER_SIZE:
        return False, None, None, None

    received_checksum = data[CHECKSUM_INDEX]
    checksum_input = bytearray(data)
    checksum_input[CHECKSUM_INDEX] = 0
    if compute_checksum(checksum_input) != received_checksum:
        return False, None, None, None

    source_address, target_layer_id, ptype, sequence, payload_count, flags, _checksum = \
        struct.unpack(HEADER_FMT, data[:HEADER_SIZE])

    if ptype != PACKET_TYPE_CONTROL:
        return False, None, None, None

    expected_total = HEADER_SIZE + payload_count * 4
    if len(data) != expected_total:
        return False, None, None, None

    payload_wire = data[HEADER_SIZE:expected_total]
    msg_bytes = reverse_words(payload_wire)  # self-inverse
    return True, flags, sequence, msg_bytes
