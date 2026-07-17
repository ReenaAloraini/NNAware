import struct
from receiver import compute_checksum, CHECKSUM_INDEX, HEADER_FMT
 
 
def build_full_packet(node_id=0, layer_id=0, cluster_id=0, reserved=0,
                       target_layer_id=15, packet_type=0, sequence=0,
                       values=None, *, corrupt_checksum=False,
                       override_payload_count=None, extra_bytes=b"",
                       truncate_to=None):
    """
    Builds a wire-format packet byte string with full control over every
    field, including deliberately invalid combinations for negative testing.
 
    Args:
        corrupt_checksum: if True, writes a checksum guaranteed NOT to match
            the packet's actual content.
        override_payload_count: if set, writes THIS value into the header's
            payloadCount field instead of len(values) — lets a test claim a
            payload count that doesn't match the bytes actually present.
        extra_bytes: appended after the real payload, for trailing-garbage tests.
        truncate_to: if set, the final packet is sliced to this many bytes.
    """
    if values is None:
        values = []
 
    source_address = ((node_id & 0x0F) << 12) | ((layer_id & 0x0F) << 8) | \
                      ((cluster_id & 0x0F) << 4) | (reserved & 0x0F)
    payload_count_field = override_payload_count if override_payload_count is not None else len(values)
    payload_bytes = struct.pack(f">{len(values)}f", *values) if values else b""
 
    header = struct.pack(HEADER_FMT, source_address, target_layer_id, packet_type,
                          sequence & 0xFF, payload_count_field, 0, 0)  # checksum placeholder
    packet = bytearray(header + payload_bytes)
 
    checksum_input = bytearray(packet)
    checksum_input[CHECKSUM_INDEX] = 0
    correct_checksum = compute_checksum(checksum_input)
    packet[CHECKSUM_INDEX] = ((correct_checksum + 1) & 0xFF) if corrupt_checksum else correct_checksum
 
    packet += extra_bytes
 
    if truncate_to is not None:
        packet = packet[:truncate_to]
 
    return bytes(packet)