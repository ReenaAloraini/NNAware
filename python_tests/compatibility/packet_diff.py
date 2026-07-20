"""
byte-level diff utility for comparing two serialized
packets, used by Test C (bit-for-bit serialization comparison). Only ever
invoked when a comparison FAILS; its job is to make the failure
immediately diagnosable rather than just reporting "bytes differ."
"""
 
# Field boundaries, matching the confirmed 8-byte NNPacketHeader layout.
# Anything at offset >= 8 is payload; which float index depends on offset.
_HEADER_FIELDS = [
    (0, 2, "sourceAddress"),
    (2, 1, "targetLayerId"),
    (3, 1, "type"),
    (4, 1, "sequenceNumber"),
    (5, 1, "payloadCount"),
    (6, 1, "flags"),
    (7, 1, "checksum"),
]
HEADER_SIZE = 8
 
 
def _field_name_for_offset(offset: int) -> str:
    for start, length, name in _HEADER_FIELDS:
        if start <= offset < start + length:
            return name
    payload_offset = offset - HEADER_SIZE
    float_index = payload_offset // 4
    byte_in_float = payload_offset % 4
    return f"payload[{float_index}] (byte {byte_in_float} of 4)"
 
 
def diff_packets(python_hex: str, cpp_hex: str) -> str:
    """
    Returns a diff report identifying exactly which bytes differ, 
    their offsets, the likely field each belongs to,
    and surrounding context bytes. Callers should check equality
    themselves before calling this — this function explains a difference,
    it doesn't detect one.
    """
    py_bytes = bytes.fromhex(python_hex)
    cpp_bytes = bytes.fromhex(cpp_hex)
 
    lines = [f"Python length: {len(py_bytes)} bytes, C++ length: {len(cpp_bytes)} bytes"]
    if len(py_bytes) != len(cpp_bytes):
        lines.append("LENGTH MISMATCH — the two packets are not even the same size.")
 
    max_len = max(len(py_bytes), len(cpp_bytes))
    diff_offsets = [i for i in range(max_len)
                     if (py_bytes[i] if i < len(py_bytes) else None) !=
                        (cpp_bytes[i] if i < len(cpp_bytes) else None)]
 
    if not diff_offsets:
        return ""  # identical after all
 
    lines.append(f"\n{len(diff_offsets)} byte(s) differ, at offsets: {diff_offsets}\n")
 
    for offset in diff_offsets:
        py_byte = py_bytes[offset] if offset < len(py_bytes) else None
        cpp_byte = cpp_bytes[offset] if offset < len(cpp_bytes) else None
        field = _field_name_for_offset(offset)
 
        py_str = f"0x{py_byte:02x}" if py_byte is not None else "<missing>"
        cpp_str = f"0x{cpp_byte:02x}" if cpp_byte is not None else "<missing>"
 
        lo = max(0, offset - 2)
        py_nearby = py_bytes[lo:min(len(py_bytes), offset + 3)].hex(" ")
        cpp_nearby = cpp_bytes[lo:min(len(cpp_bytes), offset + 3)].hex(" ")
 
        lines.append(
            f"  offset {offset:3d} ({field}):\n"
            f"      Python: {py_str}   (nearby: ...{py_nearby}...)\n"
            f"      C++:    {cpp_str}   (nearby: ...{cpp_nearby}...)"
        )
 
    return "\n".join(lines)
 