import pytest
 
from receiver import deserialize_packet, decode_address, HEADER_SIZE, NN_MAX_PAYLOAD_FLOATS
from packet_builder import build_full_packet
 
 
# ---------------------------------------------------------------------------
# Tests that run now — the real deserializePacket() logic is confirmed (by
# direct inspection of NNPacket.h) to reject each of these.
# ---------------------------------------------------------------------------
 
def test_corrupted_checksum_is_rejected():
    pkt = build_full_packet(node_id=1, layer_id=1, values=[1.0], corrupt_checksum=True)
    ok, decoded = deserialize_packet(pkt)
    assert not ok
    assert "checksum" in decoded["error"].lower()
 
 
def test_corrupted_header_byte_is_rejected():
    """Flipping any header byte changes the byte content the checksum was
    computed over, so this is caught by checksum verification."""
    pkt = bytearray(build_full_packet(node_id=1, layer_id=1, values=[1.0]))
    pkt[2] ^= 0xFF  # flip targetLayerId
    ok, decoded = deserialize_packet(bytes(pkt))
    assert not ok
    assert "checksum" in decoded["error"].lower()
 
 
def test_corrupted_payload_byte_is_rejected():
    pkt = bytearray(build_full_packet(node_id=1, layer_id=1, values=[1.0]))
    pkt[HEADER_SIZE] ^= 0xFF  # flip the first payload byte
    ok, decoded = deserialize_packet(bytes(pkt))
    assert not ok
    assert "checksum" in decoded["error"].lower()
 
 
def test_packet_shorter_than_minimum_header_size_is_rejected():
    tiny = build_full_packet(node_id=1, layer_id=1, values=[])[:HEADER_SIZE - 1]
    ok, decoded = deserialize_packet(tiny)
    assert not ok
    assert "too short" in decoded["error"].lower()
 
 
def test_truncated_payload_is_rejected():
    """Header declares 2 floats; only enough bytes for 1.5 floats present."""
    pkt = build_full_packet(node_id=1, layer_id=1, values=[1.0, 2.0])
    truncated = pkt[:-2]
    ok, decoded = deserialize_packet(truncated)
    assert not ok
 
 
def test_extra_unexpected_bytes_are_rejected():
    """
    Regression test for the bug fixed in the previous step: a valid packet
    with trailing garbage appended must now be rejected outright, not
    silently accepted with the garbage ignored.
    """
    pkt = build_full_packet(node_id=1, layer_id=1, values=[1.0], extra_bytes=b"\xDE\xAD\xBE\xEF")
    ok, decoded = deserialize_packet(pkt)
    assert not ok
 
 
def test_payload_count_larger_than_maximum_is_rejected():
    """Regression test for the second bug fixed in the previous step."""
    pkt = build_full_packet(node_id=1, layer_id=1, values=[1.0] * NN_MAX_PAYLOAD_FLOATS,
                             override_payload_count=NN_MAX_PAYLOAD_FLOATS + 1)
    ok, decoded = deserialize_packet(pkt)
    assert not ok
    assert "exceeds" in decoded["error"].lower()
 
 
def test_payload_count_inconsistent_with_packet_length_is_rejected():
    """
    payloadCount is WITHIN the valid 0-16 range, but doesn't match the
    actual number of bytes present — distinct from both the truncation
    case (too few bytes for the declared count) and the ceiling-exceeded
    case above (declared count itself invalid).
    """
    pkt = build_full_packet(node_id=1, layer_id=1, values=[1.0, 2.0, 3.0],
                             override_payload_count=2)  # claims 2; 3 floats' worth of bytes present
    ok, decoded = deserialize_packet(pkt)
    assert not ok
    assert "length mismatch" in decoded["error"].lower()
 
 
# ---------------------------------------------------------------------------
# Documented protocol gaps — the library does NOT currently validate these.
# Each test below either documents current (accepting) behavior explicitly,
# or is marked xfail/skip describing exactly what's missing and why.
# ---------------------------------------------------------------------------
 
def test_undefined_packet_type_value_is_currently_accepted():
    """
    DOCUMENTED GAP, not a test bug: NNPacket::deserializePacket casts
    whatever byte is present directly to NNPacketType with no validation
    that it's one of the three defined values (0=DATA, 1=CONTROL, 2=ACK).
    This test documents CURRENT behavior (acceptance, surfaced as
    "UNKNOWN(N)") rather than inventing a rejection the library doesn't do.
    """
    pkt = build_full_packet(node_id=1, layer_id=1, packet_type=99, values=[1.0])
    ok, decoded = deserialize_packet(pkt)
    assert ok, "current implementation has no type validation and accepts any byte value"
    assert decoded["type"] == "UNKNOWN(99)"
 
 
@pytest.mark.xfail(reason="NNPacket::deserializePacket does not validate `type` against the "
                           "defined NNPacketType enum values — a protocol gap, not yet a library "
                           "feature. This documents DESIRED future behavior (rejecting undefined "
                           "type values) and is expected to fail until the library adds explicit "
                           "type validation.", strict=True)
def test_invalid_packet_type_should_eventually_be_rejected():
    pkt = build_full_packet(node_id=1, layer_id=1, packet_type=99, values=[1.0])
    ok, decoded = deserialize_packet(pkt)
    assert not ok, "an undefined type value should be rejected (not yet implemented)"
 
 
@pytest.mark.parametrize("raw_address", [0x0000, 0xFFFF, 0x1234, 0xABCD])
def test_every_16_bit_source_address_decodes_without_error(raw_address):
    """
    There is no possible 'invalid' encoded source address in the current
    protocol: decodeAddress() unconditionally masks every field to 4 bits,
    so ANY raw 16-bit value decodes into SOME in-range (0-15) NNAddress by
    construction. This documents that structural property directly, rather
    than inventing a rejection case the masking design makes impossible.
    """
    node_id, layer_id, cluster_id, reserved = decode_address(raw_address)
    assert 0 <= node_id <= 15
    assert 0 <= layer_id <= 15
    assert 0 <= cluster_id <= 15
    assert 0 <= reserved <= 15
 
 
@pytest.mark.skip(reason="targetLayerId is a raw uint8_t (0-255) with no documented valid-range "
                          "restriction anywhere in NNPacket.h or NNNode.h, and deserializePacket "
                          "performs no validation on it. There is currently no defined 'invalid' "
                          "value to test against — skipped rather than inventing a validation rule "
                          "the protocol does not specify.")
def test_invalid_target_layer_is_rejected():
    pass
 