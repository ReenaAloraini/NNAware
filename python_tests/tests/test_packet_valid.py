import pytest
 
from receiver import deserialize_packet, decode_address, NN_MAX_PAYLOAD_FLOATS
from packet_builder import build_full_packet
 
ABS_TOLERANCE = 1e-5
REL_TOLERANCE = 1e-6  # float32 gives ~7 significant decimal digits
 
 
def assert_floats_close(actual, expected):
    """
    Compares using BOTH an absolute and a relative tolerance, taking the
    looser (larger) bound — standard practice for float32 comparisons.
    A fixed absolute tolerance alone is too tight for large-magnitude
    values: float32 only guarantees ~7 significant decimal digits, so a
    value like 123456.789 legitimately round-trips as 123456.7890625,
    which is correct behavior, not a precision bug in the packet format.
    """
    assert len(actual) == len(expected), f"length mismatch: {len(actual)} vs {len(expected)}"
    for a, e in zip(actual, expected):
        tolerance = max(ABS_TOLERANCE, REL_TOLERANCE * abs(e))
        assert abs(a - e) < tolerance, f"{a} != {e} (tolerance {tolerance})"
 
 
#  Valid DATA / CONTROL / ACK packets 
 
@pytest.mark.parametrize("packet_type,type_name", [(0, "DATA"), (1, "CONTROL"), (2, "ACK")])
def test_valid_packet_by_type(packet_type, type_name):
    pkt = build_full_packet(node_id=3, layer_id=2, packet_type=packet_type, sequence=7, values=[1.0])
    ok, decoded = deserialize_packet(pkt)
    assert ok, f"valid {type_name} packet should be accepted: {decoded}"
    assert decoded["type"] == type_name
 
 
# Payload size variations 
 
def test_empty_payload():
    pkt = build_full_packet(node_id=1, layer_id=1, values=[])
    ok, decoded = deserialize_packet(pkt)
    assert ok
    assert decoded["payload_count"] == 0
    assert decoded["values"] == ()
 
 
def test_one_float_payload():
    pkt = build_full_packet(node_id=1, layer_id=1, values=[3.14])
    ok, decoded = deserialize_packet(pkt)
    assert ok
    assert decoded["payload_count"] == 1
    assert_floats_close(decoded["values"], [3.14])
 
 
def test_multiple_float_payload():
    values = [1.0, -2.5, 3.75, 0.0, 100.125]
    pkt = build_full_packet(node_id=1, layer_id=1, values=values)
    ok, decoded = deserialize_packet(pkt)
    assert ok
    assert decoded["payload_count"] == len(values)
    assert_floats_close(decoded["values"], values)
 
 
def test_maximum_allowed_payload():
    values = [float(i) for i in range(NN_MAX_PAYLOAD_FLOATS)]
    pkt = build_full_packet(node_id=1, layer_id=1, values=values)
    ok, decoded = deserialize_packet(pkt)
    assert ok, f"packet at exactly NN_MAX_PAYLOAD_FLOATS ({NN_MAX_PAYLOAD_FLOATS}) should be accepted"
    assert decoded["payload_count"] == NN_MAX_PAYLOAD_FLOATS
    assert_floats_close(decoded["values"], values)
 
 
# Sequence number behavior 
 
@pytest.mark.parametrize("sequence", [0, 1, 42, 254, 255])
def test_sequence_number_preserved(sequence):
    pkt = build_full_packet(node_id=1, layer_id=1, sequence=sequence, values=[1.0])
    ok, decoded = deserialize_packet(pkt)
    assert ok
    assert decoded["sequence"] == sequence
 
 
def test_sequence_number_wraparound_255_to_0():
    """
    sequenceNumber is a single byte; 255 -> 0 wraparound is expected
    wire-format behavior (masked with & 0xFF when constructed), not an
    error. This test locks in that expected behavior explicitly rather
    than leaving it implicit.
    """
    pkt_255 = build_full_packet(node_id=1, layer_id=1, sequence=255, values=[1.0])
    ok255, decoded255 = deserialize_packet(pkt_255)
    assert ok255 and decoded255["sequence"] == 255
 
    pkt_wrapped = build_full_packet(node_id=1, layer_id=1, sequence=256, values=[1.0])  # wraps to 0
    ok0, decoded0 = deserialize_packet(pkt_wrapped)
    assert ok0 and decoded0["sequence"] == 0
 
 
# Address encoding / decoding 
 
@pytest.mark.parametrize("node_id,layer_id,cluster_id,reserved", [
    (0, 0, 0, 0),
    (15, 15, 15, 15),   # maximum value for every 4-bit field simultaneously
    (5, 3, 2, 0),
    (1, 0, 0, 0),
])
def test_address_encode_decode_roundtrip(node_id, layer_id, cluster_id, reserved):
    encoded = ((node_id & 0x0F) << 12) | ((layer_id & 0x0F) << 8) | \
              ((cluster_id & 0x0F) << 4) | (reserved & 0x0F)
    decoded_node, decoded_layer, decoded_cluster, decoded_reserved = decode_address(encoded)
    assert (decoded_node, decoded_layer, decoded_cluster, decoded_reserved) == \
           (node_id, layer_id, cluster_id, reserved)
 
 
#  Full round-trip and field preservation 
 
def test_serialize_then_deserialize_full_roundtrip():
    """End-to-end: build a packet with known field values, decode it back,
    confirm every field survives unchanged."""
    pkt = build_full_packet(node_id=4, layer_id=6, cluster_id=1, target_layer_id=7,
                             packet_type=0, sequence=99, values=[2.5, -1.25, 0.0])
    ok, decoded = deserialize_packet(pkt)
    assert ok
    assert decoded["node_id"] == 4
    assert decoded["layer_id"] == 6
    assert decoded["cluster_id"] == 1
    assert decoded["target_layer_id"] == 7
    assert decoded["type"] == "DATA"
    assert decoded["sequence"] == 99
    assert_floats_close(decoded["values"], [2.5, -1.25, 0.0])
 
 
def test_all_header_fields_preserved_independently():
    """
    Isolated check that EVERY header field survives independently, with
    distinct field names in the assertion messages — distinguishes this
    from the general round-trip test above by pinpointing exactly which
    field broke, rather than one broad assertion covering all of them.
    """
    pkt = build_full_packet(node_id=9, layer_id=8, cluster_id=3, reserved=0,
                             target_layer_id=12, packet_type=1, sequence=200, values=[7.0])
    ok, decoded = deserialize_packet(pkt)
    assert ok
    assert decoded["node_id"] == 9, "nodeId not preserved"
    assert decoded["layer_id"] == 8, "layerId not preserved"
    assert decoded["cluster_id"] == 3, "clusterId not preserved"
    assert decoded["target_layer_id"] == 12, "targetLayerId not preserved"
    assert decoded["type"] == "CONTROL", "type not preserved"
    assert decoded["sequence"] == 200, "sequenceNumber not preserved"
 
 
def test_float_values_decoded_correctly_including_negatives_and_zero():
    values = [0.0, -0.25, 10.0, -1.0, 0.5, 123456.789]
    pkt = build_full_packet(node_id=1, layer_id=1, values=values)
    ok, decoded = deserialize_packet(pkt)
    assert ok
    assert_floats_close(decoded["values"], values)
 