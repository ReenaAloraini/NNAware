"""
Tests cross-language compatibility between the real Python implementation
and the real C++ implementation (cpp_runner.py, which
invokes the actual compiled protocol_compatibility executable .

These tests are SKIPPED, not failed, if the C++ executable hasn't been
built yet, so the rest of the Python-only test suite can still run without
requiring a C++ toolchain to be present.
"""
import pytest

from transmitter import build_packet
from receiver import deserialize_packet
from cpp_runner import CppRunner, CppExecutableError
from packet_diff import diff_packets

try:
    _runner = CppRunner()
    CPP_AVAILABLE = True
    _skip_reason = None
except CppExecutableError as e:
    CPP_AVAILABLE = False
    _skip_reason = str(e)

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason=_skip_reason or "C++ executable not found")

# Slightly looser than the pure-Python suite's tolerance: values here pass
# through TWO independent float32 implementations (Python struct + C++
# float), not one, so a hair more slack is appropriate.
FLOAT_TOLERANCE = 1e-4


def assert_floats_close(actual, expected):
    assert len(actual) == len(expected), f"length mismatch: {len(actual)} vs {len(expected)}"
    for a, e in zip(actual, expected):
        tol = max(FLOAT_TOLERANCE, 1e-6 * abs(e))
        assert abs(a - e) < tol, f"{a} != {e} (tolerance {tol})"


TEST_VECTORS = [
    # (node_id, layer_id, target_layer_id, sequence, values)
    (0, 0, 15, 0, [0.0]),
    (2, 1, 15, 5, [3.14, 42.0]),
    (15, 15, 255, 255, [-1.5, 0.0, 99.25]),
    (7, 4, 9, 200, []),
    (1, 1, 1, 1, [float(i) for i in range(16)]),  # maximum payload (16 floats)
]


# Test A: C++ serialize -> Python deserialize 

@pytest.mark.parametrize("node_id,layer_id,target_layer_id,seq,values", TEST_VECTORS)
def test_A_cpp_serialize_python_deserialize(node_id, layer_id, target_layer_id, seq, values):
    source_address = _runner.encode_address(node_id=node_id, layer_id=layer_id)["encoded"]

    cpp_result = _runner.serialize(source_address=source_address, target_layer_id=target_layer_id,
                                    packet_type=0, sequence=seq, values=values)
    assert cpp_result["ok"], f"C++ serialize failed: {cpp_result}"

    packet_bytes = bytes.fromhex(cpp_result["bytes_hex"])
    ok, decoded = deserialize_packet(packet_bytes)
    assert ok, f"Python REJECTED a packet C++ itself produced: {decoded}"

    assert decoded["node_id"] == node_id
    assert decoded["layer_id"] == layer_id
    assert decoded["target_layer_id"] == target_layer_id
    assert decoded["sequence"] == seq
    assert_floats_close(decoded["values"], values)


# Test B: Python serialize -> C++ deserialize 

@pytest.mark.parametrize("node_id,layer_id,target_layer_id,seq,values", TEST_VECTORS)
def test_B_python_serialize_cpp_deserialize(node_id, layer_id, target_layer_id, seq, values):
    pkt_bytes = build_packet(node_id=node_id, layer_id=layer_id, target_layer_id=target_layer_id,
                              sequence=seq, values=values)

    cpp_result = _runner.deserialize(pkt_bytes.hex())
    assert cpp_result["ok"], f"C++ REJECTED a packet Python itself produced: {cpp_result}"

    addr = _runner.decode_address(cpp_result["source_address"])
    assert addr["node_id"] == node_id
    assert addr["layer_id"] == layer_id
    assert cpp_result["target_layer_id"] == target_layer_id
    assert cpp_result["sequence"] == seq
    assert_floats_close(cpp_result["values"], values)


# Test C: bit-for-bit serialization comparison 

@pytest.mark.parametrize("node_id,layer_id,target_layer_id,seq,values", TEST_VECTORS)
def test_C_bit_for_bit_serialization_match(node_id, layer_id, target_layer_id, seq, values):
    python_hex = build_packet(node_id=node_id, layer_id=layer_id, target_layer_id=target_layer_id,
                               sequence=seq, values=values).hex()

    source_address = _runner.encode_address(node_id=node_id, layer_id=layer_id)["encoded"]
    cpp_result = _runner.serialize(source_address=source_address, target_layer_id=target_layer_id,
                                    packet_type=0, sequence=seq, values=values)
    assert cpp_result["ok"], f"C++ serialize failed: {cpp_result}"
    cpp_hex = cpp_result["bytes_hex"]

    if python_hex != cpp_hex:
        report = diff_packets(python_hex, cpp_hex)
        pytest.fail(
            f"Python and C++ produced DIFFERENT bytes for the same logical packet.\n"
            f"Python hex: {python_hex}\nC++ hex:    {cpp_hex}\n\n{report}"
        )