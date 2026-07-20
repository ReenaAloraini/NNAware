"""
a fixed set of named packets whose expected hex bytes are frozen 
in compatibility/golden_packets.json.

These golden values were NOT hand-typed. They were generated from the real
transmitter.py, each one individually cross-verified against the real C++
implementation via cpp_runner.py BEFORE being written to the JSON file.
"""
import json
import pathlib
import pytest

from receiver import deserialize_packet
from transmitter import build_packet
from cpp_runner import CppRunner, CppExecutableError

GOLDEN_FILE = pathlib.Path(__file__).parent.parent / "compatibility" / "golden_packets.json"

with open(GOLDEN_FILE) as f:
    GOLDEN_PACKETS = json.load(f)

try:
    _runner = CppRunner()
    CPP_AVAILABLE = True
    _skip_reason = None
except CppExecutableError as e:
    CPP_AVAILABLE = False
    _skip_reason = str(e)


# --- Sanity: the golden file itself is well-formed and non-empty ----------

def test_golden_file_loaded_and_non_empty():
    assert len(GOLDEN_PACKETS) >= 10, "expected at least 10 golden vectors per the spec's required list"


def test_all_golden_vectors_generated_via_real_transmitter():
    """
    Locks in that the packet_builder.py workaround is no longer needed —
    if a future change reintroduces the gap this test suite originally
    caught, and someone quietly falls back to packet_builder again for
    convenience, this test will catch that regression in provenance.
    """
    for name, vector in GOLDEN_PACKETS.items():
        assert vector["generated_via"] == "transmitter", (
            f"golden vector '{name}' is not generated via the real transmitter.py "
            f"— this was the exact workaround needed before the packet_type fix."
        )


@pytest.mark.parametrize("required_category", [
    "data_one_float", "data_multiple_floats", "control_packet", "ack_packet",
    "empty_payload_data", "maximum_payload", "sequence_zero", "sequence_255",
    "negative_float", "zero_float", "positive_float", "non_integer_float",
])
def test_required_category_present(required_category):
    assert required_category in GOLDEN_PACKETS, f"missing required golden category: {required_category}"


# --- Python reproduces every golden vector exactly, via the real transmitter --

@pytest.mark.parametrize("name,vector", GOLDEN_PACKETS.items())
def test_python_reproduces_golden_bytes(name, vector):
    actual = build_packet(node_id=vector["node_id"], layer_id=vector["layer_id"],
                           target_layer_id=vector["target_layer_id"],
                           sequence=vector["sequence"], values=vector["values"],
                           packet_type=vector["packet_type"]).hex()

    assert actual == vector["expected_bytes_hex"], (
        f"golden vector '{name}' ({vector['description']}): Python produced {actual}, "
        f"expected {vector['expected_bytes_hex']}. If this fails, either the wire format "
        f"has changed (update the golden vectors deliberately) or a real regression occurred."
    )


# --- C++ reproduces every golden vector exactly ------------------------------

@pytest.mark.skipif(not CPP_AVAILABLE, reason=_skip_reason or "C++ executable not found")
@pytest.mark.parametrize("name,vector", GOLDEN_PACKETS.items())
def test_cpp_reproduces_golden_bytes(name, vector):
    source_address = _runner.encode_address(node_id=vector["node_id"], layer_id=vector["layer_id"])["encoded"]
    result = _runner.serialize(source_address=source_address, target_layer_id=vector["target_layer_id"],
                                packet_type=vector["packet_type"], sequence=vector["sequence"],
                                values=vector["values"])
    assert result["ok"], f"C++ serialize failed for golden vector '{name}': {result}"
    assert result["bytes_hex"] == vector["expected_bytes_hex"], (
        f"golden vector '{name}': C++ produced {result['bytes_hex']}, "
        f"expected {vector['expected_bytes_hex']}."
    )


# --- Every golden vector also round-trips through deserialize correctly -----

@pytest.mark.parametrize("name,vector", GOLDEN_PACKETS.items())
def test_golden_bytes_deserialize_correctly(name, vector):
    pkt_bytes = bytes.fromhex(vector["expected_bytes_hex"])
    ok, decoded = deserialize_packet(pkt_bytes)
    assert ok, f"golden vector '{name}' failed to deserialize: {decoded}"
    assert decoded["node_id"] == vector["node_id"]
    assert decoded["layer_id"] == vector["layer_id"]
    assert decoded["sequence"] == vector["sequence"]
    assert decoded["payload_count"] == len(vector["values"])


# --- Dedicated regression test for the packet_type fix itself ---------------

def test_transmitter_packet_type_regression():
    """
    Direct regression test for the fix: build_packet() must accept
    packet_type and actually honor it (not silently send DATA regardless,
    as it did before). Also confirms the old-style call (no packet_type
    argument at all) still defaults to DATA — the backward-compatibility
    guarantee the fix was required to preserve.
    """
    control_pkt = build_packet(node_id=1, layer_id=1, target_layer_id=15,
                                sequence=1, values=[], packet_type=1)
    assert control_pkt[3] == 1, "packet_type=1 (CONTROL) was not honored — regression of the original bug"

    ack_pkt = build_packet(node_id=1, layer_id=1, target_layer_id=15,
                            sequence=1, values=[1.0], packet_type=2)
    assert ack_pkt[3] == 2, "packet_type=2 (ACK) was not honored — regression of the original bug"

    old_style_pkt = build_packet(node_id=1, layer_id=1, target_layer_id=15, sequence=1, values=[1.0])
    assert old_style_pkt[3] == 0, "default packet_type must remain DATA for backward compatibility"