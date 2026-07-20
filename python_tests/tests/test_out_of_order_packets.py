"""
out-of-order packet handling. this tests
what can be verified at the wire-format level, specifically, that 
reordering a sequence of packets does not corrupt or misattribute any 
individual packet's content, and that every packet remains independently 
decodable regardless of the position it arrives in relative to the others.
"""
import pytest

from receiver import deserialize_packet
from packet_builder import build_full_packet
from cpp_runner import CppRunner, CppExecutableError
import fault_injection as fi

try:
    _runner = CppRunner()
    CPP_AVAILABLE = True
    _skip_reason = None
except CppExecutableError as e:
    CPP_AVAILABLE = False
    _skip_reason = str(e)


def _build_labeled_sequence(count=5):
    """
    Builds a sequence of packets, each carrying a distinct, self-
    identifying sequence number AND a distinct payload value — so after
    reordering, we can verify each packet's DECODED content still
    correctly identifies which original packet it was, independent of its
    new position in the list.
    """
    return [
        build_full_packet(node_id=1, layer_id=1, sequence=i, values=[float(i) * 10.0])
        for i in range(count)
    ]


# --- Core property: reordering does not corrupt or misattribute content ----

def test_reordered_packets_each_still_decode_correctly():
    original = _build_labeled_sequence(count=5)
    reordered = fi.reorder_packets(original, new_order=[3, 0, 4, 1, 2])

    for pkt in reordered:
        ok, decoded = deserialize_packet(pkt)
        assert ok, f"a packet that decoded fine before reordering failed after: {pkt.hex()}"


def test_reordered_packets_retain_correct_individual_identity():
    """
    The critical property: packet content is self-describing via its own
    embedded sequence number and payload — reordering the LIST does not
    scramble what's INSIDE each packet. The packet now at list position 0
    (originally packet #3) must still report sequence=3 and value=30.0,
    not sequence=0.
    """
    original = _build_labeled_sequence(count=5)
    new_order = [3, 0, 4, 1, 2]
    reordered = fi.reorder_packets(original, new_order=new_order)

    for position_in_reordered_list, original_index in enumerate(new_order):
        ok, decoded = deserialize_packet(reordered[position_in_reordered_list])
        assert ok
        assert decoded["sequence"] == original_index, (
            f"packet now at list position {position_in_reordered_list} should still "
            f"self-identify as original sequence {original_index}, got {decoded['sequence']}"
        )
        expected_value = float(original_index) * 10.0
        assert abs(decoded["values"][0] - expected_value) < 1e-6, (
            f"packet now at list position {position_in_reordered_list} should still "
            f"carry its original value {expected_value}, got {decoded['values'][0]}"
        )


def test_reordering_preserves_the_full_set_of_decoded_content():
    """
    Order-independence from a different angle: decode every packet in the
    ORIGINAL order and the REORDERED order, collect (sequence, value)
    pairs from each, and confirm the two SETS are identical — nothing was
    lost, duplicated, or altered by the act of reordering alone.
    """
    original = _build_labeled_sequence(count=6)
    reordered = fi.reorder_packets(original, new_order=[5, 3, 1, 4, 0, 2])

    def decode_all(packets):
        results = set()
        for pkt in packets:
            ok, decoded = deserialize_packet(pkt)
            assert ok
            results.add((decoded["sequence"], round(decoded["values"][0], 6)))
        return results

    original_set = decode_all(original)
    reordered_set = decode_all(reordered)
    assert original_set == reordered_set, (
        f"reordering changed the set of decodable content.\n"
        f"original: {original_set}\nreordered: {reordered_set}"
    )


# Cross-language: both Python and C++ agree on out-of-order decoding

@pytest.mark.skipif(not CPP_AVAILABLE, reason=_skip_reason or "C++ executable not found")
def test_reordered_packets_decode_identically_in_python_and_cpp():
    original = _build_labeled_sequence(count=5)
    reordered = fi.reorder_packets(original, new_order=[2, 4, 0, 3, 1])

    for pkt in reordered:
        py_ok, py_decoded = deserialize_packet(pkt)
        cpp_result = _runner.deserialize(pkt.hex())

        assert py_ok and cpp_result["ok"], (
            f"Python and C++ disagreed on whether this packet decodes: "
            f"Python ok={py_ok}, C++ ok={cpp_result['ok']}, bytes={pkt.hex()}"
        )
        assert py_decoded["sequence"] == cpp_result["sequence"]
        assert abs(py_decoded["values"][0] - cpp_result["values"][0]) < 1e-6


#Edge cases

def test_reversed_order_still_decodes_correctly():
    """A specific, easy-to-reason-about reordering: fully reversed."""
    original = _build_labeled_sequence(count=4)
    reversed_seq = fi.reorder_packets(original, new_order=[3, 2, 1, 0])

    for position, pkt in enumerate(reversed_seq):
        ok, decoded = deserialize_packet(pkt)
        assert ok
        assert decoded["sequence"] == 3 - position


def test_single_packet_reorder_is_a_no_op():
    """A sequence of exactly one packet has only one valid 'order' — sanity check."""
    original = _build_labeled_sequence(count=1)
    result = fi.reorder_packets(original, new_order=[0])
    assert result == original