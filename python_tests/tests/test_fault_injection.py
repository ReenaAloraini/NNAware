"""
tests for the fault_injection.py helpers themselves, and, where relevant, 
confirms each fault type actually causes the real receiver.py to reject the
mutated packet — tying fault injection directly back to validating production 
behavior, not just testing the helpers in a vacuum.
"""
import pytest

from receiver import deserialize_packet
from packet_builder import build_full_packet
import fault_injection as fi


@pytest.fixture
def valid_packet():
    return build_full_packet(node_id=1, layer_id=1, values=[1.0, 2.0, 3.0])


@pytest.fixture
def packet_sequence():
    return [build_full_packet(node_id=1, layer_id=1, sequence=i, values=[float(i)]) for i in range(5)]


#Single-byte / single-packet faults 

def test_flip_byte_changes_only_the_targeted_byte(valid_packet):
    mutated = fi.flip_byte(valid_packet, offset=2)
    assert mutated[2] == valid_packet[2] ^ 0xFF
    for i in range(len(valid_packet)):
        if i != 2:
            assert mutated[i] == valid_packet[i], f"byte at offset {i} changed unexpectedly"


def test_flip_byte_out_of_range_raises():
    with pytest.raises(ValueError):
        fi.flip_byte(b"\x00" * 4, offset=10)


def test_corrupt_checksum_causes_rejection(valid_packet):
    mutated = fi.corrupt_checksum(valid_packet)
    ok, decoded = deserialize_packet(mutated)
    assert not ok
    assert "checksum" in decoded["error"].lower()


def test_corrupt_payload_byte_causes_rejection(valid_packet):
    mutated = fi.corrupt_payload_byte(valid_packet, payload_byte_index=0)
    ok, decoded = deserialize_packet(mutated)
    assert not ok


def test_corrupt_payload_byte_out_of_range_raises(valid_packet):
    with pytest.raises(ValueError):
        fi.corrupt_payload_byte(valid_packet, payload_byte_index=999)


def test_truncate_causes_rejection(valid_packet):
    mutated = fi.truncate(valid_packet, length=5)
    ok, decoded = deserialize_packet(mutated)
    assert not ok
    assert "too short" in decoded["error"].lower()


def test_append_extra_bytes_causes_rejection(valid_packet):
    mutated = fi.append_extra_bytes(valid_packet)
    ok, decoded = deserialize_packet(mutated)
    assert not ok


def test_change_packet_type_without_recompute_causes_checksum_rejection(valid_packet):
    """Default behavior: type changed, checksum left stale -> rejected."""
    mutated = fi.change_packet_type(valid_packet, new_type=1)
    ok, decoded = deserialize_packet(mutated)
    assert not ok
    assert "checksum" in decoded["error"].lower()


def test_change_packet_type_with_recompute_is_accepted_with_new_type(valid_packet):
    """With recompute_checksum=True: a genuinely valid packet of the new type."""
    mutated = fi.change_packet_type(valid_packet, new_type=1, recompute_checksum=True)
    ok, decoded = deserialize_packet(mutated)
    assert ok
    assert decoded["type"] == "CONTROL"


def test_change_payload_count_without_recompute_causes_rejection(valid_packet):
    mutated = fi.change_payload_count(valid_packet, new_count=1)
    ok, decoded = deserialize_packet(mutated)
    assert not ok


def test_change_payload_count_with_recompute_but_mismatched_length_still_rejected(valid_packet):
    """
    Even recomputing the checksum can't make this valid: payloadCount=1
    but the packet still physically contains 3 floats' worth of bytes —
    the exact-length check (fixed earlier in this project) catches it
    independently of the checksum.
    """
    mutated = fi.change_payload_count(valid_packet, new_count=1, recompute_checksum=True)
    ok, decoded = deserialize_packet(mutated)
    assert not ok
    assert "length mismatch" in decoded["error"].lower()


# --- Sequence-level faults ---------------------------------------------------

def test_drop_packets_removes_only_specified_indices(packet_sequence):
    result = fi.drop_packets(packet_sequence, indices_to_drop=[1, 3])
    assert len(result) == 3
    assert result == [packet_sequence[0], packet_sequence[2], packet_sequence[4]]


def test_reorder_packets_applies_given_permutation(packet_sequence):
    result = fi.reorder_packets(packet_sequence, new_order=[2, 0, 1, 3, 4])
    assert result[0] == packet_sequence[2]
    assert result[1] == packet_sequence[0]
    assert result[2] == packet_sequence[1]


def test_reorder_packets_rejects_invalid_permutation(packet_sequence):
    with pytest.raises(ValueError):
        fi.reorder_packets(packet_sequence, new_order=[0, 0, 1, 2, 3])  # not a valid permutation


def test_duplicate_packets_inserts_adjacent_copy(packet_sequence):
    result = fi.duplicate_packets(packet_sequence, indices_to_duplicate=[1])
    assert len(result) == 6
    assert result[1] == packet_sequence[1]
    assert result[2] == packet_sequence[1]  # the duplicate
    assert result[3] == packet_sequence[2]  # sequence continues normally after


def test_duplicate_packets_all_duplicates_decode_correctly(packet_sequence):
    """
    Ties back to real behavior: a duplicated packet must still be a valid,
    independently-decodable packet, and this scenario is one the framework
    is specifically DESIGNED to tolerate (see this module's docstring on
    duplicate_packets).
    """
    result = fi.duplicate_packets(packet_sequence, indices_to_duplicate=[0, 2])
    for pkt in result:
        ok, decoded = deserialize_packet(pkt)
        assert ok


def test_with_delay_produces_schedule_tuples(packet_sequence):
    schedule = fi.with_delay(packet_sequence, delay_seconds=0.5)
    assert len(schedule) == len(packet_sequence)
    for pkt, delay in schedule:
        assert delay == 0.5
    assert [p for p, _ in schedule] == packet_sequence


def test_with_variable_delay_pairs_correctly(packet_sequence):
    delays = [0.1, 0.2, 0.3, 0.4, 0.5]
    schedule = fi.with_variable_delay(packet_sequence, delays)
    assert [d for _, d in schedule] == delays


def test_with_variable_delay_length_mismatch_raises(packet_sequence):
    with pytest.raises(ValueError):
        fi.with_variable_delay(packet_sequence, delays_seconds=[0.1, 0.2])  # wrong length