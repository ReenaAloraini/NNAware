"""
reusable, composable fault-injection helpers for testing packet handling
 under corruption, loss, reordering, duplication, and delay.
"""
from typing import List, Sequence, Tuple

from receiver import compute_checksum, CHECKSUM_INDEX, HEADER_SIZE


#Single-packet, byte-level faults 

def flip_byte(packet: bytes, offset: int) -> bytes:
    """
    Inverts every bit in the byte at `offset` (XOR 0xFF) — a full BYTE
    flip, matching the spec's wording ("flipping a selected byte"), not a
    single-bit flip.
    """
    if offset < 0 or offset >= len(packet):
        raise ValueError(f"offset {offset} out of range for a {len(packet)}-byte packet")
    mutated = bytearray(packet)
    mutated[offset] ^= 0xFF
    return bytes(mutated)


def corrupt_checksum(packet: bytes) -> bytes:
    """
    Sets the checksum byte to a value GUARANTEED not to match the packet's
    actual content, regardless of what checksum was already present —
    reuses the real compute_checksum() (from receiver.py, not
    reimplemented) to compute the correct value first, then deliberately
    picks a different one.
    """
    mutated = bytearray(packet)
    checksum_input = bytearray(packet)
    checksum_input[CHECKSUM_INDEX] = 0
    correct = compute_checksum(checksum_input)
    mutated[CHECKSUM_INDEX] = (correct + 1) & 0xFF
    return bytes(mutated)


def corrupt_payload_byte(packet: bytes, payload_byte_index: int = 0) -> bytes:
    """Flips one byte WITHIN the payload region (offset >= HEADER_SIZE)."""
    offset = HEADER_SIZE + payload_byte_index
    if offset >= len(packet):
        raise ValueError(
            f"payload_byte_index {payload_byte_index} is out of range for a packet "
            f"with only {max(0, len(packet) - HEADER_SIZE)} payload bytes"
        )
    return flip_byte(packet, offset)


def truncate(packet: bytes, length: int) -> bytes:
    """Cuts the packet down to exactly `length` bytes (or fewer, if the
    packet was already shorter)."""
    return packet[:length]


def append_extra_bytes(packet: bytes, extra: bytes = b"\xDE\xAD\xBE\xEF") -> bytes:
    """Appends trailing garbage after a (presumably valid) packet."""
    return packet + extra


def _fix_checksum(mutated: bytearray) -> bytearray:
    """Internal helper: recomputes and writes a CORRECT checksum for the
    given (already-mutated) packet bytes."""
    checksum_input = bytearray(mutated)
    checksum_input[CHECKSUM_INDEX] = 0
    mutated[CHECKSUM_INDEX] = compute_checksum(checksum_input)
    return mutated


def change_packet_type(packet: bytes, new_type: int, recompute_checksum: bool = False) -> bytes:
    """
    Overwrites the type byte (offset 3). By default leaves the checksum
    UNCHANGED, so the mutated packet will also fail checksum validation —
    simulating a genuine in-transit corruption of that specific byte. Pass
    recompute_checksum=True to instead simulate "this packet was always
    this type" (checksum recalculated to match), useful for testing
    type-specific handling in isolation from checksum rejection.
    """
    mutated = bytearray(packet)
    mutated[3] = new_type & 0xFF
    if recompute_checksum:
        mutated = _fix_checksum(mutated)
    return bytes(mutated)


def change_payload_count(packet: bytes, new_count: int, recompute_checksum: bool = False) -> bytes:
    """Overwrites the payloadCount byte (offset 5). See change_packet_type
    for the recompute_checksum behavior explanation."""
    mutated = bytearray(packet)
    mutated[5] = new_count & 0xFF
    if recompute_checksum:
        mutated = _fix_checksum(mutated)
    return bytes(mutated)


#Sequence-level faults (operate on a stream of packets) 

def drop_packets(packets: Sequence[bytes], indices_to_drop: Sequence[int]) -> List[bytes]:
    """Returns a new list with the packets at the given indices removed,
    simulating packet loss on a best-effort transport."""
    drop_set = set(indices_to_drop)
    return [p for i, p in enumerate(packets) if i not in drop_set]


def reorder_packets(packets: Sequence[bytes], new_order: Sequence[int]) -> List[bytes]:
    """
    Returns packets rearranged according to new_order — a list of original
    indices in their new sequence (e.g. [2, 0, 1] means "third packet
    first, then first, then second"). Simulates out-of-order delivery.
    Raises ValueError if new_order is not a valid permutation of the
    original indices, since a partial/invalid permutation would silently
    drop or duplicate packets — that's what drop_packets/duplicate_packets
    are for, not this function.
    """
    if sorted(new_order) != list(range(len(packets))):
        raise ValueError(f"new_order must be a permutation of range({len(packets)}), got {new_order}")
    return [packets[i] for i in new_order]


def duplicate_packets(packets: Sequence[bytes], indices_to_duplicate: Sequence[int]) -> List[bytes]:
    """
    Returns a new list where each packet at an index in
    indices_to_duplicate appears twice, immediately adjacent to the
    original. Simulates a receiver hearing the same broadcast more than
    once — a real, EXPECTED scenario for this framework's actual
    broadcast-style transports (BLE advertising, UDP broadcast, MQTT),
    which the framework is designed to tolerate via idempotent processing
    (see NNInputBuffer's overwrite-not-error behavior).
    """
    dup_set = set(indices_to_duplicate)
    result = []
    for i, p in enumerate(packets):
        result.append(p)
        if i in dup_set:
            result.append(p)
    return result


def with_delay(packets: Sequence[bytes], delay_seconds: float) -> List[Tuple[bytes, float]]:
    """
    Returns a SCHEDULE: a list of (packet, delay_before_send) tuples, all
    using the same fixed delay. Does NOT itself sleep or send anything —
    intentionally just data, so a future network simulator can consume the
    schedule and decide how to actually enact the delay (asyncio,
    threading, a simulated clock, ...) rather than this helper blocking
    the calling test.
    """
    return [(p, delay_seconds) for p in packets]


def with_variable_delay(packets: Sequence[bytes], delays_seconds: Sequence[float]) -> List[Tuple[bytes, float]]:
    """Same as with_delay, but a different delay per packet."""
    if len(delays_seconds) != len(packets):
        raise ValueError(
            f"delays_seconds must have the same length as packets "
            f"({len(delays_seconds)} != {len(packets)})"
        )
    return list(zip(packets, delays_seconds))