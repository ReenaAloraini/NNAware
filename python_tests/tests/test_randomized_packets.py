"""
randomized cross-language compatibility testing using a deterministic seed 
for reproducibility.

Also includes randomized corruption tests, confirming Python and C++ agree
on rejecting the same corrupted packets, reusing fault_injection.py rather
than reimplementing corruption logic here.

A FIXED SEED (RANDOM_SEED below) makes every run identical — a failure
here is always reproducible, never a flaky one-off, and re-running after a
fix confirms the SAME case that failed now passes, not just that some
different random case happened to pass this time.
"""
import random
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

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason=_skip_reason or "C++ executable not found")

RANDOM_SEED = 42     # fixed for reproducibility across every run and every machine
NUM_ROUNDTRIP_TRIALS = 40   # kept moderate: each trial makes 2 subprocess calls to the C++ executable
NUM_CORRUPTION_TRIALS = 20

FLOAT_TOLERANCE = 1e-4


def _random_valid_packet_params(rng: random.Random) -> dict:
    """
    Generates parameters for one random but VALID packet, respecting the
    actual protocol constraints confirmed from NNPacket.h/NNAddress.h:
    4-bit node/layer/cluster IDs, a payload count within
    NN_MAX_PAYLOAD_FLOATS, and ONLY defined packet type values (0/1/2) —
    deliberately excluding undefined type values, since Part 1 already
    documented type validation as a known protocol gap, not something to
    fold into "valid packet" generation here.
    """
    node_id = rng.randint(0, 15)
    layer_id = rng.randint(0, 15)
    cluster_id = rng.randint(0, 15)
    target_layer_id = rng.randint(0, 255)
    packet_type = rng.choice([0, 1, 2])
    sequence = rng.randint(0, 255)
    payload_count = rng.randint(0, 16)
    # Finite, moderate-magnitude floats — NaN/Inf/subnormals are
    # deliberately excluded as out of scope: neither implementation
    # currently defines behavior for them.
    values = [rng.uniform(-10000.0, 10000.0) for _ in range(payload_count)]
    return dict(node_id=node_id, layer_id=layer_id, cluster_id=cluster_id,
                target_layer_id=target_layer_id, packet_type=packet_type,
                sequence=sequence, values=values)


def _assert_floats_close(actual, expected):
    assert len(actual) == len(expected), f"length mismatch: {len(actual)} vs {len(expected)}"
    for a, e in zip(actual, expected):
        tol = max(FLOAT_TOLERANCE, 1e-6 * abs(e))
        assert abs(a - e) < tol, f"{a} != {e} (tolerance {tol})"


# --- Randomized round-trip: Python -> C++ -> C++ -> Python ------------------

@pytest.mark.parametrize("trial", range(NUM_ROUNDTRIP_TRIALS))
def test_G_randomized_round_trip_python_cpp_python(trial):
    rng = random.Random(RANDOM_SEED + trial)  # each trial gets its own reproducible sub-seed
    params = _random_valid_packet_params(rng)

    # Step 1: Python serialize
    py_bytes = build_full_packet(node_id=params["node_id"], layer_id=params["layer_id"],
                                  cluster_id=params["cluster_id"], target_layer_id=params["target_layer_id"],
                                  packet_type=params["packet_type"], sequence=params["sequence"],
                                  values=params["values"])
    py_hex = py_bytes.hex()

    # Step 2: C++ deserialize
    cpp_deser = _runner.deserialize(py_hex)
    assert cpp_deser["ok"], f"trial {trial}: C++ rejected a Python-built packet: {cpp_deser}\nparams={params}"
    assert cpp_deser["sequence"] == params["sequence"], f"trial {trial}: sequence not preserved"
    assert cpp_deser["target_layer_id"] == params["target_layer_id"], f"trial {trial}: target_layer_id not preserved"
    _assert_floats_close(cpp_deser["values"], params["values"])

    # Step 3: C++ re-serialize, from the fields IT decoded
    cpp_reser = _runner.serialize(source_address=cpp_deser["source_address"],
                                   target_layer_id=cpp_deser["target_layer_id"],
                                   packet_type=cpp_deser["type"], sequence=cpp_deser["sequence"],
                                   values=list(cpp_deser["values"]))
    assert cpp_reser["ok"], f"trial {trial}: C++ re-serialize failed: {cpp_reser}"

    # Bytes must be identical to the ORIGINAL Python bytes — a full round
    # trip through both implementations must be lossless and byte-stable.
    assert cpp_reser["bytes_hex"] == py_hex, (
        f"trial {trial}: round-trip bytes changed.\n"
        f"original Python: {py_hex}\nafter C++ round-trip: {cpp_reser['bytes_hex']}\nparams={params}"
    )

    # Step 4: Python deserialize (of C++'s re-serialized bytes)
    final_bytes = bytes.fromhex(cpp_reser["bytes_hex"])
    ok, decoded = deserialize_packet(final_bytes)
    assert ok, f"trial {trial}: Python rejected the fully round-tripped packet: {decoded}"
    assert decoded["node_id"] == params["node_id"], f"trial {trial}: node_id not preserved"
    assert decoded["layer_id"] == params["layer_id"], f"trial {trial}: layer_id not preserved"
    assert decoded["sequence"] == params["sequence"], f"trial {trial}: sequence not preserved (final)"
    _assert_floats_close(decoded["values"], params["values"])


# --- Randomized corruption: Python and C++ must agree on rejection ---------

@pytest.mark.parametrize("trial", range(NUM_CORRUPTION_TRIALS))
def test_G_randomized_corruption_rejected_by_both(trial):
    """
    Random valid packets, each corrupted with a fault type CONFIRMED (in
    test_packet_invalid.py / test_fault_injection.py) to be rejected by
    both implementations. Packet-type corruption is deliberately excluded
    here — Part 1 already documented that neither implementation currently
    validates packet type at all, so it isn't a "both must reject" case.
    """
    rng = random.Random(1000 + RANDOM_SEED + trial)
    params = _random_valid_packet_params(rng)
    py_bytes = build_full_packet(node_id=params["node_id"], layer_id=params["layer_id"],
                                  cluster_id=params["cluster_id"], target_layer_id=params["target_layer_id"],
                                  packet_type=params["packet_type"], sequence=params["sequence"],
                                  values=params["values"])

    has_payload = len(params["values"]) > 0
    fault_choices = ["checksum", "truncate", "extra_bytes"] + (["payload_byte"] if has_payload else [])
    fault_type = rng.choice(fault_choices)

    if fault_type == "checksum":
        mutated = fi.corrupt_checksum(py_bytes)
    elif fault_type == "truncate":
        cut_at = rng.randint(1, max(1, len(py_bytes) - 1))
        mutated = fi.truncate(py_bytes, cut_at)
    elif fault_type == "extra_bytes":
        garbage = bytes(rng.randint(0, 255) for _ in range(rng.randint(1, 5)))
        mutated = fi.append_extra_bytes(py_bytes, garbage)
    else:  # payload_byte
        idx = rng.randint(0, len(params["values"]) * 4 - 1)
        mutated = fi.corrupt_payload_byte(py_bytes, payload_byte_index=idx)

    py_ok, py_decoded = deserialize_packet(mutated)
    cpp_result = _runner.deserialize(mutated.hex())

    assert not py_ok, (
        f"trial {trial} ({fault_type}): Python UNEXPECTEDLY accepted a corrupted packet: {mutated.hex()}"
    )
    assert not cpp_result["ok"], (
        f"trial {trial} ({fault_type}): C++ UNEXPECTEDLY accepted a corrupted packet: {mutated.hex()}"
    )