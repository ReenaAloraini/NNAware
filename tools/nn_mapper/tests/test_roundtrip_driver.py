"""
Standalone script (not pytest, run directly) that exercises the full
mapper -> real nn_setup -> real C++ round-trip chain on the AND-gate
golden fixture. Prints PASS/FAIL. Meant to be run manually while wiring
up the round-trip harness; pytest coverage can wrap this later.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAPPER_ROOT = os.path.dirname(HERE)
sys.path.insert(0, MAPPER_ROOT)

from core.model_io import load_model
from core.topology import build_topology, device_count
from core.roundtrip import run_round_trip_check

NN_SETUP_DIR = os.path.abspath(os.path.join(MAPPER_ROOT, "..", "nnaware-library-patch", "tools", "nn_setup"))
SRC_DIR = os.path.abspath(os.path.join(MAPPER_ROOT, "..", "nnaware-library-patch", "src"))

if __name__ == "__main__":
    model = load_model(os.path.join(MAPPER_ROOT, "examples", "and_gate.json"))
    node_layers = build_topology(model)
    n_devices = device_count(node_layers)
    hardware_ids = [0xAABBCCDD00000000 + i for i in range(n_devices)]

    print(f"device_count = {n_devices}, hardware_ids = {[hex(h) for h in hardware_ids]}")
    print(f"nn_setup_dir = {NN_SETUP_DIR}")
    print(f"src_dir      = {SRC_DIR}")

    passed, output = run_round_trip_check(model, node_layers, hardware_ids, SRC_DIR, nn_setup_dir=NN_SETUP_DIR)

    print(output)
    print("ROUND-TRIP CHECK:", "PASSED" if passed else "FAILED")
    sys.exit(0 if passed else 1)