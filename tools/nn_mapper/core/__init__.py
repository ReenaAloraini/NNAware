"""
nnaware-mapper core: maps an offline-trained, fully-connected, inference-only
network onto the NNAware library's node model.

Pipeline (recommended): model_io.load_model() -> topology.build_topology()
-> constraints.validate_topology() [+ simulate.simulate() as a truth-table
check] -> codegen.write_network_json() (feeds tools/nn_setup/
generate_manifest.py, which does per-node expansion + backup pairing +
dynamic device assignment).

Pipeline (direct, static manifest): same through validate_topology(),
then codegen.assign_hardware_ids() -> codegen.write_devices_json() (feeds
tools/nn_setup/setup_tool.py --config directly, bypassing
generate_manifest.py).

simulate.simulate() is a pure-Python re-implementation of NNNode::execute()
used only by tests/test_and_gate.py as a correctness oracle -- not part of
the deployment path.

Re-exported here for convenience (`from core import build_topology`, etc.)
so callers don't need to know the internal module layout.
"""
from core.model_io import ModelSpec, LayerSpec, load_model
from core.topology import build_topology, device_count
from core.constraints import ConstraintError, validate_topology
from core.codegen import (
    NN_TERMINAL_LAYER_SENTINEL,
    all_nodes,
    physical_nodes,
    assign_hardware_ids,
    model_to_network_json,
    write_network_json,
    to_devices_json,
    write_devices_json,
    to_cpp_header,
    node_label,
)
from core.simulate import simulate as simulate_network

__all__ = [
    "ModelSpec",
    "LayerSpec",
    "load_model",
    "build_topology",
    "device_count",
    "ConstraintError",
    "validate_topology",
    "NN_TERMINAL_LAYER_SENTINEL",
    "all_nodes",
    "physical_nodes",
    "assign_hardware_ids",
    "model_to_network_json",
    "write_network_json",
    "to_devices_json",
    "write_devices_json",
    "to_cpp_header",
    "node_label",
    "simulate_network",
]