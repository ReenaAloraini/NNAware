"""Golden fixture: a 2-input AND gate mapped onto exactly 1 PHYSICAL device
(the output neuron). The 2 inputs are the virtual input layer -- confirmed
by test_reference_network.cpp to have no NNNode of their own -- and bias
is a native NNNodeConfig field, so neither consumes a hardware_id.

This is the mapper's correctness oracle — every pipeline change should
still reproduce the AND truth table before anything more complex is trusted.
"""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import codegen, constraints, model_io, simulate, topology

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "examples", "and_gate.json")

TRUTH_TABLE = [
    ((0.0, 0.0), 0.0),
    ((0.0, 1.0), 0.0),
    ((1.0, 0.0), 0.0),
    ((1.0, 1.0), 1.0),
]


def _build():
    model = model_io.load_model(FIXTURE)
    nodes = topology.build_topology(model)
    constraints.validate_topology(nodes, available_devices=1)
    return nodes


def test_device_count_and_addresses():
    nodes = _build()
    assert topology.device_count(nodes) == 1  # just the output neuron -- inputs are virtual
    assert len(nodes[0]) == 2  # the two virtual inputs still exist as node dicts...
    assert all(n["is_input"] for n in nodes[0])  # ...but are never physical devices
    assert len(nodes[1]) == 1  # the AND neuron
    output_node = nodes[1][0]
    assert output_node["predecessor_mask"] == 0b11  # inputs 0,1 only
    assert output_node["weight_count"] == 2
    assert output_node["bias"] == -1.5
    assert output_node["successor_layer_id"] is None  # terminal layer, internal representation


def test_truth_table():
    nodes = _build()
    for inputs, expected in TRUTH_TABLE:
        raw = simulate.simulate(nodes, list(inputs))[0]
        got = 1.0 if raw > 0.0 else 0.0
        assert got == expected, f"AND{inputs} -> raw={raw}, thresholded={got}, expected={expected}"


def test_devices_json_requires_hardware_ids():
    nodes = _build()
    try:
        codegen.to_devices_json(nodes)
        assert False, "expected a ValueError for missing hardware_id"
    except ValueError as e:
        assert "hardware_id" in str(e)


def test_assign_hardware_ids_skips_virtual_inputs():
    nodes = _build()
    try:
        codegen.assign_hardware_ids(nodes, [1, 2, 3])  # too many -- only 1 physical device exists
        assert False, "expected a ValueError: only 1 physical device, not 3"
    except ValueError as e:
        assert "1" in str(e)

    codegen.assign_hardware_ids(nodes, [42])  # correct count: just the output neuron
    assert nodes[1][0]["hardware_id"] == 42
    assert all(n.get("hardware_id") is None for n in nodes[0])  # inputs untouched


def test_devices_json_schema():
    nodes = _build()
    codegen.assign_hardware_ids(nodes, [3])  # just the output neuron
    devices = codegen.to_devices_json(nodes)
    assert len(devices) == 1  # the virtual input layer never appears here

    output_device = devices[0]
    assert output_device["address"]["layerId"] == 1
    assert output_device["hardwareId"] == "0x0000000000000003"
    assert output_device["predecessorMask"] == 0b11
    assert output_device["predecessorLayerId"] == 0
    assert output_device["activationType"] == "RELU"
    assert output_device["bias"] == -1.5
    assert output_device["weights"] == [1.0, 1.0]
    assert output_device["successorLayerId"] == codegen.NN_TERMINAL_LAYER_SENTINEL  # 255, not 0


def test_network_json_schema():
    model = model_io.load_model(FIXTURE)
    net = codegen.model_to_network_json(model)
    assert net["inputSize"] == 2
    assert len(net["layers"]) == 1  # only the compute layer -- input layer isn't in "layers"
    layer = net["layers"][0]
    assert layer["nodes"] == 1
    assert layer["activationType"] == "RELU"
    assert layer["weights"] == [[1.0, 1.0]]
    assert layer["bias"] == [-1.5]
    assert "backups" not in layer  # v1 scope: no automatic backup-pair generation


def test_codegen_cpp_header():
    nodes = _build()
    output_header = codegen.to_cpp_header(nodes[1][0])
    assert "NNActivationType::RELU" in output_header
    assert "0x0003" in output_header  # predecessorMask 0b11
    assert "-1.50000000f" in output_header  # native bias field
    assert "predecessorLayerId = 0;" in output_header
    assert "/* successorLayerId */   255," in output_header  # terminal sentinel, not 0

    input_header = codegen.to_cpp_header(nodes[0][0])
    assert "VIRTUAL input node" in input_header


if __name__ == "__main__":
    test_device_count_and_addresses()
    test_truth_table()
    test_devices_json_requires_hardware_ids()
    test_assign_hardware_ids_skips_virtual_inputs()
    test_devices_json_schema()
    test_network_json_schema()
    test_codegen_cpp_header()
    print("All tests passed.")