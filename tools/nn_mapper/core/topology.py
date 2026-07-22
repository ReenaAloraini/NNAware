"""Map a ModelSpec onto the NNAware library's node model.

v1 mapping rules (see accompanying chat discussion for the reasoning):

  - One physical device per neuron IN A COMPUTE LAYER. The input layer
    (layer 0) is VIRTUAL, not hardware -- confirmed directly by
    test_reference_network.cpp: "x0/x1 have no NNNode of their own,
    matching the earlier design decision that raw sensor inputs are
    simulated as directly-injected packets, not NNNode instances." Layer-0
    node dicts still exist here (needed to compute layer-1's
    predecessorMask and to drive simulate.py's truth-table check), but
    codegen.py never emits them as devices or asks for a hardware_id.
  - A layer's node_layer_id equals its position in ModelSpec.layers
    (0 = virtual input layer, matching generate_manifest.py's own
    convention of reserving layerId 0 for the raw input feed).
  - Every neuron in layer i is fully connected to every neuron in layer i-1,
    so predecessorMask is simply "all node_ids in layer i-1", built at once
    rather than edge-by-edge.
  - Bias is a native field on NNNodeConfig (as of the library patch adding
    `float bias` + `sum = config.bias` in execute()) — each computed neuron
    carries its own bias value directly, no synthetic predecessor needed.
  - predecessor_layer_id is explicit per node (NNNodeConfig's cross-layer
    collision fix — a node must state which layer its predecessorMask's
    node IDs actually live in, not assume layer_id - 1). Always the
    previous node-layer here, since v1 is a straight feed-forward stack.
  - preceding_siblings_mask (feeds NNScheduler's NNWindowConfig) is derived
    from transmit_slot order within a layer: node nid's mask covers every
    node_id < nid in the same layer, matching transmit_slot == node_id.
  - successor_layer_id is None for the terminal (last) layer internally;
    codegen.py converts that to 255, matching test_reference_network.cpp's
    own terminal sentinel (O0's successorLayerId = 255) -- NOT 0, since 0
    is a real, meaningful layer id (the virtual input layer) here.
  - hardware_id is left as None for compute-layer nodes -- the mapper has
    no way to know which physical chip plays which role. Must be filled in
    (see codegen.assign_hardware_ids()) before codegen.write_devices_json()
    will produce a valid devices.json. Input-layer nodes never get one,
    since they're never provisioned as devices at all.
"""
from __future__ import annotations

from typing import Dict, List

from core.model_io import ModelSpec

NN_ADDRESS_FIELD_BITS = 4
NN_ADDRESS_FIELD_MAX = 1 << NN_ADDRESS_FIELD_BITS  # 16: max node_id/layer_id/cluster_id value + 1


def _predecessor_mask(count: int) -> int:
    """Bitmask covering node_ids 0..count-1."""
    return (1 << count) - 1


def build_topology(model: ModelSpec) -> Dict[int, List[dict]]:
    """Returns {node_layer_id: [node dict, ...]} covering every physical device needed."""
    node_layers: Dict[int, List[dict]] = {}

    input_size = model.layers[0].size
    node_layers[0] = [
        {
            "node_id": nid,
            "layer_id": 0,
            "cluster_id": 0,
            "reserved": 0,
            "predecessor_mask": 0,
            "predecessor_layer_id": 0,
            "preceding_siblings_mask": _predecessor_mask(nid),
            "successor_layer_id": 1 if len(model.layers) > 1 else None,
            "transmit_slot": nid,
            "activation": "linear",
            "weights": [],
            "weight_count": 0,
            "bias": 0.0,
            "is_input": True,
            "hardware_id": None,
        }
        for nid in range(input_size)
    ]

    for i in range(1, len(model.layers)):
        layer = model.layers[i]
        prev_size = model.layers[i - 1].size

        mask = _predecessor_mask(prev_size)  # real neurons 0..prev_size-1 only

        nodes = []
        for nid in range(layer.size):
            weights = list(layer.weights[nid])
            nodes.append(
                {
                    "node_id": nid,
                    "layer_id": i,
                    "cluster_id": 0,
                    "reserved": 0,
                    "predecessor_mask": mask,
                    "predecessor_layer_id": i - 1,
                    "preceding_siblings_mask": _predecessor_mask(nid),
                    "successor_layer_id": i + 1 if i + 1 < len(model.layers) else None,
                    "transmit_slot": nid,
                    "activation": layer.activation,
                    "weights": weights,
                    "weight_count": len(weights),
                    "bias": layer.bias[nid],
                    "is_input": False,
                    "hardware_id": None,
                }
            )
        node_layers[i] = nodes

    return node_layers


def device_count(node_layers: Dict[int, List[dict]]) -> int:
    """Physical devices only -- excludes the virtual input layer."""
    return sum(1 for nodes in node_layers.values() for n in nodes if not n["is_input"])