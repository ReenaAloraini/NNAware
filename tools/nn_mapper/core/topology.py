##Map a ModelSpec onto the NNAware library's node model.

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