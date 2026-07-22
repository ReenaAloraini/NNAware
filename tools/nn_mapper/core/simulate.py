"""Pure-Python re-implementation of NNNode::execute()'s semantics.

Used to verify a mapped topology produces the right numbers *before* ever
generating device configs or touching hardware — same "prove it on the
desktop first" discipline the lab manual uses for NNAddress/NNPacket/NNNode.
Not a substitute for the real g++ desktop test against NNNode.h; it's a
fast check on the mapper's own output.
"""
from __future__ import annotations

import math
from typing import Dict, List


def _activation(name: str, x: float) -> float:
    if name == "relu":
        return x if x > 0.0 else 0.0
    if name == "sigmoid":
        return 1.0 / (1.0 + math.exp(-x))
    if name == "tanh":
        return math.tanh(x)
    return x  # linear


def simulate(node_layers: Dict[int, List[dict]], input_values: List[float]) -> List[float]:
    """input_values: one value per layer-0 node, in node_id order."""
    outputs: Dict[tuple, float] = {}

    for layer_id in sorted(node_layers.keys()):
        for node in node_layers[layer_id]:
            if node["is_input"]:
                outputs[(layer_id, node["node_id"])] = input_values[node["node_id"]]
            else:
                mask = node["predecessor_mask"]
                vals = [
                    outputs[(layer_id - 1, sender_id)]
                    for sender_id in range(16)
                    if mask & (1 << sender_id)
                ]
                s = node["bias"] + sum(v * w for v, w in zip(vals, node["weights"]))
                outputs[(layer_id, node["node_id"])] = _activation(node["activation"], s)

    last_layer = max(node_layers.keys())
    return [outputs[(last_layer, n["node_id"])] for n in node_layers[last_layer]]