"""Load a simple, transport-agnostic description of an offline-trained network.

v1 scope (matches the NNAware library's current capabilities):
  - fully connected layers only
  - inference only (no on-device training)
  - layer 0 is the input layer: {"size": N, "activation": "linear"}, no
    weights/bias — it has no predecessors, its value is seeded externally
    (a real sensor reading, or a fixed test value for a prototype like an
    AND gate).
  - every later layer: {"size": N, "activation": "relu"|"sigmoid"|"tanh"|"linear",
    "weights": [[...]] (N rows, one per neuron, each of length prev_size),
    "bias": [...] (N values, one per neuron)}

NNNodeConfig (as currently written in NNNode.h) has no bias field, so bias
is not represented here as a per-neuron scalar destined for the wire format.
topology.py turns each layer's bias vector into predecessor weights against
a synthetic constant-output node instead (see topology.py docstring) — this
module only validates and carries the bias values through.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

SUPPORTED_ACTIVATIONS = {"relu", "sigmoid", "tanh", "linear"}


@dataclass
class LayerSpec:
    size: int
    activation: str
    weights: Optional[List[List[float]]] = None  # [neuron_index][prev_neuron_index]
    bias: Optional[List[float]] = None


@dataclass
class ModelSpec:
    layers: List[LayerSpec]


def load_model(path: str) -> ModelSpec:
    with open(path) as f:
        raw = json.load(f)

    raw_layers = raw.get("layers")
    if not raw_layers or len(raw_layers) < 2:
        raise ValueError("Model must define at least an input layer and one computed layer.")

    layers: List[LayerSpec] = []
    prev_size = None
    for i, l in enumerate(raw_layers):
        if "size" not in l:
            raise ValueError(f"Layer {i}: missing 'size'.")
        size = l["size"]
        activation = l.get("activation", "linear")
        if activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Layer {i}: activation '{activation}' is not supported by NNActivationType "
                f"(supported: {sorted(SUPPORTED_ACTIVATIONS)})."
            )
        weights = l.get("weights")
        bias = l.get("bias")

        if i == 0:
            if weights is not None or bias is not None:
                raise ValueError("Layer 0 is the input layer and must not define weights/bias.")
        else:
            if weights is None or bias is None:
                raise ValueError(f"Layer {i}: fully-connected layers must define both 'weights' and 'bias'.")
            if len(weights) != size:
                raise ValueError(f"Layer {i}: expected {size} weight rows (one per neuron), got {len(weights)}.")
            for r, row in enumerate(weights):
                if len(row) != prev_size:
                    raise ValueError(
                        f"Layer {i}, neuron {r}: expected {prev_size} weights "
                        f"(fully connected to all of layer {i - 1}), got {len(row)}."
                    )
            if len(bias) != size:
                raise ValueError(f"Layer {i}: expected {size} bias values, got {len(bias)}.")

        layers.append(LayerSpec(size=size, activation=activation, weights=weights, bias=bias))
        prev_size = size

    return ModelSpec(layers=layers)