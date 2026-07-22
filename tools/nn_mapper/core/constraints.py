"""Validate a mapped topology against limits imposed by the NNAware library itself.

These aren't tunable mapper parameters — they're hard ceilings baked into the
library's wire format and structs:
  - NNAddress: node_id/layer_id/cluster_id/reserved are each 4-bit fields (0-15).
  - predecessorMask is a uint16_t -> at most 16 predecessors per node.
  - NN_MAX_PAYLOAD_FLOATS = 16 caps a single NNPacket's payload (not binding
    in v1: every node emits exactly one float).
"""
from __future__ import annotations

from typing import Dict, List, Optional

NN_ADDRESS_FIELD_MAX = 16
NN_MAX_PREDECESSORS = 16
NN_MAX_PAYLOAD_FLOATS = 16


class ConstraintError(Exception):
    pass


def validate_topology(node_layers: Dict[int, List[dict]], available_devices: Optional[int] = None) -> None:
    errors: List[str] = []

    if len(node_layers) > NN_ADDRESS_FIELD_MAX:
        errors.append(
            f"Network has {len(node_layers)} node-layers; layer_id is a 4-bit field "
            f"(max {NN_ADDRESS_FIELD_MAX})."
        )

    for layer_id, nodes in node_layers.items():
        # node_id ceiling applies to every node in the layer's address space,
        # including the virtual input layer's node dicts (layer 0 still needs
        # valid 4-bit node_ids even though it's never provisioned as hardware).
        if len(nodes) > NN_ADDRESS_FIELD_MAX:
            errors.append(
                f"Layer {layer_id}: {len(nodes)} nodes exceeds the 4-bit node_id ceiling "
                f"of {NN_ADDRESS_FIELD_MAX}."
            )
        for node in nodes:
            if node["weight_count"] > NN_MAX_PREDECESSORS:
                errors.append(
                    f"Layer {layer_id}, node {node['node_id']}: {node['weight_count']} predecessors "
                    f"exceeds the uint16_t predecessorMask ceiling of {NN_MAX_PREDECESSORS}."
                )

    # Physical devices only -- the virtual input layer (layer 0) is never
    # provisioned as hardware, confirmed by test_reference_network.cpp.
    total = sum(1 for nodes in node_layers.values() for n in nodes if not n["is_input"])
    if available_devices is not None and total > available_devices:
        errors.append(
            f"Network needs {total} physical device(s) (one per compute-layer neuron; "
            f"the input layer is virtual, not hardware); only {available_devices} available."
        )

    if errors:
        raise ConstraintError("\n".join(errors))