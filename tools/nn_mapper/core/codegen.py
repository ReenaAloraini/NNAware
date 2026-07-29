"""Emit mapper output in three forms:

  1. network.json (model_to_network_json/write_network_json) -- THE
     RECOMMENDED PATH. Matches tools/nn_setup/generate_manifest.py's
     schema, so `python generate_manifest.py --network <this file>
     --hardware-ids hardware_ids.json` (or setup_tool.py's own
     `--network/--hardware-ids` dynamic mode) does the per-node topology
     expansion. Preferred over emitting devices.json directly: it avoids
     duplicating generate_manifest.py's expansion logic here, and gets
     backup-pair expansion and live HELLO-based dynamic device assignment
     for free. This module names the backup PAIRS (the "backups" field,
     from the app's per-layer selectors); generate_manifest.py derives
     every other backup field from them. Built
     straight from ModelSpec, not from node_layers -- topology.py's
     per-node expansion is still used locally for validation/simulation
     (constraints.py, simulate.py), just not for this export.
  2. devices.json (to_devices_json/write_devices_json) -- matches
     tools/nn_setup/device_manifest.py's schema exactly, for the case
     where you want a fully pre-assigned static manifest instead of going
     through generate_manifest.py (e.g. no dynamic pool, or debugging the
     mapper's own topology expansion directly).
  3. A direct C++ header per node (to_cpp_header) matching NNNodeConfig's
     field layout, for reading/debugging a node's config by eye, or for
     the fallback per-device-flash path if OTA provisioning isn't used.

The virtual input layer (layer 0) is NEVER emitted as a device and never
needs a hardware_id -- confirmed by test_reference_network.cpp: raw inputs
are directly-injected packets, not real NNNode instances. physical_nodes()
below is the one place that filters it out; every other function in this
module operates on whatever list it's given.

Terminal (last) layer uses successorLayerId = 255, matching
test_reference_network.cpp's own sentinel for "no further layer" -- NOT 0,
since 0 is the real, meaningful virtual-input layer id here. (network.json
doesn't carry this field at all -- generate_manifest.py derives it itself.)

devices.json requires a hardwareId per physical device that this mapper
has no way to know on its own -- write_devices_json() raises if any
compute-layer node's hardware_id hasn't been assigned yet. Assign them
with assign_hardware_ids() below, or by setting node["hardware_id"]
directly. network.json has no such requirement -- hardware assignment is
generate_manifest.py's/setup_tool.py's job for that path.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from core.model_io import ModelSpec

_ACTIVATION_ENUM = {
    "relu": "NNActivationType::RELU",
    "sigmoid": "NNActivationType::SIGMOID",
    "tanh": "NNActivationType::TANH",
    "linear": "NNActivationType::LINEAR",
}

# Matches device_manifest.py's ACTIVATION_TYPES keys exactly.
_ACTIVATION_NAME = {
    "relu": "RELU",
    "sigmoid": "SIGMOID",
    "tanh": "TANH",
    "linear": "LINEAR",
}

NN_TERMINAL_LAYER_SENTINEL = 255  # matches test_reference_network.cpp's O0Config.successorLayerId


def eligible_backup_targets(layer_size: int, node_id: int) -> List[int]:
    """Which siblings node_id may be assigned to back up: everyone in the layer
    but itself. The single source of truth for that rule -- the UI builds its
    dropdown from this, and _validate_layer_backups() checks against it, so the
    options offered and the options accepted can never drift apart."""
    if layer_size < 2:
        return []
    return [j for j in range(layer_size) if j != node_id]


def _validate_layer_backups(layer_index: int, layer_size: int, layer_backups: Dict[int, int]) -> None:
    """Rejects only what generate_manifest.py/device_manifest.py would reject anyway,
    but with a message naming the layer as the UI shows it. Deliberately does NOT
    reject 2-node layers or non-last backup targets: those shapes need the
    FailoverNode sketch's round-start gate and turn-stall release, not a ban."""
    # Checked first: in a one-node layer EVERY pairing is invalid, and saying so
    # once is friendlier than reporting whichever per-pair rule happens to trip.
    if layer_size < 2:
        raise ValueError(
            f"layer {layer_index + 1} has {layer_size} node(s); a backup must be a sibling "
            f"in the SAME layer, so this layer can't have one."
        )
    for backer, target in layer_backups.items():
        if not (0 <= backer < layer_size):
            raise ValueError(
                f"layer {layer_index + 1}: node {backer} is out of range for a layer "
                f"with {layer_size} node(s)."
            )
        if target not in eligible_backup_targets(layer_size, backer):
            raise ValueError(
                f"layer {layer_index + 1}: node {backer} can't back up node {target}. "
                f"Valid targets are {eligible_backup_targets(layer_size, backer)}."
            )


def model_to_network_json(
    model: ModelSpec,
    backups: Optional[Dict[int, Dict[int, int]]] = None,
    resend_grace_ms: Optional[Dict[int, int]] = None,
) -> dict:
    """
    Serializes a ModelSpec straight into network.json's shape -- no
    per-node expansion here, that's tools/nn_setup/generate_manifest.py's
    job (see module docstring). model.layers[0] (the input layer) becomes
    inputSize; every later layer becomes one network.json layer entry.

    backups: {layer_index: {backer_node_id: target_node_id}}, where layer_index
    indexes model.layers[1:] -- the same index as the emitted "layers" list, so
    the device-side layerId is layer_index + 1 (generate_manifest.py reserves
    layerId 0 for the virtual input feed). Backup duty is always sibling-to-
    sibling within one layer, which is why it lives on the layer entry.
    Everything else the device needs (backupTargetAddress, backupWeights,
    backupTargetBias, layerRosterMask, ...) is derived by generate_manifest.py
    from the target's own entry -- see its docstring.

    resend_grace_ms: {layer_index: ms}, emitted only for layers that actually
    have a backup. generate_manifest.py defaults it to 50 when absent; the UI
    passes a larger value, since 50ms is optimistic for WiFi multicast.

    Passing neither emits exactly what this function always emitted, so callers
    that don't care about fault tolerance are unaffected.
    """
    backups = backups or {}
    resend_grace_ms = resend_grace_ms or {}

    for layer_index in backups:
        if not (0 <= layer_index < len(model.layers) - 1):
            raise ValueError(
                f"backups references layer index {layer_index}, but this model has "
                f"{len(model.layers) - 1} compute layer(s)."
            )

    layers = []
    for layer_index, layer in enumerate(model.layers[1:]):
        entry = {
            "nodes": layer.size,
            "activationType": _ACTIVATION_NAME[layer.activation],
            "weights": [list(row) for row in layer.weights],
            "bias": list(layer.bias),
        }

        layer_backups = backups.get(layer_index)
        if layer_backups:
            _validate_layer_backups(layer_index, layer.size, layer_backups)
            # String keys match generate_manifest.py's documented shape; it reads
            # them back with int(backer_key).
            entry["backups"] = {str(b): t for b, t in sorted(layer_backups.items())}
            if layer_index in resend_grace_ms:
                entry["resendGraceMs"] = int(resend_grace_ms[layer_index])

        layers.append(entry)

    return {"inputSize": model.layers[0].size, "layers": layers}


def backup_detection_caveats(layer_size: int, layer_backups: Dict[int, int]) -> List[dict]:
    """
    Structural facts about backup pairings whose recovery is SLOWER, not broken.
    Nothing here blocks deployment. Returns tagged dicts rather than prose so the
    wording -- and any firmware-specific detail like a timeout duration -- stays
    in the view layer; this module has no business knowing what the sketch is
    called or how long it waits.

    Kinds:
      no_round_end_signal -- layer has exactly 2 nodes, so once you exclude the
        backer and its target there is no third sibling whose transmission could
        signal "the round finished". Detection has to be started some other way.
      target_not_last -- devices transmit in node-id order
        (precedingSiblingsMask == (1 << nodeId) - 1), so if the target isn't last
        the nodes after it stall in WAITING_FOR_TURN when it dies, and can only
        transmit once something releases their turn. Carries suggested_target,
        the pairing that avoids the stall entirely.
    """
    caveats: List[dict] = []
    if not layer_backups:
        return caveats

    if layer_size == 2:
        caveats.append({"kind": "no_round_end_signal"})
    for backer, target in sorted(layer_backups.items()):
        if target < layer_size - 1:
            caveats.append({
                "kind": "target_not_last",
                "backer": backer,
                "target": target,
                "suggested_target": layer_size - 1,
            })
    return caveats


def write_network_json(
    model: ModelSpec,
    path: str,
    backups: Optional[Dict[int, Dict[int, int]]] = None,
    resend_grace_ms: Optional[Dict[int, int]] = None,
) -> None:
    with open(path, "w") as f:
        json.dump(model_to_network_json(model, backups, resend_grace_ms), f, indent=2)
        f.write("\n")


def node_label(node: dict) -> str:
    return f"L{node['layer_id']}_N{node['node_id']}"


def all_nodes(node_layers: Dict[int, List[dict]]) -> List[dict]:
    """Flattened, layer-then-node_id ordered list of every node, INCLUDING the virtual input layer."""
    result = []
    for layer_id in sorted(node_layers.keys()):
        result.extend(node_layers[layer_id])
    return result


def physical_nodes(node_layers: Dict[int, List[dict]]) -> List[dict]:
    """Same ordering as all_nodes(), but excludes the virtual input layer -- this is
    the list that actually corresponds to physical devices needing provisioning."""
    return [n for n in all_nodes(node_layers) if not n["is_input"]]


def assign_hardware_ids(node_layers: Dict[int, List[dict]], hardware_ids: List[int]) -> None:
    """
    Convenience helper: assigns hardware_ids in order (ascending layer_id,
    then ascending node_id -- same order as physical_nodes()) to every
    PHYSICAL device. Input-layer nodes are skipped entirely -- they're
    never provisioned, so they never get a hardware_id. Mutates
    node_layers in place. Raises if the count doesn't match exactly -- a
    silent partial assignment would be worse than a loud failure here,
    since a missing hardwareId is only caught later by write_devices_json()
    otherwise.
    """
    nodes = physical_nodes(node_layers)
    if len(hardware_ids) != len(nodes):
        raise ValueError(
            f"got {len(hardware_ids)} hardware_ids but the topology needs {len(nodes)} "
            f"(one per PHYSICAL device -- the virtual input layer doesn't count)"
        )
    for node, hw_id in zip(nodes, hardware_ids):
        node["hardware_id"] = hw_id


def _device_entry(node: dict) -> dict:
    succ = node["successor_layer_id"]
    return {
        "hardwareId": f"0x{node['hardware_id']:016X}",
        "address": {
            "nodeId": node["node_id"],
            "layerId": node["layer_id"],
            "clusterId": node["cluster_id"],
            "reserved": node["reserved"],
        },
        "predecessorMask": node["predecessor_mask"],
        "precedingSiblingsMask": node["preceding_siblings_mask"],
        "successorLayerId": succ if succ is not None else NN_TERMINAL_LAYER_SENTINEL,
        "transmitSlot": node["transmit_slot"],
        "activationType": _ACTIVATION_NAME[node["activation"]],
        "weights": node["weights"],
        "predecessorLayerId": node["predecessor_layer_id"],
        "bias": node["bias"],
    }


def to_devices_json(node_layers: Dict[int, List[dict]]) -> list:
    nodes = physical_nodes(node_layers)
    missing = [node_label(n) for n in nodes if n.get("hardware_id") is None]
    if missing:
        raise ValueError(
            f"{len(missing)} device(s) have no hardware_id assigned: {missing}. "
            f"Call assign_hardware_ids() (or set node['hardware_id'] directly) before exporting."
        )
    return [_device_entry(n) for n in nodes]


def write_devices_json(node_layers: Dict[int, List[dict]], path: str) -> None:
    with open(path, "w") as f:
        json.dump(to_devices_json(node_layers), f, indent=2)


def to_cpp_header(node: dict) -> str:
    """
    NOTE: only meaningful for PHYSICAL (non-input) nodes -- the virtual
    input layer is never flashed or provisioned as a device, so calling
    this on an input-layer node dict produces a header describing hardware
    that doesn't exist. Kept callable for either kind purely for
    inspecting the mapper's internal representation while debugging.
    """
    label = node_label(node)
    weights = node["weights"]
    lines = [
        "// Auto-generated by nnaware-mapper — do not hand-edit.",
        "#pragma once",
        '#include "NNNode.h"',
        "",
    ]

    if node["is_input"]:
        lines += [
            "// NOTE: this is a VIRTUAL input node -- it has no physical device and is",
            "// never provisioned. Its value is injected directly onto the medium as a",
            "// DATA packet from this address by whatever orchestrates a run, matching",
            "// test_reference_network.cpp. This header is for inspection only.",
        ]

    weights_line = (
        f"static const float {label}_weights[{len(weights)}] = "
        f"{{{', '.join(f'{w:.8f}f' for w in weights)}}};"
        if weights
        else f"static const float* {label}_weights = nullptr;"
    )
    lines.append(weights_line)
    lines.append("")

    succ = node["successor_layer_id"]
    lines += [
        f"NNNodeConfig {label}_cfg = {{",
        f"    /* address */            {{{node['node_id']}, {node['layer_id']}, "
        f"{node['cluster_id']}, {node['reserved']}}},",
        f"    /* predecessorMask */    {node['predecessor_mask']:#06x},",
        f"    /* successorLayerId */   {succ if succ is not None else NN_TERMINAL_LAYER_SENTINEL},",
        f"    /* transmitSlot */       {node['transmit_slot']},",
        f"    /* activationType */     {_ACTIVATION_ENUM[node['activation']]},",
        f"    /* weights */            {label + '_weights' if weights else 'nullptr'},",
        "    /* backupWeights */      nullptr,",
        f"    /* weightCount */        {len(weights)},",
        f"    /* bias */               {node['bias']:.8f}f",
        "};",
        # Set explicitly rather than positionally: NNNodeConfig's aggregate-init
        # order has backup-role fields between weightCount/bias and
        # predecessorLayerId, so leaving it out of the braces above would
        # silently keep the struct's default (0) instead of this node's real
        # value on any layer where that's wrong.
        f"{label}_cfg.predecessorLayerId = {node['predecessor_layer_id']};",
    ]
    return "\n".join(lines)
