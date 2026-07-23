"""
Expands a PyTorch-style network description (network.json) plus a pool of
physical devices (hardware_ids.json) into the full device manifest
(devices.json) that setup_tool.py / device_manifest.py expect.

network.json shape:
{
  "inputSize": 2,          // width of the raw input feature vector. By
                            // default NOT hardware -- matches the project's
                            // original reference design (see
                            // test_reference_network.cpp): raw sensor values
                            // injected directly onto the medium from virtual
                            // node IDs, never run through a provisioned
                            // NNAware device. Only sets the first layer's
                            // predecessor count, UNLESS "inputValues" below
                            // is present.
  "inputValues": [1.0, 0.5], // OPTIONAL, one value per input node. When
                            // present, EVERY input node becomes a real
                            // physical device too (layerId 0, predecessorMask
                            // 0, activationType LINEAR) -- its value is
                            // pushed to it over a dedicated INPUT_VALUE setup
                            // message (NNSetupProtocol.h's NNInputValueMsg),
                            // deliberately separate from "bias", and consumed
                            // on-device via NNNode::seedOutput(). Omit this
                            // key entirely to keep the input layer virtual,
                            // exactly as before this field existed.
  "layers": [               // one entry per COMPUTE layer (hidden/output) --
                            // these ARE hardware, one physical device per node
    {
      "nodes": 3,           // like nn.Linear's out_features
      "activationType": "RELU",
      "weights": [          // one row per node, each row length == the
        [0.5, 1.0],          // previous layer's width (inputSize for the
        [-1.0, 2.0],         // first layer) -- like nn.Linear's in_features
        [0.3, -0.6]
      ],
      "bias": [-1.5, 0.0, 2.0]   // OPTIONAL, one value per node (like
                                  // nn.Linear's own bias vector). Omit the
                                  // whole key, or an individual layer's key,
                                  // to default every node in that layer to
                                  // 0.0 -- mirrors device_manifest.py's own
                                  // optional/default-0.0 handling of "bias",
                                  // so networks written before this field
                                  // existed still generate unchanged.
    },
    {
      "nodes": 1,
      "activationType": "SIGMOID",
      "weights": [[2.0, -0.5, 0.7]]
      // no "bias" here -- defaults to [0.0] for this layer's 1 node
    }
  ]
}

Optional per-layer "backups" field -- backup duty is only ever sibling-
to-sibling within the SAME layer (device_manifest.py enforces this), so
it lives on the layer, not globally:
    "backups": {"0": 1, "1": 0}
maps backer node index -> target node index it stands in for (here: node
0 backs up node 1, and node 1 backs up node 0 -- a mutual pair, the
project's own reference-network pattern). A node absent from the keys
has no backup duty. Every OTHER backup field (backupTargetAddress,
backupTargetPredecessorMask/PredecessorLayerId, backupTargetActivationType,
backupWeights, backupTargetBias, layerRosterMask) is derived automatically,
since siblings in one layer always share the same predecessors/activation,
and the target's weights/bias are just that target's own "weights"/"bias"
entry looked up by index. The one thing that ISN'T derivable --
resendGraceMs, how long to wait for a resend before falling back to the
substitute -- can be set per layer via "resendGraceMs" (defaults to 50,
matching test_reference_network.cpp).

hardware_ids.json is just a flat pool of physical devices' hardwareId
values -- written ONCE to match whatever NN_HARDWARE_ID you flashed into
each device's .ino, never edited again. It may hold MORE entries than the
network needs -- e.g. you own 5 devices but today's test network only
needs 3 -- in which case only the first N (matching node count) are used
and the rest sit idle in the pool. It is an error only if the network
needs MORE hardware nodes than the pool has. Without "inputValues",
inputSize does NOT count toward that total (the input feed is virtual,
not a physical device); WITH "inputValues" present, each input node DOES
count (one physical device each).

Hardware IDs are assigned to nodes in order: when "inputValues" is
present, its nodes are expanded FIRST (layerId 0, node 0 first, then node
1, ...), THEN network.json's layers/nodes flattened front-to-back (first
compute layer's node 0 first, then node 1, ..., then the next layer's
node 0, ...) -- all against hardware_ids.json in the order given there.
The front of the pool is used first, so if you care WHICH physical
devices sit idle, put those last in hardware_ids.json.

Positional/topology fields are derived as follows (see NNNode.h /
NNScheduler.h for how the device side actually consumes them). Hardware
layers are numbered starting at 1, NOT 0 -- address.layerId 0 is reserved
for the virtual raw-input feed, matching the reference network's own
convention:
  - address.nodeId/layerId: the node's index within its layer / (its
    layer's index in the `layers` list) + 1
  - predecessorMask: full bitmask over the previous layer's node count
    (inputSize for the first layer) -- nodes are fully connected
    layer-to-layer, the common case
  - predecessorLayerId: previous layer's layerId (0 -- the raw input
    feed -- for the first layer)
  - precedingSiblingsMask: node IDs before this one in the same layer
  - transmitSlot: this node's index within its layer
  - successorLayerId: next layer's layerId, even for the last layer --
    no device is ever addressed there, so the extra broadcast target is
    inert
  - bias: this node's own "bias" entry, or 0.0 if the layer omits "bias"
    entirely (PATCHED -- see network.json shape above)

Usage:
    python generate_manifest.py
    python generate_manifest.py --hardware-ids hardware_ids.json --network network.json --out devices.json
"""
import argparse
import json


class GenerateError(ValueError):
    pass


def build_devices(network: dict, hardware_ids: list) -> list:
    input_size = network["inputSize"]
    layers = network["layers"]

    # PATCHED: "inputValues" is an OPTIONAL top-level field -- one value per
    # input node, pushed to that node's own physical device via a dedicated
    # INPUT_VALUE setup message (NOT the "bias" field -- see NNSetupProtocol.h's
    # NNInputValueMsg). Its absence means the input layer stays virtual/not
    # provisioned, exactly as before this field existed -- unmodified networks
    # keep producing the same devices.json shape they always did.
    input_values = network.get("inputValues")
    if input_values is not None and len(input_values) != input_size:
        raise GenerateError(
            f"'inputValues' has {len(input_values)} value(s), expected {input_size} "
            f"(one per input node, matching 'inputSize')"
        )

    compute_total = sum(layer["nodes"] for layer in layers)
    total_nodes = compute_total + (input_size if input_values is not None else 0)
    if total_nodes > len(hardware_ids):
        raise GenerateError(
            f"network.json defines {total_nodes} hardware node(s) ({compute_total} across "
            f"its 'layers'{f' + {input_size} input node(s) via inputValues' if input_values is not None else ''}) "
            f"but hardware_ids.json only has {len(hardware_ids)} hardware ID(s) -- add more "
            f"physical devices to the pool, or shrink the network"
        )

    for layer_index, layer in enumerate(layers):
        if len(layer["weights"]) != layer["nodes"]:
            raise GenerateError(
                f"layers[{layer_index}]: 'nodes' is {layer['nodes']} but 'weights' has "
                f"{len(layer['weights'])} row(s) -- one weights row is required per node"
            )
        predecessor_count = input_size if layer_index == 0 else layers[layer_index - 1]["nodes"]
        for node_id, weights in enumerate(layer["weights"]):
            if len(weights) != predecessor_count:
                raise GenerateError(
                    f"layers[{layer_index}].weights[{node_id}] has {len(weights)} value(s), "
                    f"expected {predecessor_count} (the previous layer's width"
                    f"{' == inputSize' if layer_index == 0 else ''})"
                )

        # PATCHED: "bias" is optional per layer; when present it must have
        # exactly one entry per node, same shape rule as "weights".
        if "bias" in layer and len(layer["bias"]) != layer["nodes"]:
            raise GenerateError(
                f"layers[{layer_index}]: 'nodes' is {layer['nodes']} but 'bias' has "
                f"{len(layer['bias'])} value(s) -- one bias value is required per node if given at all"
            )

        for backer_key, target_id in layer.get("backups", {}).items():
            backer_id = int(backer_key)
            if not (0 <= backer_id < layer["nodes"]) or not (0 <= target_id < layer["nodes"]):
                raise GenerateError(
                    f"layers[{layer_index}].backups: {backer_key} -> {target_id} is out of range "
                    f"for a layer with {layer['nodes']} node(s)"
                )
            if backer_id == target_id:
                raise GenerateError(
                    f"layers[{layer_index}].backups: node {backer_id} cannot back up itself"
                )

    devices = []
    hw_id_iter = iter(hardware_ids)

    # PATCHED: real physical input-layer devices, expanded first (layerId 0
    # comes before the compute layers) -- one per inputValues entry.
    # predecessorMask=0 (no predecessors at all) and activationType=LINEAR so
    # a plain NNNode::execute() pass (sum = bias = 0.0, LINEAR is the
    # identity) is harmless if ever taken; the real value arrives via the
    # separate INPUT_VALUE message and NNNode::seedOutput(), not bias.
    if input_values is not None:
        for node_id, value in enumerate(input_values):
            devices.append({
                "hardwareId": next(hw_id_iter),
                "address": {"nodeId": node_id, "layerId": 0, "clusterId": 0, "reserved": 0},
                "predecessorMask": 0,
                "precedingSiblingsMask": (1 << node_id) - 1,
                "successorLayerId": 1,
                "transmitSlot": node_id,
                "activationType": "LINEAR",
                "weights": [],
                "predecessorLayerId": 0,
                "bias": 0.0,
                "inputValue": value,
            })

    for layer_index, layer in enumerate(layers):
        layer_id = layer_index + 1  # layerId 0 is reserved for the raw input feed
        predecessor_count = input_size if layer_index == 0 else layers[layer_index - 1]["nodes"]
        predecessor_mask = (1 << predecessor_count) - 1 if predecessor_count else 0
        predecessor_layer_id = layer_id - 1
        layer_roster_mask = (1 << layer["nodes"]) - 1
        resend_grace_ms = layer.get("resendGraceMs", 50)
        backups = {int(k): v for k, v in layer.get("backups", {}).items()}
        # PATCHED: defaults every node in the layer to 0.0 bias if "bias" is absent.
        layer_bias = layer.get("bias", [0.0] * layer["nodes"])

        for node_id, weights in enumerate(layer["weights"]):
            device = {
                "hardwareId": next(hw_id_iter),
                "address": {"nodeId": node_id, "layerId": layer_id, "clusterId": 0, "reserved": 0},
                "predecessorMask": predecessor_mask,
                "precedingSiblingsMask": (1 << node_id) - 1,
                "successorLayerId": layer_id + 1,
                "transmitSlot": node_id,
                "activationType": layer["activationType"],
                "weights": weights,
                "predecessorLayerId": predecessor_layer_id,
                "bias": layer_bias[node_id],  # PATCHED
            }

            if node_id in backups:
                target_id = backups[node_id]
                device["backupRole"] = {
                    "backupTargetAddress": {"nodeId": target_id, "layerId": layer_id, "clusterId": 0, "reserved": 0},
                    "backupTargetPredecessorMask": predecessor_mask,
                    "backupTargetActivationType": layer["activationType"],
                    "backupWeights": layer["weights"][target_id],
                    "resendGraceMs": resend_grace_ms,
                    "layerRosterMask": layer_roster_mask,
                    "backupTargetPredecessorLayerId": predecessor_layer_id,
                    "backupTargetBias": layer_bias[target_id],  # PATCHED
                }

            devices.append(device)
    return devices


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hardware-ids", default="hardware_ids.json",
                         help="Path to the flat pool of physical devices' hardwareIds (written once)")
    parser.add_argument("--network", default="network.json",
                         help="Path to the PyTorch-style network description (edit this whenever topology changes)")
    parser.add_argument("--out", default="devices.json",
                         help="Path to write the expanded device manifest for setup_tool.py")
    args = parser.parse_args()

    with open(args.hardware_ids) as f:
        hardware_ids = json.load(f)
    with open(args.network) as f:
        network = json.load(f)

    try:
        devices = build_devices(network, hardware_ids)
    except GenerateError as e:
        print(f"[generate_manifest] ERROR: {e}")
        return 1

    with open(args.out, "w") as f:
        json.dump(devices, f, indent=2)
        f.write("\n")

    print(f"[generate_manifest] wrote {len(devices)} device(s) across {len(network['layers'])} "
          f"layer(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
