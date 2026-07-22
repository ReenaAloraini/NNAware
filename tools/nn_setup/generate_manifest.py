"""
Expands a PyTorch-style network description (network.json) plus a pool of
physical devices (hardware_ids.json) into the full device manifest
(devices.json) that setup_tool.py / device_manifest.py expect.

network.json shape:
{
  "inputSize": 2,          // width of the raw input feature vector -- NOT
                            // hardware. Matches the project's own reference
                            // design (see test_reference_network.cpp): raw
                            // sensor values are injected directly onto the
                            // medium from virtual node IDs, never run
                            // through a provisioned NNAware device. Only
                            // sets the first layer's predecessor count.
  "layers": [               // one entry per COMPUTE layer (hidden/output) --
                            // these ARE hardware, one physical device per node
    {
      "nodes": 3,           // like nn.Linear's out_features
      "activationType": "RELU",
      "weights": [          // one row per node, each row length == the
        [0.5, 1.0],          // previous layer's width (inputSize for the
        [-1.0, 2.0],         // first layer) -- like nn.Linear's in_features
        [0.3, -0.6]
      ]
    },
    {
      "nodes": 1,
      "activationType": "SIGMOID",
      "weights": [[2.0, -0.5, 0.7]]
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
backupWeights, layerRosterMask) is derived automatically, since siblings
in one layer always share the same predecessors/activation, and the
target's weights are just that target's own "weights" row looked up by
index. The one thing that ISN'T derivable -- resendGraceMs, how long to
wait for a resend before falling back to the substitute -- can be set
per layer via "resendGraceMs" (defaults to 50, matching
test_reference_network.cpp).

hardware_ids.json is just a flat pool of physical devices' hardwareId
values -- written ONCE to match whatever NN_HARDWARE_ID you flashed into
each device's .ino, never edited again. It may hold MORE entries than
sum(layer["nodes"] for layer in layers) -- e.g. you own 5 devices but
today's test network only needs 3 -- in which case only the first N
(matching node count) are used and the rest sit idle in the pool. It is
an error only if the network needs MORE hardware nodes than the pool
has, since inputSize does NOT count (the input feed is not a physical
device) and there is nothing left to assign.

Hardware IDs are assigned to nodes in order, walking network.json's
layers/nodes flattened front-to-back (layer 0 node 0 first, then layer 0
node 1, ..., then layer 1 node 0, ...) against hardware_ids.json in the
order given there -- the front of the pool is used first, so if you care
WHICH physical devices sit idle, put those last in hardware_ids.json.

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

    total_nodes = sum(layer["nodes"] for layer in layers)
    if total_nodes > len(hardware_ids):
        raise GenerateError(
            f"network.json defines {total_nodes} hardware node(s) (across its 'layers', "
            f"not counting inputSize) but hardware_ids.json only has {len(hardware_ids)} "
            f"hardware ID(s) -- add more physical devices to the pool, or shrink the network"
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
    for layer_index, layer in enumerate(layers):
        layer_id = layer_index + 1  # layerId 0 is reserved for the raw input feed
        predecessor_count = input_size if layer_index == 0 else layers[layer_index - 1]["nodes"]
        predecessor_mask = (1 << predecessor_count) - 1 if predecessor_count else 0
        predecessor_layer_id = layer_id - 1
        layer_roster_mask = (1 << layer["nodes"]) - 1
        resend_grace_ms = layer.get("resendGraceMs", 50)
        backups = {int(k): v for k, v in layer.get("backups", {}).items()}

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
