"""
Loads and validates a device manifest (JSON) for the setup-phase
provisioning tool.

Manifest shape: a JSON array of device objects, one per physical device:

[
  {
    "hardwareId": "0x1122334455667788",
    "address": {"nodeId": 0, "layerId": 1, "clusterId": 0, "reserved": 0},
    "predecessorMask": 3,
    "precedingSiblingsMask": 0,
    "successorLayerId": 2,
    "transmitSlot": 0,
    "activationType": "RELU",
    "weights": [0.1, -0.2, 0.3],
    "predecessorLayerId": 0,
    "bias": -1.5,
    "backupRole": {
      "backupTargetAddress": {"nodeId": 1, "layerId": 1, "clusterId": 0, "reserved": 0},
      "backupTargetPredecessorMask": 3,
      "backupTargetActivationType": "RELU",
      "backupWeights": [0.15, -0.25],
      "resendGraceMs": 50,
      "layerRosterMask": 7,
      "backupTargetPredecessorLayerId": 0,
      "backupTargetBias": -1.5
    }
  }
]

"backupRole" is OPTIONAL -- its absence is exactly how a device is
provisioned with no backup duty (NNNodeConfig::hasBackupRole stays false,
its own default), matching NNSetupAgent's behavior of never sending
BACKUP_ROLE_INFO/BACKUP_WEIGHTS_CHUNK to such a device.

PATCHED: "bias" is now a recognized top-level device field, and
"backupTargetBias" is now a recognized backupRole field -- both mirror the
same additions to NNNodeConfig in NNNode.h. Both are OPTIONAL and default
to 0.0 when absent, matching NNNodeConfig::bias's own default in NNNode.h
-- required-with-no-default would reject every manifest/generator written
before this field existed (confirmed: it broke tools/nn_setup's own
generate_manifest.py output). A manifest that needs a real bias sets it
explicitly; one that doesn't is unaffected.

PATCHED: "inputValue" is now a recognized OPTIONAL top-level device field,
present only for a predecessorMask==0 device (a real physical input-layer
node) -- setup_tool.py sends it as a dedicated INPUT_VALUE setup message,
deliberately separate from "bias" (see NNSetupProtocol.h's NNInputValueMsg).
Absent for every other device; no default is applied here since setup_tool.py
itself only emits the packet when the key is present (None/absent both mean
"don't send it").
"""
import json

from wire_format import encode_address

ACTIVATION_TYPES = {"RELU": 0, "SIGMOID": 1, "TANH": 2, "LINEAR": 3}  # NNActivationType, NNActivation.h

REQUIRED_DEVICE_FIELDS = [
    "hardwareId", "address", "predecessorMask", "precedingSiblingsMask",
    "successorLayerId", "transmitSlot", "activationType", "weights",
    "predecessorLayerId",  # which layer predecessorMask's node IDs live in (NNNodeConfig cross-layer fix)
    # "bias" deliberately NOT required -- optional, defaults to 0.0 (see module docstring).
]
REQUIRED_ADDRESS_FIELDS = ["nodeId", "layerId", "clusterId", "reserved"]
REQUIRED_BACKUP_FIELDS = [
    "backupTargetAddress", "backupTargetPredecessorMask", "backupTargetActivationType",
    "backupWeights", "resendGraceMs", "layerRosterMask", "backupTargetPredecessorLayerId",
    # "backupTargetBias" deliberately NOT required -- optional, defaults to 0.0 (see module docstring).
]


class ManifestError(ValueError):
    pass


def _activation_value(name_or_value, context: str) -> int:
    if isinstance(name_or_value, str):
        if name_or_value not in ACTIVATION_TYPES:
            raise ManifestError(f"{context}: unknown activationType {name_or_value!r}, "
                                 f"must be one of {list(ACTIVATION_TYPES)}")
        return ACTIVATION_TYPES[name_or_value]
    if name_or_value not in ACTIVATION_TYPES.values():
        raise ManifestError(f"{context}: activationType {name_or_value!r} is not a legal NNActivationType (0-3)")
    return name_or_value


def _address_value(addr, context: str) -> int:
    missing = [f for f in REQUIRED_ADDRESS_FIELDS if f not in addr]
    if missing:
        raise ManifestError(f"{context}: address missing field(s) {missing}")
    return encode_address(addr["nodeId"], addr["layerId"], addr["clusterId"], addr["reserved"])


def parse_devices(raw: list) -> list:
    """
    Validates and encodes an already-loaded manifest (a plain Python list of
    device dicts, e.g. straight out of generate_manifest.build_devices() --
    no disk round-trip required) into the wire-ready device dicts
    setup_tool.py's provisioning loop expects. load_manifest() below is just
    this plus a JSON file read.
    """
    if not isinstance(raw, list):
        raise ManifestError("manifest must be a JSON array of device objects")

    devices = []
    for i, entry in enumerate(raw):
        context = f"device[{i}]"
        missing = [f for f in REQUIRED_DEVICE_FIELDS if f not in entry]
        if missing:
            raise ManifestError(f"{context}: missing required field(s) {missing}")

        hardware_id_raw = entry["hardwareId"]
        hardware_id = int(hardware_id_raw, 16) if isinstance(hardware_id_raw, str) else int(hardware_id_raw)

        own_address_fields = entry["address"]
        own_layer_id = own_address_fields.get("layerId")
        address = _address_value(own_address_fields, context)

        device = {
            "hardwareId": hardware_id,
            "address": address,
            "predecessorMask": entry["predecessorMask"],
            "precedingSiblingsMask": entry["precedingSiblingsMask"],
            "successorLayerId": entry["successorLayerId"],
            "transmitSlot": entry["transmitSlot"],
            "activationType": _activation_value(entry["activationType"], context),
            "weights": [float(w) for w in entry["weights"]],
            "predecessorLayerId": entry["predecessorLayerId"],
            "bias": float(entry.get("bias", 0.0)),
            "inputValue": float(entry["inputValue"]) if entry.get("inputValue") is not None else None,
            "backupRole": None,
        }

        backup = entry.get("backupRole")
        if backup is not None:
            missing = [f for f in REQUIRED_BACKUP_FIELDS if f not in backup]
            if missing:
                raise ManifestError(f"{context}.backupRole: missing required field(s) {missing}")

            backup_target_layer_id = backup["backupTargetAddress"].get("layerId")
            if backup_target_layer_id != own_layer_id:
                # Mirrors NNBackupStandby's own constructor assertion
                # (NNFailover.h): a backup relationship is per-layer,
                # sibling-to-sibling only. Caught here so a misconfigured
                # manifest fails BEFORE any packet is sent, instead of only
                # as a device-side assert deep into provisioning.
                raise ManifestError(
                    f"{context}.backupRole: backupTargetAddress.layerId ({backup_target_layer_id}) "
                    f"must equal this device's own address.layerId ({own_layer_id})"
                )

            device["backupRole"] = {
                "backupTargetAddress": _address_value(backup["backupTargetAddress"], f"{context}.backupRole"),
                "backupTargetPredecessorMask": backup["backupTargetPredecessorMask"],
                "backupTargetActivationType": _activation_value(
                    backup["backupTargetActivationType"], f"{context}.backupRole"),
                "backupWeights": [float(w) for w in backup["backupWeights"]],
                "resendGraceMs": backup["resendGraceMs"],
                "layerRosterMask": backup["layerRosterMask"],
                "backupTargetPredecessorLayerId": backup["backupTargetPredecessorLayerId"],
                "backupTargetBias": float(backup.get("backupTargetBias", 0.0)),
            }

        devices.append(device)

    return devices


def load_manifest(path: str) -> list:
    with open(path, "r") as f:
        raw = json.load(f)
    return parse_devices(raw)
