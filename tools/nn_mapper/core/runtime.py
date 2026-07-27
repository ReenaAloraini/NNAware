"""
Runtime inference client -- talks to REAL, ALREADY-PROVISIONED (post-START)
NNAware devices over UDP multicast (NNTransportUDPMulticast) to inject input
values and collect the final prediction, timing how long the real hardware
took to answer.

Wire format is confirmed against TWO independent sources: NNPacket.h's own
serializePacket()/deserializePacket() (read directly from the library), and
the teammate's own receiver.py/transmitter.py scripts, already tested
against real hardware on a laptop. Both agree exactly: an 8-byte big-endian
header (sourceAddress uint16, targetLayerId uint8, type uint8,
sequenceNumber uint8, payloadCount uint8, flags uint8, checksum uint8)
followed by payloadCount big-endian floats; checksum = additive sum of every
byte (with the checksum byte itself zeroed) truncated to 8 bits. This module
reimplements that format directly (matching transmitter.py/receiver.py
field-for-field) rather than importing nn_setup's wire_format.py, since that
module's reverse_words() trick is specific to setup-phase CONTROL messages
(raw struct bytes reinterpreted as floats) -- runtime DATA packets carry
real floats and use the plain big-endian encoding transmitter.py/
receiver.py already use.

NNTransportUDPMulticast's group convention (see NNTransportUDPMulticast.h):
one multicast group per layer, 239.1.0.<layerId>. A device joins its OWN
layer's group and sends to the group matching a packet's targetLayerId. So:
  - Injecting a virtual input (layer 0) means sending to group <1> -- the
    first compute layer's group, since generate_manifest.py always numbers
    compute layers starting at 1 and every physical node's
    predecessorLayerId is the layer directly before it.
  - Collecting the final prediction means joining the group the TERMINAL
    layer's own successorLayerId points at. generate_manifest.py always
    sets successorLayerId = layer_id + 1, even for the last layer (that
    group is otherwise unused by any real device -- its own docstring
    calls this "the extra broadcast target is inert"), so this client must
    explicitly join group <last_layer_id + 1> to catch it.

Requires this machine to already be on the same network as the devices --
same assumption setup_tool.py makes. No WiFi credentials are needed here,
only on the device side.

NOTE: multicast networking itself could not be exercised end-to-end from
the sandbox this was developed in (no multicast-capable network device
available there) -- the packet build/parse logic below is verified against
the teammate's own tested transmitter.py/receiver.py, but the actual
group-join/send/receive behavior against real hardware has not been.
"""
from __future__ import annotations

import socket
import struct
import time
from typing import Dict, List, Tuple

HEADER_FMT = ">HBBBBBB"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 8
PACKET_TYPE_DATA = 0
NN_MAX_PAYLOAD_FLOATS = 16


def encode_address(node_id: int, layer_id: int, cluster_id: int = 0, reserved: int = 0) -> int:
    """Mirrors NNAddress.h's encodeAddress() exactly."""
    return ((node_id & 0x0F) << 12) | ((layer_id & 0x0F) << 8) | ((cluster_id & 0x0F) << 4) | (reserved & 0x0F)


def decode_address(encoded: int) -> Tuple[int, int, int, int]:
    """Mirrors NNAddress.h's decodeAddress() exactly."""
    return (encoded >> 12) & 0x0F, (encoded >> 8) & 0x0F, (encoded >> 4) & 0x0F, encoded & 0x0F


def _checksum(buf: bytes) -> int:
    """Mirrors NNPacket.h's computeChecksum(): additive sum of bytes, truncated to 8 bits."""
    return sum(buf) & 0xFF


def build_packet(node_id: int, layer_id: int, target_layer_id: int, sequence: int, values: List[float]) -> bytes:
    """Matches transmitter.py's build_packet() field-for-field (DATA packets only)."""
    source_address = encode_address(node_id, layer_id)
    payload_count = len(values)
    payload_bytes = struct.pack(f">{payload_count}f", *values)
    header_without_checksum = struct.pack(
        ">HBBBBB", source_address, target_layer_id, PACKET_TYPE_DATA,
        sequence & 0xFF, payload_count, 0,
    )
    checksum = _checksum(header_without_checksum + b"\x00" + payload_bytes)
    header = struct.pack(
        HEADER_FMT, source_address, target_layer_id, PACKET_TYPE_DATA,
        sequence & 0xFF, payload_count, 0, checksum,
    )
    return header + payload_bytes


def parse_packet(data: bytes) -> dict | None:
    """Matches receiver.py's deserialize_packet() validation order; returns None (not
    a bool/dict pair) on any failure, since this module only cares about the happy path."""
    if len(data) < HEADER_SIZE:
        return None
    received_checksum = data[7]
    buf = bytearray(data)
    buf[7] = 0
    if _checksum(bytes(buf)) != received_checksum:
        return None

    source_address, target_layer_id, ptype, sequence, payload_count, flags, _cs = \
        struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    if payload_count > NN_MAX_PAYLOAD_FLOATS:
        return None
    expected_total = HEADER_SIZE + payload_count * 4
    if len(data) != expected_total:
        return None

    values = struct.unpack(f">{payload_count}f", data[HEADER_SIZE:expected_total]) if payload_count else ()
    node_id, layer_id, cluster_id, _reserved = decode_address(source_address)
    return {
        "node_id": node_id, "layer_id": layer_id, "cluster_id": cluster_id,
        "target_layer_id": target_layer_id, "type": ptype,
        "sequence": sequence, "values": values,
    }


def layer_group(layer_id: int) -> str:
    """Mirrors NNTransportUDPMulticast::layerGroup(): 239.1.0.<layer>."""
    return f"239.1.0.{layer_id}"


def run_inference_on_devices(
    node_layers: Dict[int, List[dict]],
    input_values: List[float],
    port: int = 4210,
    timeout_s: float = 10.0,
    multicast_ttl: int = 2,
) -> Tuple[List[float], float]:
    """
    Injects input_values onto the real network as DATA packets from the virtual
    input layer, over UDP multicast, and waits for the terminal layer's real
    device(s) to answer. Returns (output_values, elapsed_seconds), where
    elapsed_seconds is measured from "all inputs sent" to "all expected outputs
    received". Raises ValueError for a bad input count, TimeoutError if not every
    expected output arrives within timeout_s.
    """
    input_nodes = [n for n in node_layers.get(0, []) if n["is_input"]]
    if len(input_values) != len(input_nodes):
        raise ValueError(f"expected {len(input_nodes)} input value(s), got {len(input_values)}")

    compute_layer_ids = [lid for lid, nodes in node_layers.items() if any(not n["is_input"] for n in nodes)]
    if not compute_layer_ids:
        raise ValueError("this topology has no physical devices to run inference on")
    last_layer_id = max(compute_layer_ids)
    output_nodes = [n for n in node_layers[last_layer_id] if not n["is_input"]]
    expected_keys = {(n["layer_id"], n["node_id"]) for n in output_nodes}

    inject_group = layer_group(1)  # every input's successor is always the first compute layer
    result_group = layer_group(last_layer_id + 1)  # matches generate_manifest.py's successorLayerId convention

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, multicast_ttl)
    sock.bind(("", port))
    mreq = struct.pack("4sl", socket.inet_aton(result_group), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.settimeout(0.5)

    try:
        seq = 0
        for node, value in zip(input_nodes, input_values):
            pkt = build_packet(node["node_id"], 0, 1, seq, [float(value)])
            sock.sendto(pkt, (inject_group, port))
            seq += 1
        start = time.monotonic()

        outputs: Dict[Tuple[int, int], float] = {}
        deadline = start + timeout_s
        while len(outputs) < len(expected_keys) and time.monotonic() < deadline:
            try:
                data, _addr = sock.recvfrom(1024)
            except socket.timeout:
                continue
            parsed = parse_packet(data)
            if not parsed or parsed["type"] != PACKET_TYPE_DATA:
                continue
            key = (parsed["layer_id"], parsed["node_id"])
            if key in expected_keys and parsed["values"]:
                outputs[key] = parsed["values"][0]
        elapsed = time.monotonic() - start

        if len(outputs) < len(expected_keys):
            missing = sorted(expected_keys - outputs.keys())
            raise TimeoutError(
                f"no response from {len(missing)} device(s) within {timeout_s:.1f}s "
                f"(layer_id, node_id): {missing}"
            )

        ordered = [outputs[(n["layer_id"], n["node_id"])] for n in output_nodes]
        return ordered, elapsed
    finally:
        sock.close()