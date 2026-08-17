"""
Runtime inference client -- talks to REAL, ALREADY-PROVISIONED (post-START)
NNAware devices over UDP multicast (NNTransportUDPMulticast) to inject input
values and collect the final prediction, timing how long the real hardware
took to answer.
"""
from __future__ import annotations

import socket
import struct
import time
from typing import Dict, List, Tuple

HEADER_FMT = ">HBBBBBB"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 8
PACKET_TYPE_DATA = 0
PACKET_TYPE_CONTROL = 1
NN_MAX_PAYLOAD_FLOATS = 16

# Mirrored from NNFailover.h -- see its NN_DIAG_LAYER_GROUP and buildFailoverReport
# comments for why group 254 can never collide with a real layer's group, and why
# this report is the only way a failover is visible off-board.
NN_DIAG_LAYER_GROUP = 254
NN_FLAG_FAILOVER_SUBSTITUTE = 0x01
NN_FLAG_FAILOVER_TEARDOWN = 0x02


class InferenceTimeout(TimeoutError):
    """Not every expected output arrived. Subclasses TimeoutError so existing
    `except TimeoutError` handlers keep working unchanged.

    Carries the run's facts as attributes rather than only in the message, so a UI
    can word its own diagnosis (and suppress a redundant one when it is already
    rendering the failover events): `events` is whatever reports arrived before it
    gave up, `aborted` says a node was reported unrecoverable rather than the clock
    simply running out, `missing` lists the (layer_id, node_id) that never
    answered, and `node_outputs` is every OTHER physical device's own output that
    did arrive before the timeout (keyed by (layer_id, node_id)), so a UI can still
    show a partial per-device breakdown instead of nothing at all."""

    def __init__(self, message: str, events: List[dict], aborted: bool, missing: List[Tuple[int, int]],
                 node_outputs: Dict[Tuple[int, int], float]):
        super().__init__(message)
        self.events = events
        self.aborted = aborted
        self.missing = missing
        self.node_outputs = node_outputs

# The RUNNING phase deliberately uses a different port from the SETUP phase
# (4210): a device keeps its setup socket and its runtime multicast socket(s)
# alive at the same time, so they must not share a port. Must match the
# sketches' RUNTIME_PORT/RUN_PORT and the app's "Runtime port" setting.
DEFAULT_RUNTIME_PORT = 4211

# How long to keep listening after the last expected output, for failover reports
# still crossing the network. Comfortably over the firmware's 150ms retransmit gap,
# and short enough not to be felt in the UI.
REPORT_DRAIN_S = 0.25


def encode_address(node_id: int, layer_id: int, cluster_id: int = 0, reserved: int = 0) -> int:
    """Mirrors NNAddress.h's encodeAddress() exactly."""
    return ((node_id & 0x0F) << 12) | ((layer_id & 0x0F) << 8) | ((cluster_id & 0x0F) << 4) | (reserved & 0x0F)


def decode_address(encoded: int) -> Tuple[int, int, int, int]:
    """Mirrors NNAddress.h's decodeAddress() exactly."""
    return (encoded >> 12) & 0x0F, (encoded >> 8) & 0x0F, (encoded >> 4) & 0x0F, encoded & 0x0F


def _checksum(buf: bytes) -> int:
    """Mirrors NNPacket.h's computeChecksum(): additive sum of bytes, truncated to 8 bits."""
    return sum(buf) & 0xFF


def build_packet(node_id: int, layer_id: int, target_layer_id: int, sequence: int,
                 values: List[float], packet_type: int = PACKET_TYPE_DATA, flags: int = 0) -> bytes:
    """Matches transmitter.py's build_packet() field-for-field. packet_type/flags
    default to a plain DATA packet; they are parameters so tests can produce the
    other kinds this module parses (see parse_failover_report) without re-authoring
    the wire format -- the same reason packet_builder.py's build_full_packet() takes
    them."""
    source_address = encode_address(node_id, layer_id)
    payload_count = len(values)
    payload_bytes = struct.pack(f">{payload_count}f", *values)
    header_without_checksum = struct.pack(
        ">HBBBBB", source_address, target_layer_id, packet_type,
        sequence & 0xFF, payload_count, flags,
    )
    checksum = _checksum(header_without_checksum + b"\x00" + payload_bytes)
    header = struct.pack(
        HEADER_FMT, source_address, target_layer_id, packet_type,
        sequence & 0xFF, payload_count, flags, checksum,
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
        "sequence": sequence, "flags": flags, "values": values,
    }


def parse_failover_report(parsed: dict) -> dict | None:
    """Decodes a failover report (NNFailover.h's buildFailoverReport) into a tagged
    fact, or None if this packet isn't one.

    Deliberately returns tagged facts rather than prose -- the wording belongs in
    the UI, the same split codegen.backup_detection_caveats() already uses.

    Note the resend-request CONTROL packet (NNFailover.h's queueResendRequest) has
    an IDENTICAL shape -- sender in sourceAddress, subject encoded as payload[0] --
    and is told apart by two independent things: it leaves flags at 0, and it is
    addressed to a layer's own group, which this client never joins.

    The CONTROL check is not redundant with the caller's: a genuine substitute is a
    DATA packet carrying NN_FLAG_FAILOVER_SUBSTITUTE too, so dropping it here would
    make this function misread the very packets it exists to distinguish from.
    """
    if parsed["type"] != PACKET_TYPE_CONTROL or not parsed["values"]:
        return None
    flags = parsed["flags"]
    if flags & NN_FLAG_FAILOVER_TEARDOWN:
        kind = "unrecoverable"
    elif flags & NN_FLAG_FAILOVER_SUBSTITUTE:
        kind = "substituted"
    else:
        return None
    failed_node, failed_layer, _cluster, _reserved = decode_address(int(parsed["values"][0]))
    return {
        "kind": kind,
        "failed": (failed_layer, failed_node),          # the node that went silent
        "backup": (parsed["layer_id"], parsed["node_id"]),  # the node that stood in for it
    }


def layer_group(layer_id: int) -> str:
    """Mirrors NNTransportUDPMulticast::layerGroup(): 239.1.0.<layer>."""
    return f"239.1.0.{layer_id}"


def run_inference_on_devices(
    node_layers: Dict[int, List[dict]],
    input_values: List[float],
    port: int = DEFAULT_RUNTIME_PORT,
    timeout_s: float = 10.0,
    multicast_ttl: int = 2,
) -> Tuple[List[float], float, List[dict], Dict[Tuple[int, int], float]]:
    """
    Injects input_values onto the real network as DATA packets from the virtual
    input layer, over UDP multicast, and waits for the terminal layer's real
    device(s) to answer. Returns (output_values, elapsed_seconds, events,
    node_outputs), where elapsed_seconds is measured from "all inputs sent" to
    "all expected outputs received", events is the list of failover facts
    observed during the run (see parse_failover_report; empty on a healthy run),
    and node_outputs is EVERY physical device's own output seen along the way,
    keyed by (layer_id, node_id) -- not just the terminal layer's -- so the
    caller can show per-device results, not only the final prediction. Raises
    ValueError for a bad input count, InferenceTimeout (a TimeoutError) if not
    every expected output arrives.
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

    # Every physical node across every compute layer -- used to build the full
    # per-device report, even though only expected_keys (the terminal layer's)
    # gates when we're done, in the wait loop below.
    all_physical_nodes = [n for lid in compute_layer_ids for n in node_layers[lid] if not n["is_input"]]
    all_expected_keys = {(n["layer_id"], n["node_id"]) for n in all_physical_nodes}

    inject_group = layer_group(1)  # every input's successor is always the first compute layer

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, multicast_ttl)
    sock.bind(("", port))
    # A layer L's devices transmit to the group matching THEIR OWN
    # successorLayerId, which generate_manifest.py always sets to L + 1 -- so to
    # observe every physical layer's own output, not just the terminal one, join
    # all of those groups, plus the diagnostics group failover reports cross.
    listen_groups = {layer_group(lid + 1) for lid in compute_layer_ids}
    listen_groups.add(layer_group(NN_DIAG_LAYER_GROUP))
    for group in listen_groups:
        mreq = struct.pack("4sl", socket.inet_aton(group), socket.INADDR_ANY)
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
        # Keyed by identity because every report is retransmitted
        # NN_DATA_RETRANSMITS times; dict order preserves arrival order.
        events: Dict[tuple, dict] = {}
        aborted = False
        deadline = start + timeout_s

        def absorb(packet: bytes) -> None:
            """Files one datagram as either a failover report or an expected output."""
            nonlocal aborted
            parsed = parse_packet(packet)
            if not parsed:
                return
            if parsed["type"] == PACKET_TYPE_CONTROL:
                report = parse_failover_report(parsed)
                if report is not None:
                    events.setdefault((report["kind"], report["failed"], report["backup"]), report)
                    # A node AND its backup both failed: no output can ever arrive.
                    aborted = aborted or report["kind"] == "unrecoverable"
                return
            if parsed["type"] != PACKET_TYPE_DATA:
                return
            key = (parsed["layer_id"], parsed["node_id"])
            if key in all_expected_keys and parsed["values"]:
                outputs[key] = parsed["values"][0]

        while not expected_keys.issubset(outputs.keys()) and not aborted and time.monotonic() < deadline:
            try:
                data, _addr = sock.recvfrom(1024)
            except socket.timeout:
                continue
            absorb(data)
        elapsed = time.monotonic() - start

        # Drain briefly for reports still in flight. A substitute in a HIDDEN layer
        # keeps propagating forward after its report is sent, so the report can
        # legitimately still be arriving as the terminal outputs land -- without this
        # the client would exit first and the failover would go unreported.
        drain_until = time.monotonic() + REPORT_DRAIN_S
        sock.settimeout(REPORT_DRAIN_S)
        while time.monotonic() < drain_until:
            try:
                data, _addr = sock.recvfrom(1024)
            except socket.timeout:
                break
            absorb(data)

        if not expected_keys.issubset(outputs.keys()):
            missing = sorted(expected_keys - outputs.keys())
            raise InferenceTimeout(
                f"{len(missing)} device(s) never answered (layer_id, node_id): {missing}",
                list(events.values()), aborted, missing, dict(outputs),
            )

        ordered = [outputs[(n["layer_id"], n["node_id"])] for n in output_nodes]
        return ordered, elapsed, list(events.values()), dict(outputs)
    finally:
        sock.close()