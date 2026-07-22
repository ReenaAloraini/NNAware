"""
NNAware setup-phase provisioning tool.

Run from a laptop/PC on the same network as a set of unconfigured NNAware
devices to assign each one an address, its topology, its weights (and
optionally a backup role + backup weights, see NNFailover.h), verify the
result, and finally tell everyone to go live. Drives the exact opcode
sequence implemented by NNSetupAgent in src/NNSetupProtocol.h.

Usage:
    python setup_tool.py --config devices.json
    python setup_tool.py --config devices.json --broadcast-addr 192.168.1.255
    python setup_tool.py --config devices.json --dry-run --dump-hex

See device_manifest.py for the devices.json schema.

PATCHED: build_device_packets() now passes device["bias"] into
pack_topology_info() and backup["backupTargetBias"] into
pack_backup_role_info(), matching both functions' new parameters.
"""
import argparse
import socket
import time

import setup_messages
import wire_format
from device_manifest import load_manifest, ManifestError


def build_device_packets(device: dict, seq_start: int):
    """
    Builds the full ordered list of (opcode, wire_bytes) packets for one
    device, without sending anything -- used both by the real provisioning
    loop and by --dry-run. Sequence numbers are assigned starting at
    seq_start and wrap at 256, mirroring NNSetupAgent's own uint8_t counter
    semantics on the device side (though the device does not care what
    sequence numbers the laptop picks -- it only ever echoes them back).
    """
    packets = []
    seq = seq_start

    def emit(opcode, msg_bytes):
        nonlocal seq
        packets.append((opcode, seq, wire_format.build_packet(opcode, seq, msg_bytes)))
        seq = (seq + 1) & 0xFF

    hw_id = device["hardwareId"]
    emit(setup_messages.ASSIGN_ADDRESS, setup_messages.pack_assign_address(hw_id, device["address"]))
    emit(setup_messages.TOPOLOGY_INFO, setup_messages.pack_topology_info(
        hw_id, device["predecessorMask"], device["precedingSiblingsMask"],
        device["successorLayerId"], device["activationType"], len(device["weights"]),
        device["transmitSlot"], device["predecessorLayerId"], device["bias"]))

    backup = device["backupRole"]
    if backup is not None:
        emit(setup_messages.BACKUP_ROLE_INFO, setup_messages.pack_backup_role_info(
            hw_id, backup["backupTargetAddress"], backup["backupTargetPredecessorMask"],
            backup["layerRosterMask"], backup["backupTargetActivationType"],
            len(backup["backupWeights"]), backup["resendGraceMs"],
            backup["backupTargetPredecessorLayerId"], backup["backupTargetBias"]))

    for opcode, values in ((setup_messages.WEIGHTS_CHUNK, device["weights"]),
                           (setup_messages.BACKUP_WEIGHTS_CHUNK, backup["backupWeights"] if backup else None)):
        if values is None:
            continue
        chunk_count = max(1, -(-len(values) // setup_messages.NN_WEIGHTS_CHUNK_MAX_FLOATS))
        for chunk_index in range(chunk_count):
            start = chunk_index * setup_messages.NN_WEIGHTS_CHUNK_MAX_FLOATS
            chunk_values = values[start:start + setup_messages.NN_WEIGHTS_CHUNK_MAX_FLOATS]
            emit(opcode, setup_messages.pack_weights_chunk(hw_id, chunk_index, chunk_count, chunk_values))

    emit(setup_messages.COMMIT_REQUEST, setup_messages.pack_commit_request(hw_id))
    return packets


def send_and_wait(sock, target_addr, packet: bytes, hardware_id: int, expected_opcode: int,
                   seq: int, retries: int, timeout: float):
    """
    Sends `packet` up to (retries + 1) times, waiting up to `timeout`
    seconds per attempt for a matching reply from `hardware_id`:
      - if expected_opcode == COMMIT_REQUEST, waits for a COMMIT_REPLY
      - otherwise waits for an ACK (acked_opcode/acked_sequence_number
        match) or a NACK (returned immediately, no point retrying a chunk
        index the device has already told us is out of range)

    Returns (ok, reply_dict) -- reply_dict is None on total timeout.
    """
    for _attempt in range(retries + 1):
        sock.sendto(packet, target_addr)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sock.settimeout(remaining)
            try:
                data, _from = sock.recvfrom(256)
            except socket.timeout:
                break

            ok, opcode, recv_seq, msg_bytes = wire_format.parse_packet(data)
            if not ok:
                continue

            if expected_opcode == setup_messages.COMMIT_REQUEST and opcode == setup_messages.COMMIT_REPLY:
                reply = setup_messages.unpack_commit_reply(msg_bytes)
                if reply["hardware_id"] == hardware_id:
                    return True, reply
            elif opcode == setup_messages.ACK:
                reply = setup_messages.unpack_ack(msg_bytes)
                if (reply["hardware_id"] == hardware_id and reply["acked_opcode"] == expected_opcode
                        and reply["acked_sequence_number"] == seq):
                    return True, reply
            elif opcode == setup_messages.NACK:
                reply = setup_messages.unpack_nack(msg_bytes)
                if (reply["hardware_id"] == hardware_id and reply["nacked_opcode"] == expected_opcode
                        and reply["nacked_sequence_number"] == seq):
                    return False, reply
    return False, None


def provision_device(sock, target_addr, device: dict, args) -> bool:
    hw_id = device["hardwareId"]
    print(f"[setup] provisioning 0x{hw_id:016X} ...")

    packets = build_device_packets(device, seq_start=0)
    checksum_expected = setup_messages.compute_weights_checksum(device["weights"])
    backup = device["backupRole"]
    backup_checksum_expected = (
        setup_messages.compute_weights_checksum(backup["backupWeights"]) if backup else 0
    )

    for opcode, seq, packet in packets:
        expected = setup_messages.COMMIT_REQUEST if opcode == setup_messages.COMMIT_REQUEST else opcode
        ok, reply = send_and_wait(sock, target_addr, packet, hw_id, expected, seq, args.retries, args.timeout)

        if opcode == setup_messages.COMMIT_REQUEST:
            if not ok or reply is None:
                print(f"[setup] 0x{hw_id:016X}: FAILED -- no COMMIT_REPLY after "
                      f"{args.retries + 1} attempt(s)")
                return False
            if not reply["success"]:
                print(f"[setup] 0x{hw_id:016X}: FAILED -- device reported COMMIT failure "
                      f"(hasBackupRole={reply['has_backup_role']})")
                return False
            if reply["computed_checksum"] != checksum_expected:
                print(f"[setup] 0x{hw_id:016X}: FAILED -- primary weights checksum mismatch "
                      f"(device={reply['computed_checksum']}, expected={checksum_expected})")
                return False
            if backup is not None and reply["backup_checksum"] != backup_checksum_expected:
                print(f"[setup] 0x{hw_id:016X}: FAILED -- backup weights checksum mismatch "
                      f"(device={reply['backup_checksum']}, expected={backup_checksum_expected})")
                return False
            print(f"[setup] 0x{hw_id:016X}: CONFIGURED (checksum OK"
                  f"{', backup checksum OK' if backup is not None else ''})")
            return True

        if not ok:
            reason = "NACK" if reply is not None else "no ACK"
            print(f"[setup] 0x{hw_id:016X}: FAILED at {setup_messages.OPCODE_NAMES[opcode]} ({reason})")
            return False

    return False  # unreachable -- COMMIT_REQUEST is always the last packet


def discover(sock, manifest_hw_ids: set, hello_window: float) -> set:
    print(f"[setup] listening for HELLO for {hello_window:.1f}s ...")
    seen = set()
    deadline = time.monotonic() + hello_window
    sock.settimeout(0.2)
    while time.monotonic() < deadline:
        try:
            data, _from = sock.recvfrom(256)
        except socket.timeout:
            continue
        ok, opcode, _seq, msg_bytes = wire_format.parse_packet(data)
        if not ok or opcode != setup_messages.HELLO:
            continue
        hw_id = setup_messages.unpack_hello(msg_bytes)["hardware_id"]
        if hw_id not in seen:
            seen.add(hw_id)
            tag = "in manifest" if hw_id in manifest_hw_ids else "NOT in manifest, ignoring"
            print(f"[setup] HELLO from 0x{hw_id:016X} ({tag})")
    missing = manifest_hw_ids - seen
    for hw_id in missing:
        print(f"[setup] WARNING: 0x{hw_id:016X} never announced HELLO -- will still attempt provisioning")
    return seen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the device manifest JSON file")
    parser.add_argument("--broadcast-addr", default="255.255.255.255",
                         help="Use 255.255.255.255 for same-machine testing, or the real subnet "
                              "broadcast address (e.g. 192.168.1.255) to reach real hardware")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--retries", type=int, default=5, help="Retries per message before giving up on a device")
    parser.add_argument("--timeout", type=float, default=0.5, help="Seconds to wait for a reply per attempt")
    parser.add_argument("--hello-window", type=float, default=10.0,
                         help="Seconds to listen for HELLO before starting provisioning")
    parser.add_argument("--dry-run", action="store_true",
                         help="Build every packet but don't open a socket or send anything")
    parser.add_argument("--dump-hex", action="store_true",
                         help="Print the hex bytes of every packet built (for cross-checking "
                              "against deserializePacket()/NNSetupAgent in a C++ test harness)")
    args = parser.parse_args()

    try:
        devices = load_manifest(args.config)
    except (ManifestError, OSError) as e:
        print(f"[setup] ERROR loading manifest: {e}")
        return 1

    if args.dry_run:
        for device in devices:
            hw_id = device["hardwareId"]
            print(f"[setup] (dry-run) device 0x{hw_id:016X}:")
            for opcode, seq, packet in build_device_packets(device, seq_start=0):
                name = setup_messages.OPCODE_NAMES[opcode]
                line = f"  seq={seq:3d} {name:<22}"
                if args.dump_hex:
                    line += f" {packet.hex()}"
                print(line)
        return 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", args.port))
    target_addr = (args.broadcast_addr, args.port)

    manifest_hw_ids = {d["hardwareId"] for d in devices}
    discover(sock, manifest_hw_ids, args.hello_window)

    results = {}
    for device in devices:
        results[device["hardwareId"]] = provision_device(sock, target_addr, device, args)

    configured = [hw_id for hw_id, ok in results.items() if ok]
    failed = [hw_id for hw_id, ok in results.items() if not ok]

    if failed:
        print(f"\n[setup] {len(failed)}/{len(devices)} device(s) FAILED -- not sending START:")
        for hw_id in failed:
            print(f"  0x{hw_id:016X}")
        print("[setup] fix the failed device(s) and re-run before broadcasting START.")
        return 1

    print(f"\n[setup] all {len(configured)} device(s) CONFIGURED -- broadcasting START x3 ...")
    start_packet = wire_format.build_packet(setup_messages.START, 0, setup_messages.pack_start())
    for _ in range(3):
        sock.sendto(start_packet, target_addr)
        time.sleep(0.2)

    print("[setup] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

