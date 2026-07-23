"""
Setup-phase message (de)serialization -- mirrors the C++ structs and
opcodes in src/NNSetupProtocol.h field-for-field.

Each pack_* function returns the raw, little-endian, #pragma-pack(1)-
equivalent struct bytes (NOT wire bytes -- pass the result to
wire_format.build_packet() before sending). Each unpack_* function takes
the already word-reversed-back struct bytes that wire_format.parse_packet()
hands back.

PATCHED: pack_topology_info() gained a `bias` parameter and
pack_backup_role_info() gained a `backup_target_bias` parameter, mirroring
NNTopologyInfoMsg/NNBackupRoleInfoMsg's new fields in NNSetupProtocol.h.
"""
import struct

# Opcodes -- must match NNSetupOpcode in NNSetupProtocol.h exactly.
HELLO = 0x01
ASSIGN_ADDRESS = 0x02
TOPOLOGY_INFO = 0x03
WEIGHTS_CHUNK = 0x04
COMMIT_REQUEST = 0x05
COMMIT_REPLY = 0x06
ACK = 0x07
NACK = 0x08
START = 0x09
BACKUP_ROLE_INFO = 0x0A
BACKUP_WEIGHTS_CHUNK = 0x0B
INPUT_VALUE = 0x0C  # laptop -> device: push a seed input value (predecessorMask==0 devices only)

OPCODE_NAMES = {
    HELLO: "HELLO", ASSIGN_ADDRESS: "ASSIGN_ADDRESS", TOPOLOGY_INFO: "TOPOLOGY_INFO",
    WEIGHTS_CHUNK: "WEIGHTS_CHUNK", COMMIT_REQUEST: "COMMIT_REQUEST", COMMIT_REPLY: "COMMIT_REPLY",
    ACK: "ACK", NACK: "NACK", START: "START", BACKUP_ROLE_INFO: "BACKUP_ROLE_INFO",
    BACKUP_WEIGHTS_CHUNK: "BACKUP_WEIGHTS_CHUNK", INPUT_VALUE: "INPUT_VALUE",
}

START_MAGIC = 0x4E4E5354  # 'NNST', matches NNStartMsg's default

# Must match NN_WEIGHTS_CHUNK_MAX_FLOATS in NNSetupProtocol.h.
NN_WEIGHTS_CHUNK_MAX_FLOATS = 11

NN_SETUP_NACK_BAD_CHUNK_INDEX = 1
NN_SETUP_NACK_WRONG_STATE = 2


def compute_weights_checksum(values) -> int:
    """
    Predicts NNSetupAgent::handleCommitRequest's computedChecksum/
    backupChecksum: computeChecksum() runs over the device's NATIVE
    little-endian in-RAM float bytes, BEFORE any wire transform -- this is
    a completely separate checksum from the wire-level integrity checksum
    in wire_format.compute_checksum, computed over different bytes for a
    different purpose.
    """
    if not values:
        return 0
    return sum(struct.pack(f"<{len(values)}f", *values)) & 0xFF


def pack_hello(hardware_id: int) -> bytes:
    return struct.pack("<QI", hardware_id, 0)


def unpack_hello(data: bytes) -> dict:
    hardware_id, _reserved = struct.unpack("<QI", data)
    return {"hardware_id": hardware_id}


def pack_assign_address(hardware_id: int, address: int) -> bytes:
    return struct.pack("<QHH", hardware_id, address, 0)


def pack_topology_info(hardware_id: int, predecessor_mask: int, preceding_siblings_mask: int,
                        successor_layer_id: int, activation_type: int, weight_count: int,
                        transmit_slot: int, predecessor_layer_id: int, bias: float) -> bytes:
    # Field order must match NNTopologyInfoMsg exactly: ...predecessorLayerId,
    # THEN bias, THEN the 3-byte pad -- see NNSetupProtocol.h.
    return struct.pack("<QHHBBBBBf3x", hardware_id, predecessor_mask, preceding_siblings_mask,
                        successor_layer_id, activation_type, weight_count, transmit_slot,
                        predecessor_layer_id, bias)


def pack_weights_chunk(hardware_id: int, chunk_index: int, chunk_count: int, values) -> bytes:
    """Also used, unmodified, for BACKUP_WEIGHTS_CHUNK -- see NNWeightsChunkMsg's comment."""
    if len(values) > NN_WEIGHTS_CHUNK_MAX_FLOATS:
        raise ValueError(f"chunk carries {len(values)} values, max is {NN_WEIGHTS_CHUNK_MAX_FLOATS}")
    padded = list(values) + [0.0] * (NN_WEIGHTS_CHUNK_MAX_FLOATS - len(values))
    return struct.pack(f"<QBBBB{NN_WEIGHTS_CHUNK_MAX_FLOATS}f", hardware_id, chunk_index,
                        chunk_count, len(values), 0, *padded)


def pack_backup_role_info(hardware_id: int, backup_target_address: int,
                           backup_target_predecessor_mask: int, layer_roster_mask: int,
                           backup_target_activation_type: int, backup_weight_count: int,
                           resend_grace_ms: int, backup_target_predecessor_layer_id: int,
                           backup_target_bias: float) -> bytes:
    # Field order must match NNBackupRoleInfoMsg exactly: ...backupTargetPredecessorLayerId,
    # THEN backupTargetBias, THEN the 3-byte pad -- see NNSetupProtocol.h.
    return struct.pack("<QHHHBBIBf3x", hardware_id, backup_target_address, backup_target_predecessor_mask,
                        layer_roster_mask, backup_target_activation_type, backup_weight_count,
                        resend_grace_ms, backup_target_predecessor_layer_id, backup_target_bias)


def pack_input_value(hardware_id: int, value: float) -> bytes:
    """Mirrors NNInputValueMsg exactly: hardwareId, then value, no padding needed (12 bytes = 3 words)."""
    return struct.pack("<Qf", hardware_id, value)


def pack_commit_request(hardware_id: int) -> bytes:
    return struct.pack("<Q", hardware_id)


def unpack_commit_reply(data: bytes) -> dict:
    hardware_id, computed_checksum, backup_checksum, success, has_backup_role = \
        struct.unpack("<QBBBB", data)
    return {
        "hardware_id": hardware_id,
        "computed_checksum": computed_checksum,
        "backup_checksum": backup_checksum,
        "success": bool(success),
        "has_backup_role": bool(has_backup_role),
    }


def unpack_ack(data: bytes) -> dict:
    hardware_id, acked_sequence_number, acked_opcode, _reserved = struct.unpack("<QBBH", data)
    return {
        "hardware_id": hardware_id,
        "acked_sequence_number": acked_sequence_number,
        "acked_opcode": acked_opcode,
    }


def unpack_nack(data: bytes) -> dict:
    hardware_id, nacked_sequence_number, nacked_opcode, reason_code, _reserved = \
        struct.unpack("<QBBBB", data)
    return {
        "hardware_id": hardware_id,
        "nacked_sequence_number": nacked_sequence_number,
        "nacked_opcode": nacked_opcode,
        "reason_code": reason_code,
    }


def pack_start() -> bytes:
    return struct.pack("<I", START_MAGIC)
