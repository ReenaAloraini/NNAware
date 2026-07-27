"""
Desktop "hardware round-trip" check: proves the full chain --
trained model -> mapper -> network.json -> REAL generate_manifest.py ->
REAL setup_tool.py packet-building -> REAL device-side NNSetupAgent
(NNSetupProtocol.h) assembling NNNodeConfig -> REAL NNNode/NNScheduler
inference -- reproduces the model's own predictions, without ever
touching a physical Wio Terminal.

This is NOT a substitute for flashing real hardware. It IS the same
"prove it on the desktop first" discipline the project already uses
everywhere else (NNAddress/NNPacket/NNNode's own g++ tests,
test_reference_network.cpp): every byte that would cross the wire to a
real device is built by the REAL Python provisioning code and consumed by
the REAL C++ device-side code, compiled and run for real. What it can't
catch: anything specific to actual radio behavior, timing, or the
physical NNTransportBLE/UDP implementations -- NNTransportLoopback stands
in for those here, exactly as it does in the library's own tests.

Requires tools/nn_setup/ to be an importable sibling directory (the real
repo layout: tools/nn_mapper/ and tools/nn_setup/ side by side) -- see
_import_nn_setup(). Also requires a g++ toolchain and a local copy of
src/ (the patched NNNode.h/NNFailover.h/NNSetupProtocol.h plus their
unchanged dependencies) to actually compile and run the generated test.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.model_io import ModelSpec
from core.codegen import model_to_network_json
from core.simulate import simulate_all

TOLERANCE = 0.0001


def _import_nn_setup(nn_setup_dir: Optional[str] = None):
    """
    Imports the REAL, currently-deployed tools/nn_setup modules (not a
    reimplementation) -- generate_manifest.build_devices(),
    device_manifest.parse_devices(), setup_tool.build_device_packets(),
    setup_messages, wire_format. Defaults to the sibling tools/nn_setup
    directory relative to this repo's real layout; pass nn_setup_dir
    explicitly if this package is laid out differently.
    """
    if nn_setup_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))  # .../tools/nn_mapper/core
        nn_setup_dir = os.path.abspath(os.path.join(here, "..", "..", "nn_setup"))
    if nn_setup_dir not in sys.path:
        sys.path.insert(0, nn_setup_dir)

    import generate_manifest
    import device_manifest
    import setup_messages
    import setup_tool
    import wire_format

    return generate_manifest, device_manifest, setup_messages, setup_tool, wire_format


@dataclass
class DeviceTestVector:
    label: str
    hardware_id: int
    packets_hex: List[str]           # ordered wire bytes (hex), setup packets + trailing START
    node_id: int
    layer_id: int
    predecessor_layer_id: int
    predecessor_mask: int
    address_expected: Tuple[int, int, int, int]   # nodeId, layerId, clusterId, reserved
    predecessor_mask_expected: int
    preceding_siblings_mask: int      # sibling markers to inject so WAITING_FOR_TURN can clear
    successor_layer_id_expected: int
    transmit_slot_expected: int
    activation_expected: str          # "RELU"/"SIGMOID"/"TANH"/"LINEAR"
    weights_expected: List[float]
    bias_expected: float
    predecessor_inputs: List[Tuple[int, int, float]]  # (nodeId, layerId, value) to inject
    expected_output: float
    case_label: str


def _predecessor_values_for(node_layers, outputs, layer_id: int, predecessor_layer_id: int, mask: int):
    vals = []
    for sender_id in range(16):
        if mask & (1 << sender_id):
            vals.append((sender_id, predecessor_layer_id, outputs[(predecessor_layer_id, sender_id)]))
    return vals


def build_round_trip_vectors(
    model: ModelSpec,
    node_layers: Dict[int, List[dict]],
    hardware_ids: List[int],
    input_cases: Optional[List[List[float]]] = None,
    nn_setup_dir: Optional[str] = None,
) -> List[DeviceTestVector]:
    """
    hardware_ids: one per PHYSICAL device, same order codegen.assign_hardware_ids()
    uses (ascending layer_id, then node_id).
    input_cases: list of full input-layer value vectors to test against. If
    None, defaults to every combination of 0.0/1.0 for networks with
    inputSize <= 4 (matches the AND-gate golden-fixture style of exhaustive
    truth-table coverage), else a single all-1.0 case.
    """
    generate_manifest, device_manifest, setup_messages, setup_tool, wire_format = _import_nn_setup(nn_setup_dir)

    network_json = model_to_network_json(model)
    # build_devices() assigns hardware_ids[i] verbatim into device["hardwareId"] with
    # no parsing -- plain ints round-trip fine through both build_devices() and the
    # JSON file device_manifest.load_manifest() reads below.
    raw_devices = generate_manifest.build_devices(network_json, list(hardware_ids))

    # There is no in-memory "parse a list of dicts" entry point on the real
    # device_manifest module -- only load_manifest(path), which is exactly what
    # setup_tool.py itself calls in production (fed by generate_manifest.py's own
    # devices.json output). Round-tripping through a real temp JSON file here, rather
    # than reimplementing load_manifest's validation/defaulting logic, keeps this
    # harness exercising the REAL code path end-to-end instead of a shortcut around it.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_manifest:
        json.dump(raw_devices, tmp_manifest)
        tmp_manifest_path = tmp_manifest.name
    try:
        parsed_devices = device_manifest.load_manifest(tmp_manifest_path)
    finally:
        os.remove(tmp_manifest_path)

    input_size = model.layers[0].size
    if input_cases is None:
        if input_size <= 4:
            input_cases = [
                [float((combo >> i) & 1) for i in range(input_size)]
                for combo in range(2 ** input_size)
            ]
        else:
            input_cases = [[1.0] * input_size]

    vectors: List[DeviceTestVector] = []
    for raw, parsed in zip(raw_devices, parsed_devices):
        addr = raw["address"]
        label = f"L{addr['layerId']}_N{addr['nodeId']}"

        for case_index, input_values in enumerate(input_cases):
            outputs = simulate_all(node_layers, input_values)
            pred_layer = raw["predecessorLayerId"]
            pred_vals = _predecessor_values_for(
                node_layers, outputs, addr["layerId"], pred_layer, raw["predecessorMask"]
            )
            expected_output = outputs[(addr["layerId"], addr["nodeId"])]

            packets = setup_tool.build_device_packets(parsed, seq_start=0)
            last_seq = packets[-1][1]
            start_seq = (last_seq + 1) & 0xFF
            start_bytes = wire_format.build_packet(setup_messages.START, start_seq, setup_messages.pack_start())
            packets_hex = [p[2].hex() for p in packets] + [start_bytes.hex()]

            vectors.append(
                DeviceTestVector(
                    label=label,
                    hardware_id=parsed["hardwareId"],
                    packets_hex=packets_hex,
                    node_id=addr["nodeId"],
                    layer_id=addr["layerId"],
                    predecessor_layer_id=pred_layer,
                    predecessor_mask=raw["predecessorMask"],
                    address_expected=(addr["nodeId"], addr["layerId"], addr["clusterId"], addr["reserved"]),
                    predecessor_mask_expected=raw["predecessorMask"],
                    preceding_siblings_mask=raw["precedingSiblingsMask"],
                    successor_layer_id_expected=raw["successorLayerId"],
                    transmit_slot_expected=raw["transmitSlot"],
                    activation_expected=raw["activationType"],
                    weights_expected=raw["weights"],
                    bias_expected=raw["bias"],
                    predecessor_inputs=[(nid, lid, v) for nid, lid, v in pred_vals],
                    expected_output=expected_output,
                    case_label=f"case{case_index}_{input_values}",
                )
            )
    return vectors


_ACTIVATION_ENUM_CPP = {
    "RELU": "NNActivationType::RELU",
    "SIGMOID": "NNActivationType::SIGMOID",
    "TANH": "NNActivationType::TANH",
    "LINEAR": "NNActivationType::LINEAR",
}


def _cpp_test_fn_name(vector: DeviceTestVector, index: int) -> str:
    safe_case = vector.case_label.replace(".", "_").replace(" ", "").replace(",", "_").replace("[", "").replace("]", "")
    return f"test_{vector.label}_{safe_case}_{index}"


def _cpp_string_array(hex_list: List[str]) -> str:
    return "{\n" + ",\n".join(f'        "{h}"' for h in hex_list) + "\n    }"


def _cpp_float_array(name: str, values: List[float]) -> str:
    if not values:
        return f"static const float {name}[1] = {{0.0f}}; // unused, weightCount=0"
    body = ", ".join(f"{v:.8f}f" for v in values)
    return f"static const float {name}[{len(values)}] = {{{body}}};"


def generate_roundtrip_cpp(vectors: List[DeviceTestVector]) -> str:
    """
    Emits a single self-contained g++ test file. One test function per
    (device, input-case) pair, each doing the full chain: feed the REAL
    wire-format setup packets into a REAL NNSetupAgent, assert the
    assembled NNNodeConfig matches what generate_manifest.py intended,
    then run a REAL NNNode/NNScheduler pass and assert the output matches
    the mapper's own simulate.simulate_all() prediction.
    """
    lines: List[str] = [
        "// Auto-generated by nnaware-mapper's core/roundtrip.py — do not hand-edit.",
        "// Desktop hardware round-trip check: real setup-protocol bytes -> real",
        "// NNSetupAgent -> real NNNode/NNScheduler inference, no physical device.",
        "#include <cassert>",
        "#include <cstdio>",
        "#include <cmath>",
        "#include <cstdint>",
        "#include <cstring>",
        "#include <string>",
        '#include "NNAddress.h"',
        '#include "NNPacket.h"',
        '#include "NNNode.h"',
        '#include "NNScheduler.h"',
        '#include "NNTransportLoopback.h"',
        '#include "NNSetupProtocol.h"',
        "",
        "constexpr float TOLERANCE = 0.0001f;",
        "",
        "static bool hexToPacket(const std::string& hex, NNPacket& out) {",
        "    uint8_t buf[128];",
        "    size_t n = hex.size() / 2;",
        "    if (n > sizeof(buf)) return false;",
        "    for (size_t i = 0; i < n; i++) {",
        "        buf[i] = static_cast<uint8_t>(std::stoul(hex.substr(i * 2, 2), nullptr, 16));",
        "    }",
        "    return deserializePacket(buf, static_cast<uint16_t>(n), out);",
        "}",
        "",
        "static NNPacket makeMarker(uint8_t nodeId, uint8_t layerId, float value) {",
        "    NNPacket pkt{};",
        "    pkt.header.sourceAddress = encodeAddress({nodeId, layerId, 0, 0});",
        "    pkt.header.type = NNPacketType::DATA;",
        "    pkt.header.payloadCount = 1;",
        "    pkt.payload[0] = value;",
        "    return pkt;",
        "}",
        "",
        "int totalPassed = 0;",
        "int totalTests = 0;",
        "",
    ]

    for i, v in enumerate(vectors):
        fn = _cpp_test_fn_name(v, i)
        lines.append(f"void {fn}() {{")
        lines.append(f'    printf("--- %s (device %s, %s) ---\\n", "{fn}", "{v.label}", "{v.case_label}");')
        lines.append("    totalTests++;")
        lines.append("")
        lines.append("    // 1. Feed the REAL wire-format setup packets into a REAL NNSetupAgent.")
        lines.append("    NNTransportLoopback transport;")
        lines.append("    NNVolatileConfigStore store;")
        lines.append(f"    NNSetupAgent agent(0x{v.hardware_id:016X}ULL, transport, store);")
        lines.append("    agent.begin();")
        lines.append("")
        lines.append(f"    const char* packetsHex[] = {_cpp_string_array(v.packets_hex)};")
        lines.append(f"    for (const char* hex : packetsHex) {{")
        lines.append("        NNPacket pkt;")
        lines.append('        bool ok = hexToPacket(hex, pkt);')
        lines.append('        assert(ok && "packet failed to deserialize -- checksum or length mismatch");')
        lines.append("        agent.onSetupPacket(pkt);")
        lines.append("    }")
        lines.append("")
        lines.append('    assert(agent.isRunning() && "device never reached RUNNING after all setup packets + START");')
        lines.append("")
        lines.append("    // 2. Assert the assembled config matches what generate_manifest.py intended.")
        lines.append("    const NNNodeConfig& cfg = agent.getNodeConfig();")
        lines.append(f"    assert(cfg.address.nodeId == {v.address_expected[0]});")
        lines.append(f"    assert(cfg.address.layerId == {v.address_expected[1]});")
        lines.append(f"    assert(cfg.address.clusterId == {v.address_expected[2]});")
        lines.append(f"    assert(cfg.address.reserved == {v.address_expected[3]});")
        lines.append(f"    assert(cfg.predecessorMask == {v.predecessor_mask_expected});")
        lines.append(f"    assert(cfg.predecessorLayerId == {v.predecessor_layer_id});")
        lines.append(f"    assert(cfg.successorLayerId == {v.successor_layer_id_expected});")
        lines.append(f"    assert(cfg.transmitSlot == {v.transmit_slot_expected});")
        lines.append(f"    assert(cfg.activationType == {_ACTIVATION_ENUM_CPP[v.activation_expected]});")
        lines.append(f"    assert(cfg.weightCount == {len(v.weights_expected)});")
        for wi, wv in enumerate(v.weights_expected):
            lines.append(f"    assert(std::fabs(cfg.weights[{wi}] - ({wv:.8f}f)) < TOLERANCE);")
        lines.append(f"    assert(std::fabs(cfg.bias - ({v.bias_expected:.8f}f)) < TOLERANCE);")
        lines.append('    printf("    config assembled by NNSetupAgent matches generate_manifest.py exactly\\n");')
        lines.append("")
        lines.append("    // 3. Run a REAL inference pass with a REAL NNNode/NNScheduler.")
        lines.append("    NNNode node(cfg);")
        lines.append(f"    NNWindowConfig windowCfg{{{v.layer_id}, {v.preceding_siblings_mask}}};")
        lines.append("    NNScheduler scheduler(node, windowCfg);")
        lines.append("")
        for nid, lid, val in v.predecessor_inputs:
            lines.append(f"    {{ NNPacket p = makeMarker({nid}, {lid}, {val:.8f}f); node.onPacketReceived(p); scheduler.onPacketObserved(p); }}")
        # Sibling markers so WAITING_FOR_TURN can clear for node_id > 0 in a layer.
        lines.append(f"    for (uint8_t sib = 0; sib < 16; sib++) {{")
        lines.append(f"        if ({v.preceding_siblings_mask} & (uint16_t(1) << sib)) {{")
        lines.append(f"            NNPacket p = makeMarker(sib, {v.layer_id}, 0.0f);")
        lines.append("            scheduler.onPacketObserved(p);")
        lines.append("        }")
        lines.append("    }")
        lines.append("")
        lines.append("    NNPacket outPkt;")
        lines.append("    bool gotOutput = false;")
        lines.append("    for (int iter = 0; iter < 20 && !gotOutput; iter++) {")
        lines.append("        scheduler.tick();")
        lines.append("        gotOutput = scheduler.hasOutputReady(outPkt);")
        lines.append("    }")
        lines.append('    assert(gotOutput && "scheduler never produced an output within 20 ticks");')
        lines.append(f"    assert(std::fabs(outPkt.payload[0] - ({v.expected_output:.8f}f)) < TOLERANCE);")
        lines.append(f'    printf("    PASSED: output = %.6f (expected {v.expected_output:.6f})\\n", outPkt.payload[0]);')
        lines.append("    totalPassed++;")
        lines.append("}")
        lines.append("")

    lines.append("int main() {")
    for i, v in enumerate(vectors):
        lines.append(f"    {_cpp_test_fn_name(v, i)}();")
    lines.append('    printf("\\n%d/%d round-trip test(s) passed.\\n", totalPassed, totalTests);')
    lines.append("    return (totalPassed == totalTests) ? 0 : 1;")
    lines.append("}")

    return "\n".join(lines)


def run_round_trip_check(
    model: ModelSpec,
    node_layers: Dict[int, List[dict]],
    hardware_ids: List[int],
    src_dir: str,
    nn_setup_dir: Optional[str] = None,
    input_cases: Optional[List[List[float]]] = None,
) -> Tuple[bool, str]:
    """
    Builds test vectors, generates the .cpp, compiles it against src_dir
    (the library's real headers) with g++, runs it, and returns
    (passed, combined_stdout_stderr). src_dir must contain NNNode.h,
    NNFailover.h, NNSetupProtocol.h, NNScheduler.h, NNTransportLoopback.h,
    NNTransport.h, NNAddress.h, NNPacket.h, NNBuffer.h, NNActivation.h.
    """
    vectors = build_round_trip_vectors(model, node_layers, hardware_ids, input_cases, nn_setup_dir)
    cpp_source = generate_roundtrip_cpp(vectors)

    with tempfile.TemporaryDirectory() as tmp:
        cpp_path = os.path.join(tmp, "test_roundtrip.cpp")
        bin_path = os.path.join(tmp, "test_roundtrip")
        with open(cpp_path, "w") as f:
            f.write(cpp_source)

        compile_proc = subprocess.run(
            ["g++", "-std=c++17", "-I", src_dir, cpp_path, "-o", bin_path],
            capture_output=True, text=True,
        )
        if compile_proc.returncode != 0:
            return False, f"COMPILE FAILED:\n{compile_proc.stdout}\n{compile_proc.stderr}"

        run_proc = subprocess.run([bin_path], capture_output=True, text=True)
        output = run_proc.stdout + run_proc.stderr
        return run_proc.returncode == 0, output