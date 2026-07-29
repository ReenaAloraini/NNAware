"""
NNAware Mapper -- Streamlit UI

High-level tool: load a pretrained, fully-connected, inference-only network,
map it onto one physical NNAware device per neuron, scan for and select real
devices, provision them, and run real inference on them -- prediction and
execution time included. No file exports or outside scripts required for
the normal path; those are still available under "Advanced" for people who
want them.

v1 scope: fully connected, inference-only, one neuron per physical device,
identical IoT devices. See core/ module docstrings for the full rationale.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import sys
import tempfile
import time
import types

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from core.model_io import load_model
from core.topology import build_topology, device_count
from core.constraints import ConstraintError, validate_topology
from core.codegen import (
    NN_TERMINAL_LAYER_SENTINEL,
    backup_detection_caveats,
    eligible_backup_targets,
    physical_nodes,
    model_to_network_json,
    to_cpp_header,
    node_label,
)
from core import simulate as simulate_mod
from core import runtime as runtime_mod
from core.roundtrip import run_round_trip_check, _import_nn_setup

EXAMPLE_PATH = os.path.join(HERE, "examples", "and_gate.json")

# Real repo layout is tools/nn_mapper (this app) + tools/nn_setup + src, all
# siblings under the repo root -- matches roundtrip.py's own default guess.
# A normal user never needs to know these paths exist; they're only surfaced
# in the UI as a fallback if auto-detection fails.
DEFAULT_NN_SETUP_DIR = os.path.abspath(os.path.join(HERE, "..", "nn_setup"))
DEFAULT_SRC_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
NN_SETUP_DIR_EXISTS = os.path.isdir(DEFAULT_NN_SETUP_DIR)
SRC_DIR_EXISTS = os.path.isdir(DEFAULT_SRC_DIR)
PLACEHOLDER_HW_ID_BASE = 0xAABBCCDD00000000  # only used for the desktop verification step, never shown


def _scan_for_devices(nn_setup_dir: str, port: int, hello_window: float):
    """Listens for real, unconfigured devices announcing themselves (HELLO), using the
    REAL setup_tool.discover() -- an empty manifest set just means every device found is
    returned, none are filtered out. Returns (sorted hardware_id list, captured log text)."""
    _generate_manifest, _device_manifest, _setup_messages, setup_tool, _wire_format = _import_nn_setup(nn_setup_dir)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", port))
    log_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(log_buffer):
            seen = setup_tool.discover(sock, set(), hello_window)
    finally:
        sock.close()
    return sorted(seen), log_buffer.getvalue()


def _deploy_to_devices(devices, broadcast_addr, port, retries, timeout, hello_window, setup_tool, setup_messages, wire_format):
    """
    Adapted from setup_tool.py's own main() -- same discover -> provision -> START
    sequence, using the REAL setup_tool.discover()/provision_device() functions (not a
    reimplementation), just returning a bool instead of an exit code so the caller can
    show PASS/FAIL in the UI. Every print() call inside discover()/provision_device()
    is real setup_tool.py output; the caller wraps this in redirect_stdout to capture it.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", port))
    target_addr = (broadcast_addr, port)
    try:
        manifest_hw_ids = {d["hardwareId"] for d in devices}
        setup_tool.discover(sock, manifest_hw_ids, hello_window)

        args_ns = types.SimpleNamespace(retries=retries, timeout=timeout)
        results = {device["hardwareId"]: setup_tool.provision_device(sock, target_addr, device, args_ns)
                   for device in devices}

        failed = [hw_id for hw_id, ok in results.items() if not ok]
        if failed:
            print(f"\n{len(failed)}/{len(devices)} device(s) FAILED -- not sending START:")
            for hw_id in failed:
                print(f"  0x{hw_id:016X}")
            return False

        print(f"\nAll {len(devices)} device(s) CONFIGURED -- broadcasting START x3 ...")
        start_packet = wire_format.build_packet(setup_messages.START, 0, setup_messages.pack_start())
        for _ in range(3):
            sock.sendto(start_packet, target_addr)
            time.sleep(0.2)
        print("Done.")
        return True
    finally:
        sock.close()


st.set_page_config(page_title="NNAware Mapper", layout="wide")
st.title("NNAware Mapper")
st.caption("Map a pretrained, fully-connected network onto NNAware IoT devices, then run it for real.")


# ---------------------------------------------------------------------------
# 1. Your network
# ---------------------------------------------------------------------------
st.header("1. Your network")

with open(EXAMPLE_PATH) as f:
    example_text = f.read()

col_upload, col_edit = st.columns([1, 2])
with col_upload:
    uploaded = st.file_uploader("Upload a model JSON", type=["json"])
    st.caption("Or edit the JSON directly →")
    use_example = st.button("Reset to AND-gate example", width="stretch")

with col_edit:
    if "model_json_text" not in st.session_state or use_example:
        st.session_state["model_json_text"] = example_text
    if uploaded is not None:
        st.session_state["model_json_text"] = uploaded.read().decode("utf-8")

    model_json_text = st.text_area(
        "Model JSON (layer 0 = input, no weights/bias; later layers fully connected)",
        value=st.session_state["model_json_text"],
        height=220,
        key="model_json_editor",
    )
    st.session_state["model_json_text"] = model_json_text

model = None
try:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp.write(model_json_text)
        tmp_path = tmp.name
    model = load_model(tmp_path)
    os.remove(tmp_path)
except (ValueError, json.JSONDecodeError, OSError) as e:
    st.error(f"Couldn't load this model: {e}")

if model is None:
    st.stop()

node_layers = build_topology(model)
n_devices = device_count(node_layers)

try:
    validate_topology(node_layers)
except ConstraintError as e:
    st.error(f"This network can't be mapped onto NNAware devices:\n\n{e}")
    st.stop()

st.success(f"This network needs **{n_devices} physical device(s)** to run.")

phys = physical_nodes(node_layers)
simple_rows = [
    {"device": node_label(n), "activation": n["activation"], "inputs": n["weight_count"], "bias": n["bias"]}
    for n in phys
]
st.dataframe(simple_rows, width="stretch", hide_index=True)

with st.expander("Show technical details (addresses, predecessor masks, wire-protocol fields)"):
    table_rows = [
        {
            "label": node_label(n),
            "layer_id": n["layer_id"],
            "node_id": n["node_id"],
            "activation": n["activation"],
            "predecessor_mask": bin(n["predecessor_mask"]),
            "predecessor_layer_id": n["predecessor_layer_id"],
            "successor_layer_id": n["successor_layer_id"] if n["successor_layer_id"] is not None else NN_TERMINAL_LAYER_SENTINEL,
            "weight_count": n["weight_count"],
            "bias": n["bias"],
        }
        for n in phys
    ]
    st.dataframe(table_rows, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Backup roles (optional fault tolerance)
# ---------------------------------------------------------------------------
# Only the PAIRS are chosen here; generate_manifest.py derives backupWeights,
# backupTargetBias, layerRosterMask and the rest from the target's own entry.
# Selections live in st.session_state (Streamlit persists them by widget key),
# because network_json below is rebuilt from scratch on every rerun. The key
# carries a topology signature so editing a weight keeps your selections but
# resizing a layer discards ones that no longer make sense.
topology_sig = "x".join(str(l.size) for l in model.layers)

backup_selection: dict = {}
grace_selection: dict = {}

with st.expander("Backup roles (optional -- fault tolerance)"):
    st.caption(
        "A device can also stand in for exactly ONE SIBLING in the same layer. If that "
        "sibling goes silent, this device first asks it to retransmit; if it still doesn't "
        "answer, this device computes the missing output itself from a copy of the sibling's "
        "weights and broadcasts it under the sibling's own identity, so the next layer never "
        "notices. Costs no extra hardware -- backup duty rides on a device that already has "
        "a job."
    )

    for layer_index, layer in enumerate(model.layers[1:]):
        layer_id = layer_index + 1
        size = layer.size
        label = lambda nid: node_label(node_layers[layer_id][nid])

        if not eligible_backup_targets(size, 0):
            st.caption(f"**Layer {layer_id}** — {size} node. A backup has to be a sibling in "
                       f"the same layer, so this layer can't have one.")
            continue

        st.markdown(f"**Layer {layer_id}** — {size} nodes")
        layer_backups: dict = {}
        cols = st.columns(min(size, 4))
        for nid in range(size):
            # None means "no backup duty". Self is never offered, so
            # generate_manifest.py's self-backup rejection is unreachable here.
            with cols[nid % len(cols)]:
                choice = st.selectbox(
                    f"{label(nid)} backs up",
                    options=[None] + eligible_backup_targets(size, nid),
                    format_func=lambda j: "— none —" if j is None else label(j),
                    key=f"nnaware_backup_L{layer_index}_N{nid}_{topology_sig}",
                )
            if choice is not None:
                layer_backups[nid] = choice

        if layer_backups:
            backup_selection[layer_index] = layer_backups
            grace_selection[layer_index] = int(
                st.number_input(
                    f"Layer {layer_id}: resend grace (ms) — how long to wait for a "
                    f"retransmission before substituting",
                    min_value=50, max_value=2000, value=300, step=50,
                    key=f"nnaware_grace_L{layer_index}_{topology_sig}",
                )
            )
            st.dataframe(
                [{"backup device": label(b), "stands in for": label(t)}
                 for b, t in sorted(layer_backups.items())],
                width="stretch", hide_index=True,
            )
            # codegen returns tagged facts; the wording (including anything
            # firmware-specific, like how long FailoverNode waits) belongs here.
            for c in backup_detection_caveats(size, layer_backups):
                if c["kind"] == "no_round_end_signal":
                    st.caption(
                        "ℹ️ This layer has 2 nodes, so once you exclude the backup and its "
                        "target there's no third sibling left to signal that a round finished. "
                        "FailoverNode starts detection once this node's own inputs arrive "
                        "instead. Expect a resend request on most passes -- harmless, the "
                        "healthy sibling just answers it."
                    )
                elif c["kind"] == "target_not_last":
                    st.caption(
                        f"ℹ️ {label(c['backer'])} backs up {label(c['target'])}, which isn't the "
                        f"last node in this layer. If it dies, the nodes after it stall waiting "
                        f"for their turn until FailoverNode's turn-stall timeout fires, so "
                        f"recovery takes about a second longer. Targeting "
                        f"{label(c['suggested_target'])} avoids that."
                    )

    if backup_selection:
        st.info(
            "Flash **examples/FailoverNode** to every board. examples/SetupAndRun and "
            "examples/RunningNode don't include NNFailover.h -- they accept the backup "
            "config, report it as configured, and then silently never act on it."
        )
    else:
        st.caption("No backup roles selected -- this deploys exactly as before.")

try:
    network_json = model_to_network_json(
        model, backups=backup_selection, resend_grace_ms=grace_selection,
    )
except ValueError as e:
    st.error(f"Backup roles aren't valid: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Shared network settings (used by scanning, setup, and running)
# ---------------------------------------------------------------------------
with st.expander("Network settings (defaults work for most local networks)"):
    st.caption("Setup uses UDP broadcast; running uses UDP multicast -- separate ports in case your firmware uses different ones.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Setup (provisioning)**")
        broadcast_addr = st.text_input("Broadcast address", value="255.255.255.255")
        setup_port = st.number_input("Setup port", value=4210, step=1)
        hello_window = st.number_input("Seconds to listen for devices", value=10.0, step=1.0)
        setup_retries = st.number_input("Retries per message", value=5, step=1)
        setup_timeout = st.number_input("Seconds to wait per attempt", value=0.5, step=0.1)
    with col2:
        st.markdown("**Running (inference)**")
        runtime_port = st.number_input("Runtime port", value=4211, step=1)
        run_timeout = st.number_input("Seconds to wait for a result", value=10.0, step=1.0)

nn_setup_dir = DEFAULT_NN_SETUP_DIR
if not NN_SETUP_DIR_EXISTS:
    st.warning("Couldn't auto-locate tools/nn_setup -- set it below.")
    nn_setup_dir = st.text_input("tools/nn_setup directory", value=DEFAULT_NN_SETUP_DIR, key="nn_setup_dir_main")


# ---------------------------------------------------------------------------
# 2. Devices
# ---------------------------------------------------------------------------
st.header("2. Devices")
st.caption("Scan for your physical devices, then pick which ones to use for this network.")

if st.button("Scan for devices", type="primary"):
    if not os.path.isdir(nn_setup_dir):
        st.error(f"nn_setup directory not found: {nn_setup_dir}")
    else:
        with st.spinner(f"Listening for devices ({hello_window:.0f}s)..."):
            try:
                found, scan_log = _scan_for_devices(nn_setup_dir, int(setup_port), hello_window)
                st.session_state["scanned_devices"] = found
            except Exception as e:
                st.session_state["scanned_devices"] = []
                st.error(f"{type(e).__name__}: {e}")

scanned = st.session_state.get("scanned_devices", [])
selected_ids = []
if not scanned:
    st.info("No devices scanned yet. Power on your unconfigured devices, then click Scan.")
else:
    st.write(f"Found {len(scanned)} device(s):")
    options = [f"0x{hw:016X}" for hw in scanned]
    default_selection = options[:n_devices] if len(options) >= n_devices else options
    chosen = st.multiselect(f"Select exactly {n_devices} device(s) to use", options=options, default=default_selection)
    selected_ids = sorted(int(c, 16) for c in chosen)

    if len(selected_ids) == n_devices and n_devices > 0:
        assign_preview = [
            {"device": f"0x{hw:016X}", "assigned_to": node_label(n)}
            for hw, n in zip(selected_ids, phys)
        ]
        st.dataframe(assign_preview, width="stretch", hide_index=True)
    elif n_devices > 0:
        st.warning(f"Select exactly {n_devices} device(s) (currently {len(selected_ids)}).")


# ---------------------------------------------------------------------------
# 3. Setup
# ---------------------------------------------------------------------------
st.header("3. Setup")
st.caption("Sends each selected device its address, topology, and weights, then starts them running.")

if n_devices == 0:
    st.info("No physical devices in this topology yet.")
elif len(selected_ids) != n_devices:
    st.warning("Select your devices in section 2 first.")
elif st.button("Start setup", type="primary"):
    if not os.path.isdir(nn_setup_dir):
        st.error(f"nn_setup directory not found: {nn_setup_dir}")
    else:
        log_buffer = io.StringIO()
        success = False
        with st.spinner("Provisioning devices..."):
            try:
                generate_manifest, device_manifest, setup_messages, setup_tool, wire_format = \
                    _import_nn_setup(nn_setup_dir)
                raw_devices = generate_manifest.build_devices(
                    network_json, [f"0x{h:016X}" for h in selected_ids],
                )
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                    json.dump(raw_devices, tmp)
                    tmp_manifest_path = tmp.name
                try:
                    devices = device_manifest.load_manifest(tmp_manifest_path)
                finally:
                    os.remove(tmp_manifest_path)

                with contextlib.redirect_stdout(log_buffer):
                    success = _deploy_to_devices(
                        devices, broadcast_addr, int(setup_port), int(setup_retries),
                        setup_timeout, hello_window, setup_tool, setup_messages, wire_format,
                    )
            except Exception as e:
                print(f"{type(e).__name__}: {e}", file=log_buffer)

        st.session_state["setup_done"] = success
        if success:
            st.success("All device(s) configured and started.")
        else:
            st.error("Setup failed -- see the log below.")
        st.code(log_buffer.getvalue() or "(no output)", language="text")


# ---------------------------------------------------------------------------
# 4. Run
# ---------------------------------------------------------------------------
st.header("4. Run")
st.caption("Sends your input to the real devices and waits for the real prediction, over UDP multicast.")

input_size = model.layers[0].size


def _parse_test_input(text: str, size: int) -> list:
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    if len(parts) != size:
        raise ValueError(f"expected {size} value(s) (this network's input size), got {len(parts)}")
    try:
        return [float(p) for p in parts]
    except ValueError:
        raise ValueError("every value must be a number")


run_input_text = st.text_input(
    f"Input values, comma-separated ({input_size} needed)",
    value=", ".join("0" for _ in range(input_size)),
    key=f"run_input_text_{input_size}",
)

if not st.session_state.get("setup_done"):
    st.info("Run setup (section 3) first, or try anyway if your devices are already configured.")

if st.button("Run", type="primary"):
    try:
        run_input = _parse_test_input(run_input_text, input_size)
    except ValueError as e:
        st.error(f"Couldn't parse input: {e}")
        run_input = None

    if run_input is not None:
        with st.spinner(f"Sending input and waiting up to {run_timeout:.0f}s for a result..."):
            try:
                output, elapsed = runtime_mod.run_inference_on_devices(
                    node_layers, run_input, port=int(runtime_port), timeout_s=run_timeout,
                )
                st.session_state["run_result"] = (output, elapsed, None)
            except (ValueError, TimeoutError, OSError) as e:
                st.session_state["run_result"] = (None, None, f"{type(e).__name__}: {e}")

if "run_result" in st.session_state:
    output, elapsed, error = st.session_state["run_result"]
    if error:
        st.error(error)
    else:
        col_pred, col_time = st.columns(2)
        with col_pred:
            if len(output) == 1:
                st.metric("Prediction", f"{output[0]:.4f}")
            else:
                st.write("Prediction:", output)
        with col_time:
            st.metric("Execution time", f"{elapsed * 1000:.1f} ms")

with st.expander("Quick desktop check (no hardware -- pure Python prediction for comparison)"):
    if st.button("Simulate this input"):
        try:
            sim_input = _parse_test_input(run_input_text, input_size)
            sim_output = simulate_mod.simulate(node_layers, sim_input)
            st.write("Simulated prediction:", sim_output)
        except ValueError as e:
            st.error(f"Couldn't parse input: {e}")
    if input_size <= 4:
        input_cases = [[float((combo >> i) & 1) for i in range(input_size)] for combo in range(2 ** input_size)]
        sim_rows = [{"input": case, "output": simulate_mod.simulate(node_layers, case)} for case in input_cases]
        st.dataframe(sim_rows, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Advanced: manual export (optional, for people who don't want to deploy from here)
# ---------------------------------------------------------------------------
with st.expander("Advanced: export files manually"):
    st.caption("Not needed for the flow above -- only useful if you want the raw files for your own records or scripts.")

    network_json_text = json.dumps(network_json, indent=2)
    st.download_button(
        "Download network.json", data=network_json_text,
        file_name="network.json", mime="application/json",
    )
    st.code(network_json_text, language="json")

    st.markdown("**Real device manifest (devices.json), built with generate_manifest.py:**")
    if len(selected_ids) == n_devices and n_devices > 0:
        try:
            generate_manifest, *_rest = _import_nn_setup(nn_setup_dir)
            raw_devices = generate_manifest.build_devices(
                network_json, [f"0x{h:016X}" for h in selected_ids],
            )
            devices_json_text = json.dumps(raw_devices, indent=2)
            st.download_button(
                "Download devices.json", data=devices_json_text,
                file_name="devices.json", mime="application/json",
            )
            st.code(devices_json_text, language="json")
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
    else:
        st.caption("Select your devices in section 2 to generate this.")

    st.markdown("**Per-node C++ header preview (debug only, from the mapper's own topology):**")
    for n in phys:
        st.code(to_cpp_header(n), language="cpp")


# ---------------------------------------------------------------------------
# Advanced: verify before deploying (optional, desktop-only)
# ---------------------------------------------------------------------------
with st.expander("Advanced: verify before deploying (optional, desktop-only)"):
    st.caption(
        "A desktop-only pre-flight check, run instead of deploying -- not a re-run of "
        "anything on real hardware. It feeds your mapped network through the actual "
        "provisioning code and the actual device-side C++ code (using a loopback "
        "stand-in for the wireless link), and confirms the result matches this "
        "network's own prediction. No physical device is touched or required."
    )
    st.caption(
        "Note: this check rebuilds the network description on its own, so it verifies the "
        "network WITHOUT any backup roles you selected above. It still proves the weights, "
        "topology and predictions are right -- it just doesn't cover backup provisioning."
    )

    if SRC_DIR_EXISTS:
        src_dir = DEFAULT_SRC_DIR
    else:
        st.warning("Couldn't auto-locate the library's src/ folder -- set it manually below.")
        src_dir = st.text_input("library src/ directory (patched headers)", value=DEFAULT_SRC_DIR)

    if n_devices == 0:
        st.info("No physical devices in this topology yet.")
    elif st.button("Run verification"):
        if not os.path.isdir(nn_setup_dir):
            st.error(f"nn_setup directory not found: {nn_setup_dir}")
        elif not os.path.isdir(src_dir):
            st.error(f"src directory not found: {src_dir}")
        else:
            placeholder_ids = [PLACEHOLDER_HW_ID_BASE + i for i in range(n_devices)]
            with st.spinner("Running the desktop pre-flight check..."):
                try:
                    passed, output = run_round_trip_check(
                        model, node_layers, placeholder_ids, src_dir, nn_setup_dir=nn_setup_dir,
                    )
                except Exception as e:
                    passed, output = False, f"{type(e).__name__}: {e}"

            if passed:
                st.success("Verification PASSED -- this mapping is safe to deploy.")
            else:
                st.error("Verification FAILED -- do not deploy this mapping yet.")
            with st.expander("Details"):
                st.code(output, language="text")