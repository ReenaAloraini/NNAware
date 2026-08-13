# NNAware

Run a trained neural network across a cluster of IoT devices — **one physical device per neuron**.
A laptop provisions the devices over UDP broadcast, then they perform inference by broadcasting their
activations layer by layer over UDP multicast. No device holds the whole model.

> Research library, v0.1.0. Validated on real hardware, with desktop test suites covering the
> protocol and failover logic.

## Requirements

- Python 3.9+ (for the tooling)
- `g++` and CMake (optional — only for the desktop tests and the no-hardware check)
- Arduino IDE + Wio Terminal / Seeeduino boards (for real deployment)

---

## 1. Try it without hardware

Compiles the real device C++ code and drives it with the real provisioning bytes, then checks the
result against an independent simulation:

```bash
cd tools/nn_mapper
python -c "
import sys, os; sys.path.insert(0,'.')
from core.model_io import load_model
from core.topology import build_topology, device_count
from core.roundtrip import run_round_trip_check
m  = load_model('examples/and_gate.json')
nl = build_topology(m)
ids = [0xAABBCCDD00000000 + i for i in range(device_count(nl))]
ok, out = run_round_trip_check(m, nl, ids, os.path.abspath('../../src'),
                               nn_setup_dir=os.path.abspath('../nn_setup'))
print(out); print('ROUND-TRIP:', 'PASSED' if ok else 'FAILED')
"
```

Expect `4/4 round-trip test(s) passed.` / `ROUND-TRIP: PASSED`.

## 2. Deploy to real devices 

**Flash the boards first.** Open `examples/FailoverNode`, set `WIFI_SSID`, `WIFI_PASSWORD`,
`BROADCAST_ADDR` (your subnet's broadcast address), and give each board a **unique**
`NN_HARDWARE_ID`. Flash the *same* sketch to every board — roles come from the laptop, not the
firmware.

`FailoverNode` is the complete runtime: setup phase, inference, packet retransmission, input-node
seeding, and the fault-tolerance layer. It is the sketch to deploy.

The others are narrower diagnostics, useful when something isn't working:

| Sketch | Use for |
|---|---|
| `AddressTest` | Verifying address encode/decode on-device |
| `PacketTest` | Verifying serialize/deserialize on-device |
| `SetupTest` | Testing provisioning alone — stops at `CONFIGURED`, never runs inference |

**Then run the app:**

```bash
pip install -r tools/nn_mapper/requirements.txt
streamlit run tools/nn_mapper/app.py
```

Work through its four sections: **1. Your network** (upload/edit a model, see how many devices you
need) → **2. Devices** (scan for boards, pick which to use) → **3. Setup** (provision and verify) →
**4. Run** (send an input, get the prediction). An optional panel assigns backup roles for fault
tolerance.

## 3. Or use the command line

```bash
cd tools/nn_setup
python generate_manifest.py                                      # network.json -> devices.json
python setup_tool.py --config devices.json --dry-run --dump-hex  # inspect, send nothing
python setup_tool.py --config devices.json --broadcast-addr 192.168.1.255
```

Use `--config`, **not** `--network`/`--hardware-ids`: the dynamic pool mode under-counts input nodes
when `network.json` uses `inputValues`, and fails even with enough devices in the pool.

Edit `network.json` to describe your network (one entry per compute layer; `layerId 0` is reserved
for the input feed):

```jsonc
{
  "inputSize": 1,
  "inputValues": [1.0],        // OPTIONAL: makes each input a real device. Omit to keep it virtual.
  "layers": [
    {
      "nodes": 2,
      "activationType": "RELU",          // RELU | SIGMOID | TANH | LINEAR
      "weights": [[0.5], [-1.0]],        // one row per node, each row = previous layer's width
      "bias": [0.2, 0.0],                // OPTIONAL, defaults to 0.0
      "backups": {"0": 1},               // OPTIONAL: node 0 backs up node 1 (same layer only)
      "resendGraceMs": 300               // OPTIONAL, defaults to 50
    },
    { "nodes": 1, "activationType": "SIGMOID", "weights": [[1.0, 1.0]], "bias": [-0.3] }
  ]
}
```

`hardware_ids.json` is a flat list of the `NN_HARDWARE_ID` values you flashed. Everything else — each
device's address, masks, transmit slot and backup weights — is derived automatically.

---

## Tests

```bash
cd desktop_tests && cmake -S . -B build && cmake --build build && ctest --test-dir build   # 10/10
cd python_tests  && python -m pytest -q                          # 213 passed, 1 skipped, 1 xfailed
cd tools/nn_mapper && python -m pytest tests/test_and_gate.py -q  # 17 passed
```

95 of the Python tests shell out to the C++ `protocol_compatibility` executable and skip
automatically if you haven't run the CMake build — leaving 118 that need no C++ toolchain.

## Where things live

| Path | What |
|---|---|
| `src/NNNode.h` | `NNNodeConfig` + one neuron's compute |
| `src/NNScheduler.h` | Per-node state machine and sibling turn-taking |
| `src/NNPacket.h` | Wire format — 8-byte header + big-endian floats |
| `src/NNSetupProtocol.h` | Over-the-air provisioning protocol and opcodes |
| `src/NNFailover.h` | Backup standby, resend, duplicate suppression |
| `src/NNTransport*.h` | Loopback, UDP, UDP multicast, MQTT, BLE |
| `tools/nn_mapper/` | Streamlit app: map a model onto devices, deploy, run |
| `tools/nn_setup/` | CLI provisioning toolchain |
| `desktop_tests/`, `python_tests/` | C++ and Python test suites |

Each header carries a detailed comment block explaining its design — start there for the internals.
