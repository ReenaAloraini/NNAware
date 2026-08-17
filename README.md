# NNAware 

A neural-network-aware communication framework for IoT devices.

NNAware runs a trained, fully-connected neural network across a cluster of IoT devices, with 
**one physical device per neuron**. No single device holds the whole model.

The host (the device running this framework) discovers the devices over the network and provisions each one with its address,
topology and weights. The devices then collectively act as a network and perform inference on their own: each neuron computes
its activation and broadcasts it to the next layer, until the output layer reports the prediction back.

The framework is deliberately transport-independent: the protocol and node model are
separate from the transport technology used to carry them. This version of the implementation uses WiFi
(UDP broadcast for setup, UDP multicast for inference) on Seeed Wio Terminal boards. The framework also includes implementations of
BLE, MQTT and loopback transports, however, they were not fully tested yet. 

The optional fault-tolerance mechanism lets a device act as a backup for a failed sibling node in its own
layer, recomputing the missing output from a copy of that sibling's weights so the rest of
the network never notices.

## Requirements

- Python 3.9+
- Arduino IDE with Wio Terminal / Seeeduino board support
- One Wio Terminal per neuron in the pre-trained network 

## How to run

### 1. Flash the devices

Flash every Wio Terminal with `examples/Device_Setup/Device_Setup.ino` so the framework can
recognize them.

Before flashing, edit the settings in sketch:

- `WIFI_SSID` and `WIFI_PASSWORD`:  network credentials
- `BROADCAST_ADDR`: subnet's broadcast address 
- `NN_HARDWARE_ID`: **must be unique for each device**


### 2. Run the application

The framework's UI is found in `tools/nn_mapper`:

```bash
cd tools/nn_mapper
streamlit run app.py
```

It is a localy hosted web application with four sections:

1. **Network**: upload or edit your model, and see how many devices it needs
2. **Devices**: scan for your powered-on boards and pick which ones to use
3. **Setup**: provision the selected devices and verify them
4. **Run**: send an input and get the prediction back from the real hardware 

## Project layout

| Path | Description |
|---|---|
| `src/` | The framework core modules (node, scheduler, packet format, transports, failover) |
| `examples/Device_Setup/` | Device firmware  |
| `experiments_models/` | Pre-trained models used to test the framework on real hardware  |
| `tools/nn_mapper/` | Streamlit app: map a model onto devices, provision, run |
| `tools/nn_setup/` | Command-line provisioning toolchain |
| `desktop_tests/`, `python_tests/` | C++ and Python test suites |


