"""
conftest.py — makes receiver.py, transmitter.py and packet_builder.py
importable from every test module, regardless of which directory pytest is
invoked from.
"""
import sys
import pathlib
 
_HERE = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "tests"))            # receiver, transmitter, packet_builder
sys.path.insert(0, str(_HERE / "compatibility"))    # cpp_runner, packet_diff