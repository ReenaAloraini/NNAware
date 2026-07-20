"""
conftest.py — makes receiver.py, transmitter.py (in python_tests/) and
packet_builder.py (in python_tests/tests/) importable from every test
module, regardless of which directory pytest is invoked from.
"""
import sys
import pathlib
 
_HERE = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))                    # python_tests/          -> receiver, transmitter
sys.path.insert(0, str(_HERE / "tests"))            # python_tests/tests/    -> packet_builder
sys.path.insert(0, str(_HERE / "compatibility"))    # python_tests/compatibility/ -> cpp_runner, packet_diff