"""
cpp_runner.py — Python wrapper around the real C++ protocol_compatibility
executable, via subprocess.run(). This is the ONLY place in the Python test
suite that knows how to locate and invoke the C++ executable; every
compatibility test should go through the functions here rather
than calling subprocess directly, so there is exactly one place to fix if
the executable's location or invocation convention ever changes.
 
This module contains no protocol logic of its own — it only locates the
executable, pipes JSON to its stdin, and parses JSON from its stdout.
"""
import json
import platform
import subprocess
from pathlib import Path
from typing import Optional, Union
 
 
class CppExecutableError(RuntimeError):
    """
    Raised when the C++ executable can't be found, times out, or produces
    output that isn't valid JSON at all — distinct from a normal protocol-
    level rejection, which comes back as a well-formed JSON object with
    "ok": false (e.g. a deliberately corrupted packet). Compatibility tests
    should catch protocol-level "ok": false in the returned dict, not this
    exception — this exception means something is wrong with the harness
    itself, not the packet under test.
    """
    pass
 
 
def _default_executable_path() -> Path:
    """
    Locates the compiled executable relative to this file, handling the
    Windows .exe extension automatically. Assumes the standard project layout
    """
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent  # compatibility/ -> python_tests/ -> repo root
    build_dir = repo_root / "desktop_tests" / "build"
 
    exe_name = "protocol_compatibility.exe" if platform.system() == "Windows" else "protocol_compatibility"
    candidate = build_dir / exe_name
    if candidate.exists():
        return candidate
 
    # Some CMake/IDE setups nest a Debug/Release subfolder (common with multi-config generators)
    for subdir in ("Debug", "Release", "RelWithDebInfo"):
        nested = build_dir / subdir / exe_name
        if nested.exists():
            return nested
 
    raise CppExecutableError(
        f"Could not find the compiled C++ executable at {candidate} "
        f"(or common Debug/Release subfolders).\n"
        f"Build it first:\n"
        f"  cd desktop_tests && mkdir build && cd build && cmake -G \"Ninja\" .. && cmake --build .\n"
        f"(omit -G \"Ninja\" on platforms where the default generator already works)\n"
        f"If your build directory is somewhere else, pass its path explicitly:\n"
        f"  CppRunner(executable_path=\"/path/to/protocol_compatibility\")"
    )
 
 
class CppRunner:
    """
    Thin wrapper for invoking protocol_compatibility. One instance can be
    reused across many calls — each call is still a fresh subprocess (the
    executable is a one-shot CLI, not a long-running server), but reusing
    one CppRunner instance avoids re-locating the executable on every call.
    """
 
    def __init__(self, executable_path: Optional[Union[Path, str]] = None, timeout_seconds: float = 5.0):
        self.executable_path = Path(executable_path) if executable_path else _default_executable_path()
        if not self.executable_path.exists():
            raise CppExecutableError(f"Specified executable path does not exist: {self.executable_path}")
        self.timeout_seconds = timeout_seconds
 
    def _run(self, command: str, payload: dict) -> dict:
        try:
            result = subprocess.run(
                [str(self.executable_path), command],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            raise CppExecutableError(
                f"C++ executable timed out after {self.timeout_seconds}s running '{command}'"
            ) from e
        except OSError as e:
            raise CppExecutableError(
                f"Failed to launch C++ executable at {self.executable_path}: {e}"
            ) from e
 
        stdout = result.stdout.strip()
        if not stdout:
            raise CppExecutableError(
                f"C++ executable produced no stdout for '{command}' "
                f"(exit code {result.returncode}, stderr: {result.stderr!r})"
            )
 
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as e:
            raise CppExecutableError(
                f"C++ executable's stdout was not valid JSON for '{command}': {stdout!r}"
            ) from e
 
    # --- one method per command, matching protocol_compatibility.cpp exactly ---
 
    def encode_address(self, node_id: int, layer_id: int, cluster_id: int = 0, reserved: int = 0) -> dict:
        return self._run("encode-address", {
            "node_id": node_id, "layer_id": layer_id,
            "cluster_id": cluster_id, "reserved": reserved,
        })
 
    def decode_address(self, encoded: int) -> dict:
        return self._run("decode-address", {"encoded": encoded})
 
    def checksum(self, bytes_hex: str) -> dict:
        return self._run("checksum", {"bytes_hex": bytes_hex})
 
    def serialize(self, source_address: int, target_layer_id: int, packet_type: int,
                  sequence: int, values: list, flags: int = 0) -> dict:
        return self._run("serialize", {
            "source_address": source_address, "target_layer_id": target_layer_id,
            "type": packet_type, "sequence": sequence, "flags": flags, "values": values,
        })
 
    def deserialize(self, bytes_hex: str) -> dict:
        return self._run("deserialize", {"bytes_hex": bytes_hex})
 