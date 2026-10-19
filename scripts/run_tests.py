"""Test suite runner for TorchGeo-TS-Bench."""

import os
import sys
import pytest

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, "src")
tests_dir = os.path.join(project_root, "tests")

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run():
    print("=" * 70)
    print("[*] EXECUTING TORCHGEO-TS-BENCH AUTOMATED TEST SUITE")
    print(f"    Test Directory: {tests_dir}")
    print("=" * 70)
    ret_code = pytest.main([
        "-v",
        "-s",
        tests_dir,
    ])
    if ret_code == 0:
        print("\n[+] ALL TESTS PASSED SUCCESSFULLY!")
    else:
        print(f"\n[-] TESTS FAILED WITH EXIT CODE: {ret_code}")
    sys.exit(ret_code)

if __name__ == "__main__":
    run()
