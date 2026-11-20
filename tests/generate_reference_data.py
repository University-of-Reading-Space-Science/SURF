"""Regenerate the reference data used by the regression tests in test_surf.py.

Usage, from the repository root::

    python tests/generate_reference_data.py

WARNING: this overwrites tests/reference_data/surf_regression.npz. Only run it
when a change to SURF's numerical output is intended, after checking that the
change is correct, and record the reason in the commit message.
"""
import time

import numpy as np

from test_surf import CASES, REFERENCE_FILE, summarise


def main():
    data = {}
    for name, build in CASES.items():
        start = time.perf_counter()
        for key, value in summarise(build()).items():
            data[f"{name}/{key}"] = value
        print(f"{name}: {time.perf_counter() - start:.1f} s")

    REFERENCE_FILE.parent.mkdir(exist_ok=True)
    np.savez_compressed(REFERENCE_FILE, **data)
    print(f"Wrote {len(data)} arrays to {REFERENCE_FILE}")


if __name__ == "__main__":
    main()
