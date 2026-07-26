from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write NumPy arrays as a reproducible NPZ with fixed ZIP metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            array = np.asanyarray(arrays[name])
            if array.dtype.hasobject:
                raise ValueError(f"object arrays are not allowed in deterministic NPZ: {name}")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)

