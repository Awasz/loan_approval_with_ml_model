"""Dataset version identifiers recorded with every training run."""
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def file_fingerprint(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> dict[str, str]:
    """SHA-256, size and modification time of a source file (changes whenever the file changes)."""
    path = Path(path)
    sha = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    stat = path.stat()
    return {
        "file": path.name,
        "sha256": sha.hexdigest(),
        "size_bytes": str(stat.st_size),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
    }


def frame_digest(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame (values and row order, not the index)."""
    hashed = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]
