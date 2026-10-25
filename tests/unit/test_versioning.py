import hashlib

import pandas as pd

from loan_approval.data.versioning import file_fingerprint, frame_digest


def test_file_fingerprint_matches_sha256_and_changes_with_content(tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes(b"a,b\n1,2\n")
    fp = file_fingerprint(path, chunk_size=3)
    assert fp["sha256"] == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    assert fp["size_bytes"] == "8" and fp["file"] == "data.csv"

    path.write_bytes(b"a,b\n1,3\n")
    assert file_fingerprint(path)["sha256"] != fp["sha256"]


def test_frame_digest_is_stable_and_content_sensitive():
    df = pd.DataFrame({"x": [1.0, 2.0], "c": pd.Categorical(["a", "b"])})
    assert frame_digest(df) == frame_digest(df.copy().set_axis([10, 11]))  # index ignored
    assert frame_digest(df) != frame_digest(df.assign(x=[1.0, 2.5]))
    assert frame_digest(df) != frame_digest(df.iloc[::-1])  # row order matters
