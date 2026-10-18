"""Validate the completed visual FAISS, metadata, vectors, and source mappings."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from visual_retrieval import ROOT, VISUAL_FAISS_PATH, VISUAL_METADATA_PATH


def main() -> None:
    """Fail unless the independent visual index is internally consistent."""
    payload = json.loads(VISUAL_METADATA_PATH.read_text(encoding="utf-8"))
    items = payload["items"]
    index = faiss.read_index(str(VISUAL_FAISS_PATH))
    assert index.ntotal == 2504
    assert index.d == 4096
    assert len(items) == index.ntotal
    assert [item["vector_id"] for item in items] == list(range(index.ntotal))
    assert len({item["image_id"] for item in items}) == index.ntotal
    assert all((ROOT / item["relative_path"]).is_file() for item in items)
    assert all(item["section_keys"] for item in items)
    vectors = np.empty((index.ntotal, index.d), dtype=np.float32)
    index.reconstruct_n(0, index.ntotal, vectors)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.isfinite(vectors).all()
    assert np.allclose(norms, 1.0, atol=1e-5)
    print(
        "visual_index_check=OK "
        f"ntotal={index.ntotal} dim={index.d} metadata={len(items)} "
        f"norm_min={norms.min():.6f} norm_max={norms.max():.6f}"
    )


if __name__ == "__main__":
    main()
