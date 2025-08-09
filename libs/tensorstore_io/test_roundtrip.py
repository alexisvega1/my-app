#!/usr/bin/env python3
import os
import tempfile
import numpy as np
import pytest
from libs.tensorstore_io.readwrite import write_precomputed, read_precomputed

pytestmark = pytest.mark.skipif("tensorstore" not in globals(), reason="tensorstore missing")

def test_roundtrip_n5():
    with tempfile.TemporaryDirectory() as td:
        vol = (np.random.rand(32,32,32)*255).astype(np.uint8)
        meta = write_precomputed(os.path.join(td, "vol.n5"), vol)
        back = read_precomputed(os.path.join(td, "vol.n5"))
        assert back.shape == vol.shape
        assert back.dtype == vol.dtype
        assert np.allclose(back, vol)
