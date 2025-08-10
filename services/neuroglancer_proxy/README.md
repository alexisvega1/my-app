Neuroglancer proxy service (MVP)
================================

This FastAPI endpoint returns a Neuroglancer viewer URL for a local precomputed dataset.

Quickstart

1) Install deps (from project root):

```
pip install -r requirements.txt
```

2) Export a small test volume to precomputed using CloudVolume:

```
python - <<'PY'
import os, numpy as np
from libs.neuroglancer.export_precomputed import write_precomputed_image
out_dir = "/tmp/ng_demo/image"
vol = (np.random.rand(64,64,64)*255).astype('uint8')
url = write_precomputed_image(vol, out_dir, voxel_size_nm=(8,8,8))
print("precomputed:", url)
PY
```

3) Launch the API and request a viewer URL:

```
uvicorn services.neuroglancer_proxy.app:app --host 0.0.0.0 --port 8099
```

POST JSON to http://localhost:8099/publish

```
{
  "precomputed_path": "/tmp/ng_demo/image",
  "layer_name": "image",
  "layer_type": "image"
}
```

The response contains `neuroglancer_url`. Open it in a browser.

Notes
- The exporter writes a single-scale precomputed layer for MVP. For production, add multi-resolution mip generation.
- Use absolute paths for `precomputed_path`.

