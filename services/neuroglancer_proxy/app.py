#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class PublishRequest(BaseModel):
    # Absolute path to a local precomputed layer directory, e.g., /abs/path/to/precomp/image
    precomputed_path: str
    # Optional human-friendly layer name
    layer_name: Optional[str] = "image"
    # Optional: 'image' or 'segmentation'
    layer_type: Optional[str] = "image"


@app.post("/publish")
async def publish(req: PublishRequest):
    # Validate path
    if not req.precomputed_path:
        raise HTTPException(status_code=400, detail="precomputed_path is required")
    if not os.path.isabs(req.precomputed_path):
        raise HTTPException(status_code=400, detail="precomputed_path must be absolute")
    if not os.path.isdir(req.precomputed_path):
        raise HTTPException(status_code=404, detail="precomputed_path does not exist")

    # Generate a Neuroglancer share URL using the Python API.
    # We avoid starting a server here; we simply encode a state with a single local layer.
    try:
        import neuroglancer
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"neuroglancer not installed: {exc}")

    # Configure neuroglancer to generate a URL
    neuroglancer.set_server_bind_address("0.0.0.0", 0)
    viewer = neuroglancer.Viewer()
    precomputed_url = f"precomputed://file://{req.precomputed_path}"

    with viewer.txn() as s:
        if (req.layer_type or "image").lower() == "segmentation":
            s.layers[req.layer_name or "segmentation"] = neuroglancer.SegmentationLayer(
                source=precomputed_url
            )
        else:
            s.layers[req.layer_name or "image"] = neuroglancer.ImageLayer(source=precomputed_url)

    return {"neuroglancer_url": str(viewer)}

