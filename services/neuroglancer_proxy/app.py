#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow local web app to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class LayerSpec(BaseModel):
    precomputed_path: str
    layer_name: str
    layer_type: str  # 'image' | 'segmentation'


class PublishMultiRequest(BaseModel):
    layers: list[LayerSpec]


@app.post("/publish_multi")
async def publish_multi(req: PublishMultiRequest):
    try:
        import neuroglancer
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"neuroglancer not installed: {exc}")

    # Validate
    if not req.layers:
        raise HTTPException(status_code=400, detail="layers are required")
    for layer in req.layers:
        if not os.path.isabs(layer.precomputed_path):
            raise HTTPException(status_code=400, detail=f"precomputed_path must be absolute: {layer.precomputed_path}")
        if not os.path.isdir(layer.precomputed_path):
            raise HTTPException(status_code=404, detail=f"precomputed_path does not exist: {layer.precomputed_path}")

    neuroglancer.set_server_bind_address("0.0.0.0", 0)
    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        for layer in req.layers:
            src = f"precomputed://file://{layer.precomputed_path}"
            if layer.layer_type.lower() == "segmentation":
                s.layers[layer.layer_name] = neuroglancer.SegmentationLayer(source=src)
            else:
                s.layers[layer.layer_name] = neuroglancer.ImageLayer(source=src)

    return {"neuroglancer_url": str(viewer)}


@app.post("/export_demo")
async def export_demo():
    """Create small demo image+segmentation precomputed datasets and return viewer URLs."""
    try:
        import numpy as np
        from libs.neuroglancer.export_precomputed import (
            write_precomputed_image,
            write_precomputed_segmentation,
        )
        import neuroglancer
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"missing deps: {exc}")

    base = "/tmp/ng_mvp_demo"
    img_dir = os.path.join(base, "image")
    seg_dir = os.path.join(base, "seg")
    os.makedirs(base, exist_ok=True)

    # Create synthetic image and segmentation
    z, y, x = 64, 128, 128
    img = (np.random.rand(z, y, x) * 255).astype(np.uint8)
    # Simple blobs for labels
    labels = np.zeros((z, y, x), dtype=np.uint32)
    labels[:, 16:80, 16:80] = 1
    labels[:, 48:112, 48:112] = np.where(labels[:, 48:112, 48:112] == 0, 2, labels[:, 48:112, 48:112])

    img_url = write_precomputed_image(img, img_dir)
    seg_url = write_precomputed_segmentation(labels, seg_dir)

    # Build combined viewer
    neuroglancer.set_server_bind_address("0.0.0.0", 0)
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.layers["image"] = neuroglancer.ImageLayer(source=img_url)
        s.layers["segmentation"] = neuroglancer.SegmentationLayer(source=seg_url)

    return {
        "image_path": img_dir,
        "seg_path": seg_dir,
        "image_url": img_url,
        "seg_url": seg_url,
        "viewer_url": str(viewer),
    }

