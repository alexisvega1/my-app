#!/usr/bin/env python3
from __future__ import annotations

from typing import Tuple, Literal, Optional
import os
import numpy as np


def _ensure_cloudvolume() -> None:
    try:
        import cloudvolume  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "CloudVolume is required for precomputed export. Install with: pip install cloud-volume"
        ) from exc


def _create_info(
    *,
    layer_type: Literal["image", "segmentation"],
    dtype: str,
    num_channels: int,
    resolution_nm: Tuple[float, float, float],
    chunk_size: Tuple[int, int, int],
    volume_size: Tuple[int, int, int],
    encoding: Optional[str] = None,
):
    from cloudvolume import CloudVolume

    if layer_type == "image":
        encoding = encoding or "raw"
    else:
        encoding = encoding or "compressed_segmentation"

    return CloudVolume.create_new_info(
        num_channels=num_channels,
        layer_type=layer_type,
        data_type=dtype,
        encoding=encoding,
        resolution=list(resolution_nm),  # nm
        voxel_offset=[0, 0, 0],
        chunk_size=list(chunk_size),
        volume_size=list(volume_size),
    )


def write_precomputed_image(
    volume_zyx: np.ndarray,
    out_dir: str,
    *,
    voxel_size_nm: Tuple[float, float, float] = (8.0, 8.0, 8.0),
    chunk_size: Tuple[int, int, int] = (64, 64, 64),
    encoding: str = "raw",
) -> str:
    """
    Write a grayscale image volume to Neuroglancer precomputed using CloudVolume.

    Args:
      volume_zyx: numpy array with shape (Z, Y, X) or (Z, Y, X, C)
      out_dir: destination directory (will be created). Use an absolute path.
      voxel_size_nm: physical voxel size
      chunk_size: chunk dimensions in voxels (X, Y, Z) for storage
      encoding: CloudVolume encoding for image layer

    Returns:
      The precomputed URL (file scheme) like 'precomputed://file:///...'
    """
    _ensure_cloudvolume()
    from cloudvolume import CloudVolume

    if volume_zyx.ndim == 3:
        z, y, x = volume_zyx.shape
        num_channels = 1
        data = volume_zyx[..., None]  # (Z, Y, X, 1)
    elif volume_zyx.ndim == 4:
        z, y, x, num_channels = volume_zyx.shape
        data = volume_zyx
    else:
        raise ValueError("volume must have 3 or 4 dimensions")

    os.makedirs(out_dir, exist_ok=True)

    info = _create_info(
        layer_type="image",
        dtype=str(data.dtype),
        num_channels=num_channels,
        resolution_nm=voxel_size_nm,
        chunk_size=chunk_size,
        volume_size=(x, y, z),  # CloudVolume uses XYZ order
        encoding=encoding,
    )

    vol = CloudVolume(f"file://{out_dir}", info=info, progress=False)
    vol.commit_info()
    vol.commit_provenance()

    # CloudVolume expects XYZC order on assignment, with indexing [x:y, y:y, z:z]
    # Convert (Z, Y, X, C) -> (X, Y, Z, C)
    xyzc = np.transpose(data, (2, 1, 0, 3))
    vol[:, :, :] = xyzc

    return f"precomputed://file://{os.path.abspath(out_dir)}"


def write_precomputed_segmentation(
    labels_zyx: np.ndarray,
    out_dir: str,
    *,
    voxel_size_nm: Tuple[float, float, float] = (8.0, 8.0, 8.0),
    chunk_size: Tuple[int, int, int] = (64, 64, 64),
    encoding: str = "compressed_segmentation",
) -> str:
    """
    Write a label volume to Neuroglancer precomputed using CloudVolume.

    Args:
      labels_zyx: numpy array with shape (Z, Y, X), integer dtype
      out_dir: destination directory (will be created)
      voxel_size_nm: physical voxel size
      chunk_size: chunk dimensions in voxels (X, Y, Z)
      encoding: segmentation encoding

    Returns:
      The precomputed URL (file scheme) like 'precomputed://file:///...'
    """
    _ensure_cloudvolume()
    from cloudvolume import CloudVolume

    if labels_zyx.ndim != 3:
        raise ValueError("labels must have shape (Z, Y, X)")
    if not np.issubdtype(labels_zyx.dtype, np.integer):
        raise ValueError("labels dtype must be integer")

    z, y, x = labels_zyx.shape
    os.makedirs(out_dir, exist_ok=True)

    info = _create_info(
        layer_type="segmentation",
        dtype=str(labels_zyx.dtype),
        num_channels=1,
        resolution_nm=voxel_size_nm,
        chunk_size=chunk_size,
        volume_size=(x, y, z),
        encoding=encoding,
    )

    vol = CloudVolume(f"file://{out_dir}", info=info, progress=False)
    vol.commit_info()
    vol.commit_provenance()

    # Convert (Z, Y, X) -> (X, Y, Z, 1)
    xyz = np.transpose(labels_zyx, (2, 1, 0))
    xyzc = xyz[..., None]
    vol[:, :, :] = xyzc

    return f"precomputed://file://{os.path.abspath(out_dir)}"


