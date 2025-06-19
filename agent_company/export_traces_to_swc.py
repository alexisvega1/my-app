import json
import numpy as np
import os

# Load neuron traces
with open('simulated_tracing_results/neuron_traces.json') as f:
    traces = json.load(f)

# Load segmentation
seg = np.load('simulated_neurolucida_segmentation.npy')

# Fallback: 2D skeletonization if 3D not available
try:
    from skimage.morphology import skeletonize_3d
    use_3d = True
except ImportError:
    from skimage.morphology import skeletonize
    use_3d = False

swc_dir = 'simulated_tracing_results'

for nid, trace in traces.items():
    neuron_id = int(nid)
    mask = (seg == neuron_id)
    if np.sum(mask) == 0:
        continue
    # Skeletonize
    if use_3d:
        skeleton = skeletonize_3d(mask)
    else:
        skeleton = np.zeros_like(mask, dtype=bool)
        for z in range(mask.shape[2]):
            skeleton[:, :, z] = skeletonize(mask[:, :, z])
    skel_coords = np.array(np.where(skeleton)).T
    if len(skel_coords) < 1:
        continue
    # Sort by Z for a simple chain
    skel_coords = skel_coords[np.argsort(skel_coords[:,2])]
    swc_lines = []
    for i, pt in enumerate(skel_coords):
        n = i+1
        t = 3 if i > 0 else 1  # 1=soma, 3=axon/dendrite
        x, y, z = pt
        r = 1.0
        parent = n-1 if i > 0 else -1
        swc_lines.append(f"{n} {t} {x:.2f} {y:.2f} {z:.2f} {r:.2f} {parent}\n")
    swc_path = os.path.join(swc_dir, f'neuron_{nid}_skeleton.swc')
    with open(swc_path, 'w') as f:
        f.write("# Exported full skeleton from simulated neuron traces\n")
        f.write("# n T X Y Z R PARENT\n")
        for line in swc_lines:
            f.write(line)
    print(f'Exported {swc_path} ({len(swc_lines)} nodes)') 