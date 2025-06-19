import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import json
import os

# Load background image
bg = imread('simulated_neurolucida_projection.png')

# Load neuron traces from JSON
with open('simulated_tracing_results/neuron_traces.json') as f:
    traces = json.load(f)

# Create a blank overlay
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(bg, cmap='gray')

# Assign colors
colors = plt.cm.tab10(np.linspace(0, 1, len(traces)))

# Overlay traces (projected to XY)
for i, (nid, trace) in enumerate(traces.items()):
    # Load coordinates (N, 3)
    coords = np.array(trace['centroid'])
    # For overlay, use centroid as a marker (for full centerline, would need to load all skeleton points)
    ax.plot(coords[1], coords[0], 'o', color=colors[i], label=f'Neuron {nid}')

ax.set_title('Overlay: Centerline Traces on Background')
ax.axis('off')
ax.legend()
plt.tight_layout()
plt.savefig('overlay_projection.png', dpi=150)
plt.close()
print('Overlay image saved as overlay_projection.png') 