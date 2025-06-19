import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_nd
import random
from scipy.ndimage import gaussian_filter

# Parameters
shape = (128, 128, 128)
num_neurons = 8
min_length = 60
max_length = 100
branch_prob = 0.3

segmentation = np.zeros(shape, dtype=np.int32)
traces = []

for neuron_id in range(1, num_neurons + 1):
    # Start at a random point near the bottom
    start = np.array([
        random.randint(10, shape[0] - 10),
        random.randint(10, shape[1] - 10),
        random.randint(5, 15)
    ])
    points = [start]
    current = start.copy()
    length = random.randint(min_length, max_length)
    direction = np.random.randn(3)
    direction[2] = abs(direction[2])  # Prefer upward
    direction /= np.linalg.norm(direction)
    for i in range(length):
        # Random walk with some persistence
        if random.random() < branch_prob and len(points) > 10:
            # Branch: start a new trace from here
            branch_dir = direction + 0.5 * np.random.randn(3)
            branch_dir /= np.linalg.norm(branch_dir)
            branch_len = random.randint(10, 20)
            branch_points = [current.copy()]
            for j in range(branch_len):
                branch_dir += 0.1 * np.random.randn(3)
                branch_dir /= np.linalg.norm(branch_dir)
                next_pt = branch_points[-1] + branch_dir
                next_pt = np.clip(next_pt, 0, np.array(shape) - 1)
                branch_points.append(next_pt)
            branch_points = np.array(branch_points).astype(int)
            for k in range(len(branch_points) - 1):
                rr, cc, zz = line_nd(branch_points[k], branch_points[k + 1], endpoint=True)
                segmentation[rr, cc, zz] = neuron_id
            points.extend(branch_points)
        # Continue main trace
        direction += 0.1 * np.random.randn(3)
        direction /= np.linalg.norm(direction)
        next_pt = current + direction
        next_pt = np.clip(next_pt, 0, np.array(shape) - 1)
        points.append(next_pt)
        current = next_pt
    points = np.array(points).astype(int)
    traces.append(points)
    # Rasterize trace
    for k in range(len(points) - 1):
        rr, cc, zz = line_nd(points[k], points[k + 1], endpoint=True)
        segmentation[rr, cc, zz] = neuron_id

np.save('simulated_neurolucida_segmentation.npy', segmentation)

# Optional: visualize a projection
proj = np.max(segmentation, axis=2)
plt.figure(figsize=(6,6))
plt.imshow(proj, cmap='tab10')
plt.title('Simulated Neurolucida-Style Neuron Traces (Projection)')
plt.axis('off')
plt.savefig('simulated_neurolucida_projection.png', dpi=150)
plt.close()
print('Simulated segmentation and projection saved.') 