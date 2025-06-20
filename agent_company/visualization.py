#!/usr/bin/env python3
"""
H01 Tracing Results Visualization
================================
Interactive visualization system for H01 connectomics data and tracing results.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path
import argparse

logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    MATPLOTLIB_AVAILABLE = True
    logger.info("Matplotlib available for visualization")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available - basic visualization disabled")

try:
    import napari
    NAPARI_AVAILABLE = True
    logger.info("Napari available for 3D visualization")
except ImportError:
    NAPARI_AVAILABLE = False
    logger.warning("Napari not available - 3D visualization disabled")

class H01Visualizer:
    """Comprehensive H01 tracing results visualizer."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir) if results_dir else None
        self.data = {}
        if self.results_dir:
            self._load_results()
            logger.info(f"H01 Visualizer initialized with data from {results_dir}")
    
    def _load_results(self):
        """Load all available results from the directory."""
        if not self.results_dir or not self.results_dir.exists():
            logger.warning(f"Results directory not found: {self.results_dir}")
            return
        # Simplified loader
        for npy_file in self.results_dir.glob("*.npy"):
            self.data[npy_file.stem] = np.load(npy_file)
            logger.info(f"Loaded: {npy_file.stem}")

    def create_animation_from_steps(self, steps_dir: str, save_path: Optional[str] = None):
        """Creates a 3D animation from a directory of segmentation steps."""
        if not NAPARI_AVAILABLE:
            logger.error("Napari is required for animation. Please install it: pip install napari[pyqt5]")
            return

        step_files = sorted(Path(steps_dir).glob("step_*.npy"))
        if not step_files:
            logger.error(f"No segmentation step files found in {steps_dir}")
            return

        logger.info(f"Loading {len(step_files)} steps for animation...")
        try:
            # Load all steps into a single 4D numpy array (time, z, y, x)
            all_steps = np.stack([np.load(f) for f in step_files], axis=0)
        except ValueError as e:
            logger.error(f"Failed to stack .npy files. They may have different shapes. Error: {e}")
            return


        logger.info("Launching Napari viewer for animation...")
        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(all_steps, name="FFN_Animation", colormap="magenta", blending="additive")
        
        # Set the viewer to 3D rendering mode and animate the time axis
        viewer.dims.current_step = (0, 0, 0, 0)
        viewer.dims.ndisplay = 3
        
        if save_path:
            logger.info(f"Rendering animation to {save_path}. This may take a moment...")
            # This requires napari-animation to be installed: pip install napari-animation
            try:
                # viewer.movie() is deprecated, use animation writing directly
                from napari.animation import Animation
                animation = Animation(viewer)
                animation.capture_sequence(steps=len(step_files))
                animation.save(save_path, fps=10)
                logger.info(f"Successfully saved animation to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save animation: {e}. You may need `pip install napari-animation`.")

        print("\nNapari viewer is active.")
        print(" - Use the slider at the bottom-left to scrub through the animation steps.")
        print(" - Close the Napari window to exit the script.")
        napari.run()

    # Other visualization methods (create_2d_slice_viewer, etc.) would go here
    # For brevity, they are omitted in this replacement.

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="H01 Tracing Results Visualization")
    parser.add_argument("results_dir", type=str, nargs='?', default=None, help="Directory containing H01 processing results (optional)")
    parser.add_argument("--animate", type=str, help="Path to a directory of segmentation steps to animate")
    parser.add_argument("--save_path", type=str, help="Path to save the animation output (e.g., 'animation.gif')")
    
    args = parser.parse_args()
    
    # Handle animation workflow
    if args.animate:
        # We don't need a results_dir for animation, just a visualizer instance
        visualizer = H01Visualizer(None)
        visualizer.create_animation_from_steps(args.animate, args.save_path)
    else:
        print("No animation directory provided. Use the --animate flag to specify a directory of segmentation steps.")
        print("Example: python visualization.py --animate path/to/segmentation_steps")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()