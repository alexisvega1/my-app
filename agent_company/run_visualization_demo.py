#!/usr/bin/env python3
"""
Run Visualization Demo
======================
This script orchestrates the full pipeline for generating and visualizing
a flood-filling network animation.
"""

import os
import numpy as np
import subprocess
import logging
from segmenters.ffn_v2_advanced import AdvancedFFNv2Plugin
from ffn_v2_mathematical_model import MathematicalFFNv2
import torch

# Set base path to ensure modules are found
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in os.sys.path:
    os.sys.path.insert(0, BASE_DIR)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_synthetic_volume_for_vis(path: str, shape: tuple = (100, 100, 100)):
    """Creates a synthetic volume with a clear, linear path to trace."""
    logging.info(f"Creating synthetic volume at {path} with shape {shape}")
    volume = np.zeros(shape, dtype=np.float32)
    # Create a simple, bright Z-axis line for the FFN to follow
    z_line = shape[0] // 2
    y_line = shape[1] // 2
    volume[z_line, y_line, 10:-10] = 1.0
    
    # Add some noise
    volume += np.random.normal(0, 0.1, shape)
    volume = np.clip(volume, 0, 1)
    
    np.save(path, volume)
    logging.info("Synthetic volume created.")
    return path

def main():
    """Main execution function."""
    logging.info("======== Starting Visualization Demo ========")

    # --- 1. Setup Configuration ---
    output_dir = os.path.join(BASE_DIR, "production_output")
    volume_path = os.path.join(output_dir, "synthetic_vis_volume.npy")
    # Model path is relative to the workspace root, not this script's location
    model_path = "best_mathematical_ffn_v2.pt" 
    
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Create Synthetic Data ---
    create_synthetic_volume_for_vis(volume_path)

    # --- 3. Initialize the FFN Plugin and Load the Model ---
    logging.info("Initializing FFN plugin and loading model...")
    ffn_plugin = AdvancedFFNv2Plugin()
    
    # Adjust model path to be relative to the script's execution directory
    # This assumes you run the script from the root of the `my-app` directory
    if not os.path.exists(model_path):
        # Let's try looking inside the agent_company dir as a fallback
        model_path_alt = os.path.join(BASE_DIR, model_path)
        if not os.path.exists(model_path_alt):
            logging.error(f"Model checkpoint not found at '{model_path}' or '{model_path_alt}'.")
            logging.error("Please ensure the model checkpoint is in the root or agent_company directory.")
            return
        model_path = model_path_alt

    # We must load the model state into the correct architecture
    model = MathematicalFFNv2(input_channels=1, output_channels=1, hidden_channels=64, depth=3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    ffn_plugin.model = model
    logging.info("Model loaded successfully.")

    # --- 4. Run Iterative Segmentation to Generate Steps ---
    logging.info("Running segmentation to generate visualization steps...")
    seed_point = (50, 50, 15) # Start seed on the bright line
    
    generated_steps_dir = ffn_plugin.segment_and_visualize(
        volume_path=volume_path,
        output_path=output_dir,
        seed_point=seed_point,
        max_steps=200, # Limit steps for a quick demo
        save_interval=10 # Save every 10 steps
    )

    if not generated_steps_dir:
        logging.error("Failed to generate segmentation steps.")
        return

    logging.info(f"Segmentation steps generated in: {generated_steps_dir}")

    # --- 5. Run the Visualization Script to Create Animation ---
    logging.info("Launching animation viewer...")
    vis_script_path = os.path.join(BASE_DIR, "visualization.py")
    vis_command = [
        "python",
        vis_script_path,
        "--animate",
        generated_steps_dir
    ]
    
    try:
        # We need to run this from the root directory for imports to work
        subprocess.run(vis_command, check=True, cwd=os.path.join(BASE_DIR, '..'))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.error(f"Failed to launch visualization: {e}")
        logging.error("Please ensure napari is installed (`pip install napari[pyqt5]`) and all scripts are in place.")

    logging.info("======== Visualization Demo Finished ========")

if __name__ == "__main__":
    main()