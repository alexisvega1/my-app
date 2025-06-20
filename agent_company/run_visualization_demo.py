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
from agent_company.segmenters.ffn_v2_advanced import AdvancedFFNv2Plugin
from agent_company.ffn_v2_mathematical_model import MathematicalFFNv2
import torch

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
    output_dir = "agent_company/production_output"
    volume_path = os.path.join(output_dir, "synthetic_vis_volume.npy")
    model_path = "agent_company/best_mathematical_ffn_v2.pt" # Using the best model we trained
    steps_dir = os.path.join(output_dir, "segmentation_steps")

    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Create Synthetic Data ---
    create_synthetic_volume_for_vis(volume_path)

    # --- 3. Initialize the FFN Plugin and Load the Model ---
    logging.info("Initializing FFN plugin and loading model...")
    ffn_plugin = AdvancedFFNv2Plugin()
    
    if not os.path.exists(model_path):
        logging.error(f"Model checkpoint not found at {model_path}. Please ensure it exists.")
        return
        
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
    vis_command = [
        "python",
        "agent_company/visualization.py",
        "--animate",
        generated_steps_dir
    ]
    
    try:
        subprocess.run(vis_command, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.error(f"Failed to launch visualization: {e}")
        logging.error("Please ensure napari is installed (`pip install napari[pyqt5]`) and all scripts are in place.")

    logging.info("======== Visualization Demo Finished ========")

if __name__ == "__main__":
    main() 