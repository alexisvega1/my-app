"""
Model Serving API for Enhanced Connectomics Pipeline
===================================================

Provides REST API for real-time inference and model serving.
"""

import torch
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import base64
from io import BytesIO

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import load_config, PipelineConfig
from enhanced_pipeline import EnhancedConnectomicsPipeline
from ffn_v2_mathematical_model import MathematicalFFNv2

logger = logging.getLogger(__name__)

# Pydantic models for API
class InferenceRequest(BaseModel):
    """Request model for inference."""
    data: str  # Base64 encoded numpy array
    region_coords: Optional[List[int]] = None
    region_size: Optional[List[int]] = None
    model_path: Optional[str] = None

class InferenceResponse(BaseModel):
    """Response model for inference."""
    result: str  # Base64 encoded numpy array
    processing_time: float
    model_info: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, float]

class ModelServer:
    """
    Model server for serving connectomics models via REST API.
    """
    
    def __init__(self, config_path: Optional[str] = None, 
                 model_path: Optional[str] = None,
                 environment: str = "production"):
        """
        Initialize model server.
        
        Args:
            config_path: Path to configuration file
            model_path: Path to trained model checkpoint
            environment: Environment name
        """
        self.config = load_config(config_path, environment)
        self.model_path = model_path
        self.environment = environment
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.pipeline = None
        self.model = None
        self.data_loader = None
        
        # Setup FastAPI
        self.app = FastAPI(
            title="Connectomics Model Server",
            description="REST API for connectomics model inference",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"Model server initialized on device: {self.device}")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize model on startup."""
            await self._load_model()
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return await self._health_check()
        
        @self.app.post("/inference", response_model=InferenceResponse)
        async def inference(request: InferenceRequest):
            """Run inference on input data."""
            return await self._run_inference(request)
        
        @self.app.get("/model/info")
        async def model_info():
            """Get model information."""
            return await self._get_model_info()
        
        @self.app.post("/model/reload")
        async def reload_model(model_path: str):
            """Reload model from checkpoint."""
            return await self._reload_model(model_path)
    
    async def _load_model(self):
        """Load the trained model."""
        try:
            logger.info("Loading model...")
            
            # Create pipeline
            self.pipeline = EnhancedConnectomicsPipeline(
                config_path=None, 
                environment=self.environment
            )
            
            # Setup components
            if not self.pipeline.setup_data_loader():
                raise RuntimeError("Failed to setup data loader")
            
            if not self.pipeline.setup_model():
                raise RuntimeError("Failed to setup model")
            
            # Load model checkpoint if provided
            if self.model_path and Path(self.model_path).exists():
                self.pipeline.trainer.load_checkpoint(self.model_path)
                logger.info(f"Model loaded from: {self.model_path}")
            
            self.model = self.pipeline.model
            self.data_loader = self.pipeline.data_loader
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def _health_check(self) -> HealthResponse:
        """Perform health check."""
        try:
            # Check model status
            model_loaded = self.model is not None
            
            # Check GPU status
            gpu_available = torch.cuda.is_available()
            
            # Get memory usage
            memory_usage = {}
            if gpu_available:
                memory_usage = {
                    "gpu_total": torch.cuda.get_device_properties(0).total_memory / 1e9,
                    "gpu_allocated": torch.cuda.memory_allocated(0) / 1e9,
                    "gpu_cached": torch.cuda.memory_reserved(0) / 1e9
                }
            
            return HealthResponse(
                status="healthy" if model_loaded else "unhealthy",
                model_loaded=model_loaded,
                gpu_available=gpu_available,
                memory_usage=memory_usage
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="error",
                model_loaded=False,
                gpu_available=torch.cuda.is_available(),
                memory_usage={}
            )
    
    async def _run_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference on input data."""
        try:
            start_time = time.time()
            
            # Decode input data
            data_bytes = base64.b64decode(request.data)
            data = np.load(BytesIO(data_bytes))
            
            # Convert to tensor
            data_tensor = torch.from_numpy(data).float()
            if data_tensor.dim() == 3:
                data_tensor = data_tensor.unsqueeze(0)  # Add batch dimension
            
            # Move to device
            data_tensor = data_tensor.to(self.device)
            
            # Normalize data
            data_tensor = (data_tensor - data_tensor.min()) / (data_tensor.max() - data_tensor.min() + 1e-8)
            
            # Run inference
            self.model.eval()
            with torch.no_grad():
                output = self.model(data_tensor)
            
            # Convert output to numpy
            output_np = output.cpu().numpy()
            
            # Encode output
            output_bytes = BytesIO()
            np.save(output_bytes, output_np)
            output_encoded = base64.b64encode(output_bytes.getvalue()).decode()
            
            processing_time = time.time() - start_time
            
            # Get model info
            model_info = {
                "input_shape": data.shape,
                "output_shape": output_np.shape,
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "device": str(self.device)
            }
            
            return InferenceResponse(
                result=output_encoded,
                processing_time=processing_time,
                model_info=model_info
            )
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        try:
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                "model_type": "MathematicalFFNv2",
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "device": str(self.device),
                "config": self.config.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _reload_model(self, model_path: str) -> Dict[str, str]:
        """Reload model from checkpoint."""
        try:
            if not Path(model_path).exists():
                raise HTTPException(status_code=404, detail="Model file not found")
            
            # Load new model
            self.pipeline.trainer.load_checkpoint(model_path)
            self.model = self.pipeline.model
            
            logger.info(f"Model reloaded from: {model_path}")
            
            return {"status": "success", "message": f"Model reloaded from {model_path}"}
            
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the model server."""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )


def create_model_server(config_path: Optional[str] = None,
                       model_path: Optional[str] = None,
                       environment: str = "production") -> ModelServer:
    """
    Create a model server instance.
    
    Args:
        config_path: Path to configuration file
        model_path: Path to trained model checkpoint
        environment: Environment name
        
    Returns:
        ModelServer instance
    """
    return ModelServer(config_path, model_path, environment)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Connectomics Model Server")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--environment", type=str, default="production",
                       choices=["development", "production", "colab"],
                       help="Environment to run in")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    # Create and run server
    server = create_model_server(args.config, args.model, args.environment)
    server.run(args.host, args.port) 