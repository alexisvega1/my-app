#!/usr/bin/env python3
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PublishRequest(BaseModel):
    layer_path: str
    auth_token: str

@app.post('/publish')
async def publish(req: PublishRequest):
    # TODO: validate token, generate signed URL
    signed = f"https://viewer/?layer={req.layer_path}&sig=placeholder"
    return {"neuroglancer_url": signed}
