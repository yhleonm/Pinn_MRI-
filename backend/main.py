"""FastAPI Server Entrypoint with GPU Lifespan Management and CORS."""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import AsyncIterator
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes_kspace import gpu_session, router as kspace_router
from .api.ws_training import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Inicializa el espacio-k multi-bobina en la VRAM de la RTX 4060 Ti al arrancar."""
    gpu_session.initialize_default()
    yield
    if gpu_session.trainer is not None:
        gpu_session.trainer.stop_requested = True
    gpu_session.last_maps.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Medical Insight MRI | PINN-SENSE + Complex Helmholtz EPT",
    description="Backend cientifico en PyTorch + FastAPI para reconstruccion MRI desde espacio-k y mapeo cuantitativo EPT.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(kspace_router)
app.include_router(ws_router)


@app.get("/health")
async def health_check() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
