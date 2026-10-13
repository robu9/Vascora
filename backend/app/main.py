from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import DISCLAIMER, settings
from app.routes import inference, meta, metrics, samples

API_PREFIX = "/api"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Vascora API",
        version="0.1.0",
        description=f"CAD and coronary stenosis risk prediction. {DISCLAIMER}",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for module in (meta, inference, metrics, samples):
        app.include_router(module.router, prefix=API_PREFIX)
    return app


app = create_app()
