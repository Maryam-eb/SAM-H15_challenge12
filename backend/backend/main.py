"""VisionVerse AI FastAPI application."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.api.routes import router
from backend.config.settings import settings
from backend.models.model_loader import ModelLoader, get_status


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.PRELOAD_MODELS:
        await asyncio.to_thread(ModelLoader().load_all)
    else:
        print(
            "Models load lazily on the first caption request. "
            "Set PRELOAD_MODELS=true to load them at startup."
        )
    yield


app = FastAPI(title="VisionVerse AI", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    return {"success": True, "status": "ok", **get_status()}


if settings.SERVE_FRONTEND and settings.FRONTEND_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(settings.FRONTEND_DIR / "index.html")

    app.mount(
        "/",
        StaticFiles(directory=str(settings.FRONTEND_DIR), html=True),
        name="frontend",
    )
else:
    @app.get("/", include_in_schema=False)
    def index():
        return JSONResponse({
            "success": True,
            "message": "VisionVerse AI API. Frontend serving is disabled.",
        })
