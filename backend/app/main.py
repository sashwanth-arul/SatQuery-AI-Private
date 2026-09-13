from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import analysis, health, imagery, query
from app.core.config import get_settings
from app.core.errors import SatQueryError

settings = get_settings()

app = FastAPI(title=settings.app_name, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(imagery.router, prefix=settings.api_prefix)
app.include_router(query.router, prefix=settings.api_prefix)
app.include_router(analysis.router, prefix=settings.api_prefix)


import logging

logger = logging.getLogger("app.main")


@app.exception_handler(SatQueryError)
async def satquery_error_handler(_request: Request, exc: SatQueryError) -> JSONResponse:
    if exc.status_code >= 500:
        logger.error("SatQueryError %s: %s", exc.code, exc.message)
    return JSONResponse(status_code=exc.status_code, content=exc.to_response())


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "server_error",
                "message": str(exc) or "An unexpected server error occurred.",
                "user_message": "The SatQuery backend encountered an unexpected error. Try again or check server logs.",
                "field": None,
            },
        },
    )
