"""HTTP routes for caption generation and service health."""

import asyncio
import ipaddress
import socket
from typing import Optional
from urllib.parse import urljoin, urlparse

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from backend.config.settings import settings
from backend.models import digital_twin_model
from backend.models.captioning_model import process_caption_request
from backend.models.model_loader import get_status

router = APIRouter(prefix="/api")
_INFERENCE_SEMAPHORE = asyncio.Semaphore(settings.INFERENCE_CONCURRENCY)


async def _read_upload(image: UploadFile) -> bytes:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "Only image uploads are supported")

    data = await image.read(settings.max_upload_bytes + 1)
    if not data:
        raise HTTPException(400, "The uploaded image was empty")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "Image exceeds the %d MB limit" % settings.MAX_UPLOAD_MB)
    return data


def _host_allowed(hostname: str) -> bool:
    allowed = [item.lower().strip(".") for item in settings.REMOTE_IMAGE_ALLOWED_HOSTS]
    if not allowed:
        return True
    host = hostname.lower().strip(".")
    return any(host == item or host.endswith("." + item) for item in allowed)


async def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(400, "image_url must be a valid http(s) URL")
    if parsed.username or parsed.password:
        raise HTTPException(400, "Credentials are not allowed in image_url")
    if not _host_allowed(parsed.hostname):
        raise HTTPException(403, "Remote image host is not allow-listed")

    try:
        infos = await asyncio.to_thread(
            socket.getaddrinfo,
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except Exception as exc:
        raise HTTPException(502, "Could not resolve image host: %s" % exc)

    addresses = {info[4][0] for info in infos if info and info[4]}
    if not addresses:
        raise HTTPException(502, "Image host resolved to no address")

    for address in addresses:
        try:
            ip = ipaddress.ip_address(address.split("%", 1)[0])
        except ValueError:
            raise HTTPException(400, "Image host resolved to an invalid address")
        if not ip.is_global:
            raise HTTPException(403, "Private or local image URLs are not allowed")


async def _fetch_image_url(url: str) -> bytes:
    """Download a public remote image without blocking the event loop."""

    if not settings.ALLOW_REMOTE_IMAGE_URLS:
        raise HTTPException(400, "Remote image URLs are disabled on this server")

    try:
        import httpx
    except ImportError:
        raise HTTPException(500, "httpx is required for remote image downloads")

    current_url = url
    timeout = httpx.Timeout(settings.REQUEST_TIMEOUT)
    headers = {"User-Agent": "VisionVerse-AI/2.0"}

    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=False) as client:
        for redirect_count in range(settings.MAX_REDIRECTS + 1):
            await _validate_public_url(current_url)
            try:
                async with client.stream("GET", current_url) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if not location:
                            raise HTTPException(502, "Image URL returned an empty redirect")
                        if redirect_count >= settings.MAX_REDIRECTS:
                            raise HTTPException(502, "Image URL exceeded the redirect limit")
                        current_url = urljoin(current_url, location)
                        continue

                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if content_type and not content_type.startswith("image/"):
                        raise HTTPException(400, "That URL did not return an image")

                    content_length = response.headers.get("content-length")
                    if content_length:
                        try:
                            if int(content_length) > settings.max_upload_bytes:
                                raise HTTPException(
                                    413,
                                    "Image exceeds the %d MB limit" % settings.MAX_UPLOAD_MB,
                                )
                        except ValueError:
                            pass

                    data = bytearray()
                    async for chunk in response.aiter_bytes(64 * 1024):
                        data.extend(chunk)
                        if len(data) > settings.max_upload_bytes:
                            raise HTTPException(
                                413,
                                "Image exceeds the %d MB limit" % settings.MAX_UPLOAD_MB,
                            )

                    if not data:
                        raise HTTPException(400, "The image URL returned no data")
                    return bytes(data)
            except HTTPException:
                raise
            except httpx.HTTPError as exc:
                raise HTTPException(502, "Could not download the image URL: %s" % exc)

    raise HTTPException(502, "Could not download the image URL")


async def _run_inference(image_bytes, industry, audience, tone, language):
    # Transformer inference is synchronous and CPU/GPU heavy. Running it in a
    # worker thread keeps FastAPI's event loop responsive; the semaphore avoids
    # launching too many simultaneous model generations on one device.
    async with _INFERENCE_SEMAPHORE:
        return await asyncio.to_thread(
            process_caption_request,
            image_bytes,
            industry,
            audience,
            tone,
            language,
        )


async def run(request: Request, image: Optional[UploadFile], industry: str,
              audience: str, tone: str, language: str):
    """Shared handler. Accepts multipart uploads or JSON with image_url."""

    if image is not None:
        image_bytes = await _read_upload(image)
    else:
        try:
            body = await request.json()
        except Exception:
            body = None

        if not isinstance(body, dict) or not body.get("image_url"):
            raise HTTPException(400, "Send an image file or a JSON body with image_url")

        image_bytes = await _fetch_image_url(str(body["image_url"]))
        industry = body.get("industry", industry)
        audience = body.get("audience", audience)
        tone = body.get("tone", tone)
        language = body.get("language", language)

    try:
        return await _run_inference(image_bytes, industry, audience, tone, language)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, "Caption generation failed: %s" % exc)


@router.post("/caption")
async def caption(
    request: Request,
    image: Optional[UploadFile] = File(None),
    industry: str = Form("General"),
    audience: str = Form("General"),
    tone: str = Form("Professional"),
    language: str = Form("English"),
):
    return await run(request, image, industry, audience, tone, language)


@router.post("/generate-captions")
async def generate(
    request: Request,
    image: Optional[UploadFile] = File(None),
    industry: str = Form("General"),
    audience: str = Form("General"),
    tone: str = Form("Professional"),
    language: str = Form("English"),
):
    return await run(request, image, industry, audience, tone, language)


@router.post("/digital-twin/analyze")
async def digital_twin(file: Optional[UploadFile] = File(None)):
    data = await file.read() if file is not None else None
    return digital_twin_model.analyze(data, file.filename if file else None)


@router.get("/status")
def status():
    return {"success": True, "api": "ok", **get_status()}


@router.get("/health")
def api_health():
    return {"success": True, "status": "ok"}
