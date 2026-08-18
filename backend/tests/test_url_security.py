import asyncio

import pytest
from fastapi import HTTPException

from backend.api.routes import _validate_public_url


def test_rejects_localhost():
    with pytest.raises(HTTPException):
        asyncio.run(_validate_public_url("http://127.0.0.1/image.jpg"))


def test_rejects_file_scheme():
    with pytest.raises(HTTPException):
        asyncio.run(_validate_public_url("file:///etc/passwd"))
