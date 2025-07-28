from typing import Optional
import os
import base64
import requests
from openai import AsyncOpenAI

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing OPENAI_API_KEY. Set it in your environment before calling generate_image()."
            )
        _client = AsyncOpenAI(api_key=api_key)
    return _client


async def generate_image(prompt: str) -> str:
    """Generate an image and return base64 (PNG by default)."""
    client = _get_client()
    res = await client.images.generate(
        model="dall-e-2",
        prompt=prompt
    )

    b64 = getattr(res.data[0], "b64_json", None) if res.data else None
    if not b64:
        url = getattr(res.data[0], "url", None) if res.data else None
        if not url:
            raise RuntimeError("No image returned (neither b64_json nor url)")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        b64 = base64.b64encode(resp.content).decode("ascii")
    return b64