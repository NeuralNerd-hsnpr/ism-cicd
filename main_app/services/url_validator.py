# services/url_validator.py

import asyncio
import httpx
from fastapi import HTTPException
from typing import List
from app.schemas import ImageTask # Assuming your schemas.py is in the parent 'app' directory

async def _check_url_accessibility(session: httpx.AsyncClient, url: str):
    """
    Performs an asynchronous HEAD request to check if a URL is accessible
    and returns an image content type.
    """
    try:
        # A HEAD request is very fast as it only fetches headers, not the image body.
        response = await session.head(url, timeout=10, follow_redirects=True)
        
        # Raise an exception for any 4xx or 5xx error (e.g., 404 Not Found)
        response.raise_for_status()

        # Check if the server indicated it's sending an image
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"The URL '{url}' does not point to a valid image (Content-Type: {content_type})."
            )
        
    except httpx.RequestError as e:
        # This catches DNS errors, connection refused, etc.
        raise HTTPException(
            status_code=400,
            detail=f"The URL could not be reached: '{url}'. Reason: {e.__class__.__name__}"
        )
    except httpx.HTTPStatusError as e:
        # This catches 4xx/5xx responses like 404 Not Found, 403 Forbidden.
        raise HTTPException(
            status_code=400,
            detail=f"The URL is not accessible: '{url}'. Server responded with status {e.response.status_code}."
        )

async def validate_image_urls_are_accessible(images: List[ImageTask]):
    """
    Validates that all base_url and mask_url fields in the request are live and
    point to actual images by making concurrent HEAD requests.
    Raises HTTPException if any URL fails validation.
    """
    urls_to_check = set()
    for image in images:
        urls_to_check.add(str(image.base_url))
        for operation in image.operations:
            urls_to_check.add(str(operation.mask_url))

    async with httpx.AsyncClient() as session:
        # Create a list of tasks to run concurrently.
        tasks = [_check_url_accessibility(session, url) for url in urls_to_check]
        
        # asyncio.gather runs all checks in parallel for maximum speed.
        await asyncio.gather(*tasks)

    # If we get here, all URLs were valid.
    print("All image URLs in the request were successfully validated.")