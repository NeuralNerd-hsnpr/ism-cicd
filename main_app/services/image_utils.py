from PIL import Image,ImageDraw,ImageChops
from collections import deque
import numpy as np
import math
import cv2 as cv
import random


def subtract_masks(mask1: Image.Image, mask2: Image.Image) -> Image.Image:
        """
        Given two PIL mask images (mode ‘1’, ‘L’ or ‘RGB’, black & white),
        return a new mask equal to mask1 minus mask2’s white area.

        White pixels in mask2 will be turned to black in the output.
        """
        # Ensure both are in single‑channel “L” mode (0–255)
        m1 = mask1.convert('L')
        m2 = mask2.convert('L')

        # If these aren't strictly binary (0 or 255), threshold them:
        m1 = m1.point(lambda p: 255 if p > 128 else 0)
        m2 = m2.point(lambda p: 255 if p > 128 else 0)

        # Subtract: white (255) in m2 will zero out corresponding m1 pixels.
        result = ImageChops.subtract(m1, m2)

        return result


def calculate_pixels_per_cm(
        base_pixel_width, base_pixel_height, actual_width_cm, actual_height_cm
    ):
        """
        Calculate detected_object_pixels_per_cm based on the base rectangle's
        pixel dimensions and its real-world size.
        """
        pixels_per_cm_x = base_pixel_width / actual_width_cm
        pixels_per_cm_y = base_pixel_height / actual_height_cm
        return (pixels_per_cm_x + pixels_per_cm_y) / 2

def resize_mask_rectangle(
        mask_image,
        object_width_cm,
        object_height_cm,
        position,
        include_original=False,
        detected_object_pixels_per_cm=None,
        actual_width_cm=None,
        actual_height_cm=None,
        placement_rotation=None,
        behind=False,
        cap_pose=None,  # New parameter: None, 'left', or 'right'
        margin_ratio=0.02,
        shrink_threshold=0.4,
        enforce_margins=False,
    ):
        """
        Resize, position, and rotate a new rectangle relative to a base rectangle in a mask.
        Allows optional enforcement of margin-based bounds to avoid touching image borders.
        Optionally extend the appended box toward left or right by 20% of its width when cap_pose is set.
        """
        # Convert mask and get dimensions
        mask = mask_image.convert("RGB")
        width, height = mask.size

        # Define margins in pixels
        margin_x = width * margin_ratio if enforce_margins else 0
        margin_y = height * margin_ratio if enforce_margins else 0

        # Find base rectangle bbox
        mask_data = mask.load()
        xmin, ymin, xmax, ymax = width, height, 0, 0
        for y in range(height):
            for x in range(width):
                if mask_data[x, y] == (255, 255, 255):
                    xmin = min(xmin, x)
                    ymin = min(ymin, y)
                    xmax = max(xmax, x)
                    ymax = max(ymax, y)

        base_pixel_width = xmax - xmin
        base_pixel_height = ymax - ymin

        # Pixels per cm conversion
        if detected_object_pixels_per_cm is None:
            if actual_width_cm is None or actual_height_cm is None:
                raise ValueError(
                    "If detected_object_pixels_per_cm is not provided, actual_width_cm and actual_height_cm must be given."
                )
            detected_object_pixels_per_cm = calculate_pixels_per_cm(
                base_pixel_width, base_pixel_height,
                actual_width_cm, actual_height_cm
            )

        # New rectangle size in pixels
        new_w = object_width_cm * detected_object_pixels_per_cm
        new_h = object_height_cm * detected_object_pixels_per_cm

        # Mid-lines
        x_mid = (xmin + xmax) / 2
        y_mid = (ymin + ymax) / 2

        # Determine initial placement
        if not behind:
            if position == "top":
                raw_xmin = x_mid - new_w / 2
                raw_ymin = ymin - new_h
                anchor = (x_mid, ymin)
            elif position == "top_right":
                raw_xmin = xmax
                raw_ymin = ymin - new_h
                anchor = (xmax, ymin)
            elif position == "right":
                raw_xmin = xmax
                raw_ymin = y_mid - new_h / 2
                anchor = (xmax, y_mid)
            elif position == "bottom_right":
                raw_xmin = xmax
                raw_ymin = ymax
                anchor = (xmax, ymax)
            elif position == "bottom":
                raw_xmin = x_mid - new_w / 2
                raw_ymin = ymax
                anchor = (x_mid, ymax)
            elif position == "bottom_left":
                raw_xmin = xmin - new_w
                raw_ymin = ymax
                anchor = (xmin, ymax)
            elif position == "left":
                raw_xmin = xmin - new_w
                raw_ymin = y_mid - new_h / 2
                anchor = (xmin, y_mid)
            elif position == "top_left":
                raw_xmin = xmin - new_w
                raw_ymin = ymin - new_h
                anchor = (xmin, ymin)
            elif position == "center":
                offset_pct = 0.20
                y_anchor = y_mid + offset_pct * (ymax - ymin)
                raw_xmin = x_mid - new_w / 2
                raw_ymin = y_anchor - new_h
                anchor = (x_mid, y_anchor)
            else:
                raise ValueError(f"Invalid position value: {position}")
            raw_xmax = raw_xmin + new_w
            raw_ymax = raw_ymin + new_h
        else:
            # Similar logic for 'behind' placement
            if position == "left":
                raw_xmax = x_mid
                raw_xmin = raw_xmax - new_w
                raw_ymin = y_mid - new_h / 2
            elif position == "right":
                raw_xmin = x_mid
                raw_ymin = y_mid - new_h / 2
                raw_xmax = raw_xmin + new_w
                raw_ymax = raw_ymin + new_h
            elif position == "top":
                raw_ymax = y_mid
                raw_ymin = raw_ymax - new_h
                raw_xmin = x_mid - new_w / 2
                raw_xmax = raw_xmin + new_w
            elif position == "bottom":
                raw_ymin = y_mid
                raw_xmin = x_mid - new_w / 2
                raw_xmax = raw_xmin + new_w
                raw_ymax = raw_ymin + new_h
            elif position == "top_left":
                raw_xmax = x_mid
                raw_ymax = y_mid
                raw_xmin = raw_xmax - new_w
                raw_ymin = raw_ymax - new_h
            elif position == "top_right":
                raw_xmin = x_mid
                raw_ymax = y_mid
                raw_xmax = raw_xmin + new_w
                raw_ymin = raw_ymax - new_h
            elif position == "bottom_left":
                raw_xmax = x_mid
                raw_ymin = y_mid
                raw_xmin = raw_xmax - new_w
                raw_ymax = raw_ymin + new_h
            elif position == "bottom_right":
                raw_xmin = x_mid
                raw_ymin = y_mid
                raw_xmax = raw_xmin + new_w
                raw_ymax = raw_ymin + new_h
            elif position == "center":
                raw_xmin = x_mid - new_w / 2
                raw_ymin = y_mid - new_h
                raw_xmax = raw_xmin + new_w
                raw_ymax = raw_ymin + new_h
                anchor = (x_mid, y_mid)
            else:
                raise ValueError(f"Invalid position for behind: {position}")
            if position != "center":
                anchor = ((raw_xmin + raw_xmax) / 2, (raw_ymin + raw_ymax) / 2)

        # Shrink if exceeds bounds
        max_w = width - 2 * margin_x
        max_h = height - 2 * margin_y
        scale_w = max_w / new_w if new_w > 0 else 1
        scale_h = max_h / new_h if new_h > 0 else 1
        scale = min(scale_w, scale_h, 1)
        if scale < 1 - shrink_threshold:
            scale = 1 - shrink_threshold
        if scale < 1:
            new_w *= scale
            new_h *= scale
            raw_xmin = anchor[0] - (anchor[0] - raw_xmin) * scale
            raw_ymin = anchor[1] - (anchor[1] - raw_ymin) * scale
            raw_xmax = raw_xmin + new_w
            raw_ymax = raw_ymin + new_h

        # Apply cap_pose extension from base adjusted box
        if cap_pose in ('left', 'right'):
            base_w = raw_xmax - raw_xmin
            extension = base_w * 0.1
            if cap_pose == 'left':
                raw_xmin -= extension
            else:
                raw_xmax += extension

        # Clamp to margin area if enforced
        if enforce_margins:
            new_xmin = max(raw_xmin, margin_x)
            new_ymin = max(raw_ymin, margin_y)
            new_xmax = min(raw_xmax, width - margin_x)
            new_ymax = min(raw_ymax, height - margin_y)
        else:
            new_xmin, new_ymin, new_xmax, new_ymax = raw_xmin, raw_ymin, raw_xmax, raw_ymax

        new_w = new_xmax - new_xmin
        new_h = new_ymax - new_ymin

        # Prepare corners
        corners = [
            (new_xmin, new_ymin), (new_xmax, new_ymin),
            (new_xmax, new_ymax), (new_xmin, new_ymax)
        ]

        # Apply rotation
        if placement_rotation is None:
            placement_rotation = random.uniform(-10, 10)
        if placement_rotation:
            angle = math.radians(placement_rotation)
            rotated = []
            for x, y in corners:
                rx = anchor[0] + (x - anchor[0]) * math.cos(angle) - (y - anchor[1]) * math.sin(angle)
                ry = anchor[1] + (x - anchor[0]) * math.sin(angle) + (y - anchor[1]) * math.cos(angle)
                if enforce_margins:
                    rx = min(max(rx, margin_x), width - margin_x)
                    ry = min(max(ry, margin_y), height - margin_y)
                rotated.append((rx, ry))
            corners = rotated

        # Draw
        canvas = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(canvas)
        if include_original:
            draw.rectangle([(xmin, ymin), (xmax, ymax)], fill="white")
        draw.polygon(corners, fill="white")

        return canvas

def get_bounding_box_masked_image(masked_image: Image.Image) -> Image.Image:
        """
        Extracts the bounding box of a masked (black and white) image and returns an updated mask image
        where only the bounding box area is white, and the rest of the mask is black.

        Args:
            masked_image (Image.Image): A black and white image where white (255) represents the mask.

        Returns:
            Image.Image: Mask image with only the bounding box area set to white.
        """
        # Convert PIL image to numpy array
        masked_array = np.array(masked_image)

        if len(masked_array.shape) != 2:
            raise ValueError("The input image must be a grayscale image.")

        # Find the coordinates of the white mask (value 255)
        coords = np.column_stack(np.where(masked_array == 255))

        if coords.size == 0:
            raise ValueError("No mask found in the image.")

        # Get the bounding box coordinates
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Create a new mask of the same size as the input image
        bounding_box_mask = Image.new("L", masked_image.size, 0)
        draw = ImageDraw.Draw(bounding_box_mask)

        # Draw a white rectangle for the bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], fill=255)

        return bounding_box_mask

def composite_images(foreground_pil: Image.Image, background_pil: Image.Image) -> Image.Image:
    """
    Composites a foreground image (with transparency) onto a background image.
    The background image is resized to match the foreground's size.
    """
    # Resize background to match foreground
    if background_pil.size != foreground_pil.size:
        background_pil = background_pil.resize(foreground_pil.size, Image.LANCZOS)
    
    # Ensure both are RGBA
    foreground_pil = foreground_pil.convert("RGBA")
    background_pil = background_pil.convert("RGBA")
    
    # Composite
    return Image.alpha_composite(background_pil, foreground_pil)

def filter_mask(base_mask: Image.Image, filter_masks: list[Image.Image]) -> Image.Image:
        """
        Removes white areas (255) from base_mask wherever any of the filter_masks have white.

        Args:
            base_mask (PIL.Image): Base black and white mask (mode "L").
            filter_masks (List[PIL.Image]): List of masks to exclude areas from base_mask.

        Returns:
            PIL.Image: A new mask with excluded areas removed.
        """
        # Convert base mask to numpy array
        base_array = np.array(base_mask).astype(np.uint8)

        for mask in filter_masks:
            # Ensure the mask is same size as base
            mask = mask.resize(base_mask.size).convert("L")
            filter_array = np.array(mask).astype(np.uint8)

            # Any white (255) pixel in the filter_mask will set that pixel in base to 0 (black)
            base_array[filter_array == 255] = 0

        # Convert back to PIL Image
        result_mask = Image.fromarray(base_array, mode="L")
        return result_mask

def apply_filter(image_pil: Image.Image, filter_type: str) -> Image.Image:
    """
    Applies a named filter to a PIL Image.
    This function acts as a dispatcher, calling the correct private function
    based on the filter_type string.
    """
    print(f"IMAGE_UTILS: Applying filter of type '{filter_type}'...")
                    
    # Create a mapping from the filter name string to the actual function
    filter_map = {
       "Gaussian Blur": _apply_gaussian_blur,
        "Sharpen": _apply_sharpening_filter,
        "Edge Detection":_apply_edge_detection,
        "Invert": _apply_invert_filter,
        "Brightness": _adjust_brightness_contrast,
        "Grayscale":_apply_grayscale_filter,
        "Sepia":_apply_sepia,
        "Posterize": _apply_posterize,
        "Emboss":_apply_emboss,
        "Dilate":_apply_dilate,
        "Erode":_apply_erode,
        "Film Grain":_apply_film_grain,
        "Fisheye":_apply_fisheye,
        "Pixelate":_apply_pixelate,
        "Heatmap":_apply_heatmap,
        "Mosaic":_apply_mosaic,
        "Cartoon Effect":_apply_cartoon,
        "Chalk Drawing":_apply_chalk_drawing,
        "Sketch":_apply_sketch,
        "Soft Focus":_apply_soft_focus,
        "Solarize":_apply_solarize,
    }

    # Get the function from the map
    filter_function = filter_map.get(filter_type)
    
    if filter_function:
        # If the filter exists, call it
        return filter_function(image_pil)
    else:
        # If the filter name is unknown, log a warning and return the original image
        print(f"WARNING: Unknown filter type '{filter_type}'. Returning original image.")
        return image_pil
    
def load_image_from_url(url: str) -> Image.Image:
    """A utility to load an image from a URL, needed by the task."""
    import httpx
    from io import BytesIO
    try:
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Failed to load image from {url}: {e}")
        raise

def create_transparent_highlight(base_image: Image.Image, mask_image: Image.Image, highlight_color=(0, 255, 0, 128)) -> Image.Image:
        """
        Overlay a semi-transparent color on the masked area of the base image.

        Parameters:
            base_image (PIL.Image): The original image (RGB or RGBA).
            mask_image (PIL.Image): Black-and-white mask where white = masked area.
            highlight_color (tuple): RGBA tuple for the highlight color.
                                    Default is semi-transparent green.

        Returns:
            PIL.Image: RGBA image with highlighted masked area.
        """
        # Ensure base image is RGBA
        base_rgba = base_image.convert("RGBA")
        
        # Ensure the mask is in grayscale
        mask_gray = mask_image.convert("L")
        
        # Create a solid color overlay of the highlight color
        highlight_layer = Image.new("RGBA", base_rgba.size, highlight_color)
        
        # Composite the highlight layer onto the base image using the mask
        result = Image.composite(highlight_layer, base_rgba, mask_gray)
        
        return result

def pil_to_cv(image):
        """Convert PIL image to OpenCV format."""
        return cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

def cv_to_pil(image):
    """Convert OpenCV image to PIL format."""
    return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

def _apply_gaussian_blur(image):
    frame = pil_to_cv(image)
    result = cv.GaussianBlur(frame, (15, 15), 0)
    return cv_to_pil(result)

def _apply_sharpening_filter(image, strength=1):
    frame = pil_to_cv(image)
    base_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    kernel = base_kernel + (strength - 1) * np.array(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    )
    result = cv.filter2D(frame, -1, kernel)
    return cv_to_pil(result)

def _apply_edge_detection(image):
    frame = pil_to_cv(image)
    edges = cv.Canny(frame, 100, 200)
    return Image.fromarray(edges)  # Return a grayscale PIL image

def _apply_invert_filter(image):
    frame = pil_to_cv(image)
    result = cv.bitwise_not(frame)
    return cv_to_pil(result)

def _adjust_brightness_contrast(image, alpha=1.0, beta=0):
    frame = pil_to_cv(image)
    result = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)
    return cv_to_pil(result)

def _apply_grayscale_filter(image):
    frame = pil_to_cv(image)
    result = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return Image.fromarray(result)

def _apply_sepia(image):
    frame = pil_to_cv(image)
    kernel = np.array(
        [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
    )
    result = cv.transform(frame, kernel)
    return cv_to_pil(result)

def _apply_posterize(image, levels=4):
    frame = np.array(image).astype(np.float32)
    posterized = np.floor(frame / (256 / levels)) * (256 / levels)
    return Image.fromarray(posterized.astype(np.uint8))

def _apply_emboss(image):
    frame = pil_to_cv(image)
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    result = cv.filter2D(frame, -1, kernel)
    return cv_to_pil(result)

def _apply_dilate(image):
    frame = pil_to_cv(image)
    kernel = np.ones((5, 5), np.uint8)
    result = cv.dilate(frame, kernel, iterations=1)
    return cv_to_pil(result)

def _apply_erode(image):
    frame = pil_to_cv(image)
    kernel = np.ones((5, 5), np.uint8)
    result = cv.erode(frame, kernel, iterations=1)
    return cv_to_pil(result)

def _apply_film_grain(image, grain_intensity=0.5, grain_size=8):
    frame = pil_to_cv(image)
    h, w, c = frame.shape
    noise = np.random.uniform(0, 1, (h // grain_size, w // grain_size, c)).astype(
        np.float32
    )
    noise = cv.resize(noise, (w, h), interpolation=cv.INTER_LINEAR)
    noise = (noise * (grain_intensity * 255)).astype(np.uint8)
    noisy_image = cv.add(frame, noise)
    return cv_to_pil(np.clip(noisy_image, 0, 255))

def _apply_fisheye(image):
    frame = pil_to_cv(image)
    h, w = frame.shape[:2]
    K = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]], dtype=np.float32)
    D = np.array([0.3, -0.3, 0, 0], dtype=np.float32)
    result = cv.fisheye.undistortImage(frame, K, D)
    return cv_to_pil(result)

def _apply_pixelate(image, pixel_size=10):
    frame = pil_to_cv(image)
    h, w = frame.shape[:2]
    frame = cv.resize(
        frame, (w // pixel_size, h // pixel_size), interpolation=cv.INTER_LINEAR
    )
    result = cv.resize(frame, (w, h), interpolation=cv.INTER_NEAREST)
    return cv_to_pil(result)

def _apply_heatmap(image):
    frame = pil_to_cv(image)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    result = cv.applyColorMap(gray, cv.COLORMAP_JET)
    return cv_to_pil(result)

def _apply_mosaic(image, block_size=10):
    frame = pil_to_cv(image)
    h, w = frame.shape[:2]
    frame = cv.resize(
        frame, (w // block_size, h // block_size), interpolation=cv.INTER_LINEAR
    )
    result = cv.resize(frame, (w, h), interpolation=cv.INTER_NEAREST)
    return cv_to_pil(result)

def _apply_cartoon(image):
    frame = pil_to_cv(image)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 7)
    edges = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 10
    )
    color = cv.bilateralFilter(frame, 9, 250, 250)
    result = cv.bitwise_and(color, color, mask=edges)
    return cv_to_pil(result)

def _apply_chalk_drawing(image):
    frame = pil_to_cv(image)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    inv_gray = cv.bitwise_not(gray)
    blurred = cv.GaussianBlur(inv_gray, (21, 21), 0)
    result = cv.divide(gray, 255 - blurred, scale=256)
    return Image.fromarray(result)

def _apply_sketch(image):
    frame = pil_to_cv(image)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    inv_gray = cv.bitwise_not(gray)
    blurred = cv.GaussianBlur(inv_gray, (21, 21), 0)
    result = cv.divide(gray, blurred, scale=256)
    return Image.fromarray(result)

def _apply_soft_focus(image):
    frame = pil_to_cv(image)
    blurred = cv.GaussianBlur(frame, (15, 15), 0)
    result = cv.addWeighted(frame, 0.7, blurred, 0.3, 0)
    return cv_to_pil(result)

def _apply_solarize(image, threshold=128):
    frame = np.array(image)
    solarized = np.where(frame < threshold, frame, 255 - frame)
    return Image.fromarray(solarized.astype(np.uint8))