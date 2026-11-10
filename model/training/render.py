"""Turn Quick Draw strokes into model input images.

This is the training-side twin of ``renderForModel`` in the website
(``AIPlaygroundSection.tsx``) followed by the API's preprocessing. Keeping the two
visually identical is what makes the model see the same pictures in training and in
production. Only Pillow and numpy are used, so it runs anywhere and is easy to test.

A drawing is a list of strokes; a stroke is ``[xs, ys]`` (Quick Draw "simplified" format).
The result is a ``uint8`` array where 255 is ink and 0 is background, which is exactly what
the API feeds the model after inverting the black-on-white picture.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def _points(drawing):
    return [(float(x), float(y)) for xs, ys, *_ in drawing for x, y in zip(xs, ys, strict=True)]


def render_drawing(
    drawing,
    image_size: int,
    *,
    canvas: int = 280,
    margin: int = 25,
    line_width: float = 22,
    min_extent: float = 10,
) -> np.ndarray:
    """Render ``drawing`` as a ``(image_size, image_size)`` uint8 array (255 = ink)."""
    points = _points(drawing)
    if not points:
        return np.zeros((image_size, image_size), dtype=np.uint8)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    extent = max(max_x - min_x, max_y - min_y, min_extent)
    scale = (canvas - 2 * margin) / extent
    offset_x = canvas / 2 - (min_x + max_x) / 2 * scale
    offset_y = canvas / 2 - (min_y + max_y) / 2 * scale

    image = Image.new("L", (canvas, canvas), 255)
    draw = ImageDraw.Draw(image)
    radius = line_width / 2

    for stroke_x, stroke_y, *_ in drawing:
        path = [
            (x * scale + offset_x, y * scale + offset_y)
            for x, y in zip(stroke_x, stroke_y, strict=True)
        ]
        if len(path) == 1:  # a single tap is a dot
            path.append((path[0][0] + 0.01, path[0][1]))
        draw.line(path, fill=0, width=round(line_width), joint="curve")
        for cx, cy in (path[0], path[-1]):  # round caps
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=0)

    # Same shrink as the API (Image.Resampling.LANCZOS), then invert to ink = 255.
    small = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return 255 - np.asarray(small, dtype=np.uint8)
