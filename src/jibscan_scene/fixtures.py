from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def generate_synthetic_squid_rgba(path: str | Path, size: tuple[int, int] = (128, 256)) -> Path:
    """Create the deterministic v0 RGBA geometry fixture.

    This is deliberately schematic. It exists only to test geometry and alpha,
    and must never be interpreted as a biological or measured sample.
    """
    if size != (128, 256):
        raise ValueError("v0 synthetic fixture currently supports only 128x256")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((38, 20, 90, 166), fill=(220, 92, 76, 255), outline=(255, 185, 160, 255), width=3)
    draw.polygon([(38, 68), (12, 112), (42, 106)], fill=(188, 63, 66, 235))
    draw.polygon([(90, 68), (116, 112), (86, 106)], fill=(188, 63, 66, 235))
    draw.ellipse((44, 146, 84, 188), fill=(205, 76, 70, 255))
    for x, dx in [(49, -18), (57, -8), (65, 4), (73, 14), (81, 22)]:
        draw.line((x, 178, x + dx, 238), fill=(212, 82, 74, 255), width=5)
        draw.ellipse((x + dx - 2, 234, x + dx + 3, 240), fill=(235, 130, 112, 240))
    draw.ellipse((52, 154, 59, 161), fill=(18, 18, 18, 255))
    draw.ellipse((70, 154, 77, 161), fill=(18, 18, 18, 255))
    image.save(output)
    return output
