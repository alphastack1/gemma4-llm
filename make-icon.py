"""Generate gemma4.ico — blue star icon matching the app's branding."""

from PIL import Image, ImageDraw
from pathlib import Path

OUT = Path(__file__).parent / "gemma4.ico"

BG = (9, 9, 11, 255)          # #09090b  (dark background)
ACCENT = (59, 130, 246, 255)   # #3b82f6  (blue accent)


def render(size):
    img = Image.new("RGBA", (size, size), BG)
    d = ImageDraw.Draw(img)
    s = size / 24.0  # viewport scale (SVG viewBox is 24x24)

    # Star polygon: M12,2 l2.4,7.4 H22 l-6.2,4.5 2.4,7.4 L12,16.8 l-6.2,4.5 2.4-7.4 L2,9.4 h7.6z
    star_points = [
        (12, 2),
        (14.4, 9.4),
        (22, 9.4),
        (15.8, 13.9),
        (18.2, 21.3),
        (12, 16.8),
        (5.8, 21.3),
        (8.2, 13.9),
        (2, 9.4),
        (9.6, 9.4),
    ]

    scaled = [(x * s, y * s) for x, y in star_points]
    d.polygon(scaled, fill=ACCENT)

    return img


def main():
    sizes = [256, 128, 64, 48, 32, 16]
    images = [render(s) for s in sizes]
    images[0].save(OUT, format="ICO", sizes=[(s, s) for s in sizes])
    print(f"  wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(f"  sizes: {sizes}")


if __name__ == "__main__":
    main()
