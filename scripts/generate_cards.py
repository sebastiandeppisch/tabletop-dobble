"""Generate synthetic Dobble photos with ground truth for testing the matcher."""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
SYMBOLS = list(
    "🍎🍌🍇🍉🍓🥕🌽🌶🍄🌵🌲🌻🌹🍀🔥💧⛄🌙⭐🌈⚡☂🎈🎁🔑🔒💡⏰📌✂🖍📎🎸🎺🥁⚽🏀🎲🧩🚗🚲⚓✈🚀🐶🐱🐭🐸🐢🐟🐞🦋🐘🦒🦉👻💀👁👄❤❓❗🔔🏠"
)
SYMBOLS_PER_CARD = 8
# size range, spacing and margin relative to the card radius
LAYOUTS = {
    "sparse": (0.28, 0.5, 0.7, 0.05),
    "medium": (0.34, 0.56, 0.58, 0.02),
    "dense": (0.4, 0.62, 0.42, -0.02),
}


def render_symbol(char: str, size: int, angle: float, outline: bool) -> Image.Image:
    font = ImageFont.truetype(EMOJI_FONT, 109)
    canvas = Image.new("RGBA", (180, 180), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).text((90, 90), char, font=font, embedded_color=True, anchor="mm")
    if outline:
        alpha = canvas.getchannel("A").point(lambda a: 255 if a > 40 else 0).filter(ImageFilter.MaxFilter(7))
        stroke = Image.new("RGBA", canvas.size, (35, 30, 30, 255))
        stroke.putalpha(alpha)
        canvas = Image.alpha_composite(stroke, canvas)
    canvas = canvas.crop(canvas.getbbox())
    scale = size / max(canvas.size)
    canvas = canvas.resize((max(1, int(canvas.width * scale)), max(1, int(canvas.height * scale))), Image.LANCZOS)
    return canvas.rotate(angle, expand=True, resample=Image.BICUBIC)


def place_symbols(card_radius: int, chars: list[str], rng: random.Random, layout: str) -> list[dict]:
    placed: list[dict] = []
    low, high, spacing, margin = LAYOUTS[layout]
    for char in chars:
        for _ in range(800):
            size = rng.randint(int(card_radius * low), int(card_radius * high))
            r = size * spacing
            dist = rng.uniform(0, card_radius * 0.92 - size * 0.45)
            phi = rng.uniform(0, 2 * math.pi)
            x, y = dist * math.cos(phi), dist * math.sin(phi)
            if all(math.hypot(x - p["x"], y - p["y"]) > r + p["r"] + card_radius * margin for p in placed):
                placed.append({"char": char, "x": x, "y": y, "r": r, "size": size})
                break
        else:
            return []
    return placed


def background(w: int, h: int, rng: random.Random) -> Image.Image:
    base = np.array([rng.randint(60, 150), rng.randint(40, 110), rng.randint(20, 80)], dtype=np.float32)
    noise = np.random.default_rng(rng.randint(0, 10**6)).normal(0, 12, (h // 8, w // 8, 1)).astype(np.float32)
    noise = np.array(Image.fromarray(noise[:, :, 0]).resize((w, h), Image.BICUBIC))[:, :, None]
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(rows, dtype=np.float64))
    return (vt[-1] / vt[-1, -1]).reshape(3, 3)


def tilt(img: Image.Image, truth: dict, strength: float, rng: random.Random) -> Image.Image:
    w, h = img.size
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    top = rng.uniform(0, strength) * w
    dst = src + np.array([[top, rng.uniform(-0.03, 0.03) * h], [-top, rng.uniform(-0.03, 0.03) * h], [0, 0], [0, 0]])
    forward = homography(src, dst)
    inverse = np.linalg.inv(forward)
    inverse /= inverse[2, 2]
    warped = img.transform((w, h), Image.PERSPECTIVE, tuple(inverse.flatten()[:8]), Image.BICUBIC, fillcolor=(90, 70, 50))

    def project(x: float, y: float) -> tuple[float, float]:
        p = forward @ np.array([x, y, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    for card in truth["cards"]:
        card["cx"], card["cy"] = project(card["cx"], card["cy"])
        for s in card["symbols"]:
            s["x"], s["y"] = project(s["x"], s["y"])
    return warped


def generate(index: int, rng: random.Random, out_dir: Path, perspective: float, layout: str) -> None:
    w, h = 1600, 1000
    img = background(w, h, rng)
    shared = rng.choice(SYMBOLS)
    pool = [s for s in SYMBOLS if s != shared]
    rng.shuffle(pool)
    decks = [
        [shared] + pool[: SYMBOLS_PER_CARD - 1],
        [shared] + pool[SYMBOLS_PER_CARD - 1 : 2 * (SYMBOLS_PER_CARD - 1)],
    ]
    radius = rng.randint(330, 390)
    gap = rng.randint(-10, 60)
    cy = h // 2 + rng.randint(-40, 40)
    centers = [(w // 2 - radius - gap // 2, cy + rng.randint(-50, 50)), (w // 2 + radius + gap // 2, cy + rng.randint(-50, 50))]

    truth = {"shared": shared, "cards": []}
    draw = ImageDraw.Draw(img)
    for (cx, cy_), chars in zip(centers, decks):
        rng.shuffle(chars)
        draw.ellipse((cx - radius, cy_ - radius, cx + radius, cy_ + radius), fill=(250, 248, 242), outline=(200, 200, 195), width=3)
        while not (symbols := place_symbols(radius, chars, rng, layout)):
            pass
        card_truth = {"cx": cx, "cy": cy_, "r": radius, "symbols": []}
        for s in symbols:
            sprite = render_symbol(s["char"], s["size"], rng.uniform(0, 360), layout == "dense")
            px, py = int(cx + s["x"] - sprite.width / 2), int(cy_ + s["y"] - sprite.height / 2)
            img.paste(sprite, (px, py), sprite)
            card_truth["symbols"].append({"char": s["char"], "x": cx + s["x"], "y": cy_ + s["y"], "r": s["r"]})
        truth["cards"].append(card_truth)

    if perspective > 0:
        img = tilt(img, truth, perspective, rng)

    arr = np.asarray(img).astype(np.float32)
    gx = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.9, 1.1), w)[None, :, None]
    arr = arr * gx + np.random.default_rng(index).normal(0, 4, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    img.save(out_dir / f"card_{index:02d}.jpg", quality=88)
    (out_dir / f"card_{index:02d}.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent.parent / "tests" / "fixtures"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    style = sys.argv[4] if len(sys.argv) > 4 else "mixed"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for i in range(count):
        layout = style if style != "mixed" else ("dense" if i % 4 >= 2 else "sparse")
        generate(i, rng, out, perspective=0.0 if i % 2 == 0 else 0.22, layout=layout)
