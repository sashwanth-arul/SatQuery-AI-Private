"""Generates three high-fidelity synthetic aerial sample images for GeoVision controlled demonstrations.
Outputs to:
  - frontend/public/demo-aerial/
  - backend/data/demo-samples/
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DEST = REPO_ROOT / "frontend" / "public" / "demo-aerial"
BACKEND_DEST = REPO_ROOT / "backend" / "data" / "demo-samples"


def create_urban_sample(width: int = 800, height: int = 600) -> Image.Image:
    """Creates a realistic aerial urban scene with 4 major buildings, 3 tree clusters,

    connected roads, and open ground.
    """
    img = Image.new("RGB", (width, height), color=(180, 175, 160)) # Open ground / dry earth base
    draw = ImageDraw.Draw(img)

    # 1. Add textured open ground (sand/grass noise)
    noise = np.random.RandomState(42).normal(0, 8, (height, width, 3)).astype(np.int16)
    base_arr = np.array(img, dtype=np.int16) + noise
    base_arr = np.clip(base_arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(base_arr)
    draw = ImageDraw.Draw(img)

    # 2. Road Network (dark asphalt with yellow lane markings)
    # Horizontal road: y = 260 to 330
    draw.rectangle([0, 260, width, 330], fill=(55, 58, 62))
    # Vertical connecting road: x = 370 to 440
    draw.rectangle([370, 0, 440, height], fill=(55, 58, 62))
    # Road markings (dashed white / yellow)
    for x in range(0, width, 40):
        draw.line([(x, 295), (x + 20, 295)], fill=(230, 230, 230), width=2)
    for y in range(0, height, 40):
        draw.line([(405, y), (405, y + 20)], fill=(230, 230, 230), width=2)

    # 3. Four Prominent Building Structures (Distinct rooftops, borders, shadows)
    # Building 1: Top-Left (Commercial complex)
    # Shadow
    draw.rectangle([75, 55, 315, 225], fill=(30, 32, 35))
    # Main roof
    draw.rectangle([60, 40, 300, 210], fill=(215, 218, 224), outline=(130, 135, 145), width=3)
    # Rooftop HVAC / utilities
    draw.rectangle([90, 70, 160, 130], fill=(160, 165, 175))
    draw.rectangle([200, 120, 270, 180], fill=(150, 155, 165))

    # Building 2: Top-Right (Large industrial warehouse - Largest)
    draw.rectangle([495, 55, 755, 225], fill=(30, 32, 35))
    draw.rectangle([480, 40, 740, 210], fill=(195, 140, 125), outline=(120, 80, 70), width=3)
    # Ridged roof texture
    for rx in range(500, 720, 25):
        draw.line([(rx, 45), (rx, 205)], fill=(175, 120, 105), width=2)

    # Building 3: Bottom-Left (Office block)
    draw.rectangle([75, 385, 315, 555], fill=(30, 32, 35))
    draw.rectangle([60, 370, 300, 540], fill=(225, 225, 230), outline=(140, 145, 155), width=3)
    draw.rectangle([110, 410, 250, 500], fill=(185, 190, 200), outline=(150, 155, 165), width=2)

    # Building 4: Bottom-Right (Residential / administrative complex)
    draw.rectangle([515, 385, 735, 555], fill=(30, 32, 35))
    draw.rectangle([500, 370, 720, 540], fill=(210, 205, 195), outline=(135, 130, 120), width=3)
    draw.rectangle([540, 400, 680, 510], fill=(170, 165, 155), outline=(130, 125, 115), width=2)

    # 4. Tree Canopy Clusters (Rich green, organic circular canopies)
    # Tree Cluster 1: Along North-West open ground
    for cx, cy, r in [(330, 80, 22), (345, 110, 26), (335, 145, 24), (325, 175, 20)]:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(34, 125, 45), outline=(20, 85, 30), width=2)
        draw.ellipse([cx - r//2, cy - r//2, cx + r//3, cy + r//3], fill=(45, 155, 60))

    # Tree Cluster 2: Bottom-Center park/median
    for cx, cy, r in [(330, 410, 25), (340, 450, 28), (330, 490, 24), (325, 520, 20)]:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(28, 115, 40), outline=(18, 75, 25), width=2)
        draw.ellipse([cx - r//2, cy - r//2, cx + r//3, cy + r//3], fill=(40, 145, 55))

    # Tree Cluster 3: East boulevard buffer
    for cx, cy, r in [(465, 400, 24), (460, 440, 26), (465, 480, 25), (460, 515, 22)]:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(32, 120, 42), outline=(22, 80, 28), width=2)
        draw.ellipse([cx - r//2, cy - r//2, cx + r//3, cy + r//3], fill=(42, 150, 58))

    return img


def create_water_sample(width: int = 800, height: int = 600) -> Image.Image:
    """Creates a realistic aerial coastal/lake scene with a large continuous water body,

    shoreline vegetation, surrounding terrain, and nearby road.
    """
    img = Image.new("RGB", (width, height), color=(195, 185, 165)) # Sandy soil / shoreline
    draw = ImageDraw.Draw(img)

    # 1. Texture for terrain
    noise = np.random.RandomState(43).normal(0, 6, (height, width, 3)).astype(np.int16)
    arr = np.clip(np.array(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # 2. Large Continuous Water Body on the Right (~38% of image)
    # Winding organic coastal shoreline
    water_poly = [
        (460, 0), (800, 0), (800, 600), (520, 600),
        (510, 520), (490, 460), (470, 400), (485, 340),
        (475, 270), (460, 200), (445, 140), (455, 70),
    ]
    draw.polygon(water_poly, fill=(28, 95, 165), outline=(22, 75, 135))

    # Inner deep water gradient
    deep_water = [
        (560, 0), (800, 0), (800, 600), (600, 600),
        (580, 480), (560, 360), (550, 220), (540, 100),
    ]
    draw.polygon(deep_water, fill=(20, 72, 138))

    # Water shimmer / waves
    for y in range(20, 580, 35):
        draw.line([(580 + (y % 40), y), (780, y)], fill=(40, 115, 185), width=2)

    # 3. Shoreline Vegetation / Mangrove Buffer
    for px, py, r in [
        (440, 60, 18), (430, 110, 22), (440, 160, 20), (455, 220, 24),
        (465, 280, 22), (455, 350, 25), (460, 410, 22), (480, 470, 24),
        (495, 530, 26), (505, 570, 20)
    ]:
        draw.ellipse([px - r, py - r, px + r, py + r], fill=(30, 115, 45), outline=(18, 70, 28), width=2)
        draw.ellipse([px - r//2, py - r//2, px + r//3, py + r//3], fill=(42, 145, 60))

    # 4. Coastal Road (Asphalt)
    draw.rectangle([160, 0, 220, height], fill=(60, 63, 68))
    for y in range(0, height, 40):
        draw.line([(190, y), (190, y + 20)], fill=(230, 230, 230), width=2)

    # 5. Small coastal observatory / building
    draw.rectangle([260, 240, 360, 340], fill=(220, 220, 225), outline=(130, 130, 140), width=2)
    draw.rectangle([285, 265, 335, 315], fill=(175, 175, 185))

    return img


def create_forest_sample(width: int = 800, height: int = 600) -> Image.Image:
    """Creates a realistic aerial forest scene with dense canopy (~82% coverage),

    varied tree clusters, and a winding dirt road/path.
    """
    # 1. Base terrain / earth background
    img = Image.new("RGB", (width, height), color=(175, 155, 125)) # Earth tan clearing
    draw = ImageDraw.Draw(img)

    # 2. Rich canopy texture with diverse green shades covering ~82% of the scene
    np.random.seed(44)
    # Exclude central winding corridor (clearing & path)
    for _ in range(520):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        # Skip if directly inside the clearing path corridor
        path_y_est = 290 + int(30 * np.sin(cx / 60.0))
        if abs(cy - path_y_est) < 32 and np.random.rand() > 0.15:
            continue
        # Also leave a scenic clearing in bottom right
        if cx > 680 and cy > 480 and np.random.rand() > 0.2:
            continue
        r = np.random.randint(18, 48)
        colors = [
            (28, 98, 36), (35, 115, 42), (22, 78, 28),
            (42, 130, 50), (18, 68, 24), (32, 108, 40)
        ]
        color_variant = colors[np.random.randint(0, len(colors))]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color_variant, outline=(15, 55, 20))

    # 3. Winding unpaved forest path / clearing (earth tan)
    path_points = [
        (0, 310), (120, 290), (240, 320), (360, 270),
        (480, 290), (600, 260), (720, 300), (800, 280)
    ]
    for i in range(len(path_points) - 1):
        draw.line([path_points[i], path_points[i+1]], fill=(160, 140, 110), width=36)
        draw.line([path_points[i], path_points[i+1]], fill=(185, 165, 135), width=24)

    # Edge trees overhanging the path
    for px, py in [(180, 265), (320, 310), (440, 245), (580, 285), (690, 275)]:
        r = 20
        draw.ellipse([px - r, py - r, px + r, py + r], fill=(45, 140, 55), outline=(20, 70, 25), width=2)

    return img


def main():
    FRONTEND_DEST.mkdir(parents=True, exist_ok=True)
    BACKEND_DEST.mkdir(parents=True, exist_ok=True)

    samples = [
        ("urban_sample.png", create_urban_sample()),
        ("water_sample.png", create_water_sample()),
        ("forest_sample.png", create_forest_sample()),
    ]

    print("==================================================")
    print("Generating GeoVision Controlled Demo Samples")
    print("==================================================")

    hashes = {}
    for filename, img in samples:
        f_path = FRONTEND_DEST / filename
        b_path = BACKEND_DEST / filename
        img.save(f_path, format="PNG")
        img.save(b_path, format="PNG")

        data = f_path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        hashes[filename] = sha256
        print(f"Generated {filename}: {img.size[0]}x{img.size[1]} px | SHA256: {sha256}")

    print("Finished generating sample images.")


if __name__ == "__main__":
    main()
