from pathlib import Path
from PIL import Image

src = Path("tikz_S1_1.png")

# Try common image extensions if none provided
if src.suffix == "":
    for ext in [".png", ".pdf", ".jpg", ".jpeg", ".tif", ".tiff"]:
        cand = src.with_suffix(ext)
        if cand.exists():
            src = cand
            break

if not src.exists():
    raise FileNotFoundError(f"Could not find input figure at {src!s}")

img = Image.open(src)

# If it's a PDF, PIL will usually load the first page (requires poppler in many envs)
# Convert to RGB so saving to PNG is consistent
img = img.convert("RGB")

w, h = img.size
cuts = [0, w // 3, 2 * w // 3, w]  # equal horizontal spacing

out_dir = src.parent
stem = src.stem

parts = []
for k in range(3):
    box = (cuts[k], 0, cuts[k + 1], h)  # (left, upper, right, lower)
    part = img.crop(box)
    out_path = out_dir / f"{stem}_part{k+1}.png"
    part.save(out_path, dpi=(300, 300))
    parts.append(out_path)

parts