"""Turn the turntable's RGBA frames into the README GIF, with PIL - there is no ffmpeg on a Spark.

    python gif.py frames_dir out.gif [--size 480] [--frames 60] [--bg 18,20,24]

The published GIFs (spark/results/dresser_demo.gif, dresser_flash_next.gif) are 480x480, 60 frames,
loop 0, 80 ms x40 + 90 ms x20 with the 90 every third frame, on a flat (18, 20, 24). That is what
this writes: 150 rendered frames sampled evenly down to 60, composited over the background - the
render leaves it transparent - and quantised through one palette shared by every frame, so the
background cannot shimmer from frame to frame.
"""

import sys
from pathlib import Path

from PIL import Image

args = sys.argv[1:]


def option(name, fallback):
    return args[args.index(name) + 1] if name in args else fallback


frames_dir, out = Path(args[0]), Path(args[1])
size, want = int(option("--size", "480")), int(option("--frames", "60"))
background = tuple(int(v) for v in option("--bg", "18,20,24").split(","))

src = sorted(frames_dir.glob("frame_*.png"))
if len(src) < want:
    raise SystemExit(f"gif: {len(src)} frames in {frames_dir}, need at least {want}")
picked = [src[round(i * len(src) / want)] for i in range(want)]

flat = []
for path in picked:
    frame = Image.open(path).convert("RGBA")
    plate = Image.new("RGBA", frame.size, (*background, 255))
    plate.alpha_composite(frame)
    flat.append(plate.convert("RGB").resize((size, size), Image.LANCZOS))

# One palette for the whole GIF, taken from every third frame stacked into a single image: a palette
# built from one frame alone loses colours the swing brings round later.
sample = Image.new("RGB", (size, size * len(flat[::3])))
for i, frame in enumerate(flat[::3]):
    sample.paste(frame, (0, i * size))
palette = sample.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
# Pin the background into the palette, and collapse everything next to it onto the same value.
# Pinning one entry is not enough: the median cut puts several entries within a unit or two of the
# background, the nearest-colour search picks whichever it likes, and the flat area comes out
# (17, 20, 24) mottled - which is what the older of the two published GIFs does.
entries = palette.getpalette()
for i in range(len(entries) // 3):
    if sum((entries[3 * i + c] - background[c]) ** 2 for c in range(3)) <= 12:
        entries[3 * i:3 * i + 3] = list(background)
palette.putpalette(entries)
palette.load()        # putpalette alone leaves the quantiser reading the old palette

quantised = [f.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for f in flat]
durations = [90 if i % 3 == 1 else 80 for i in range(want)]

out.parent.mkdir(parents=True, exist_ok=True)
quantised[0].save(out, save_all=True, append_images=quantised[1:], duration=durations, loop=0,
                  optimize=False, disposal=1)
check = Image.open(out).convert("RGB")
flat = {c: n for n, c in check.getcolors(size * size)}.get(background, 0) / (size * size)
print(f"GIF {out} {out.stat().st_size} bytes {size}x{size} {want} frames "
      f"{durations.count(80)}x80ms + {durations.count(90)}x90ms loop 0 "
      f"background {background} on {flat * 100:.1f}% of frame 1")
