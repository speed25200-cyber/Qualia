"""Planche contact : python3 stills.py t1 t2 ... -> stills/sheet.png (et une image par instant)."""
import sys
from PIL import Image
import render
ts = [float(a) for a in sys.argv[1:]]
ims = []
for t in ts:
    im = Image.fromarray(render.frame(t))
    im.save(f"stills/t{t:05.2f}.png")
    ims.append(im.resize((480, 270)))
cols = 4
sheet = Image.new("RGB", (480 * cols, 270 * ((len(ims) + cols - 1) // cols)))
for i, im in enumerate(ims):
    sheet.paste(im, ((i % cols) * 480, (i // cols) * 270))
sheet.save("stills/sheet.png")
