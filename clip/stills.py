"""Planche contact : python3 stills.py t1 t2 ... -> stills/sheet.png"""
import sys
from PIL import Image
import render
ts = [float(a) for a in sys.argv[1:]]
ims = []
for t in ts:
    im = Image.fromarray(render.frame(int(round(t * render.FPS))))
    im.save(f"stills/t{t:06.2f}.png")
    ims.append(im.resize((480, 270)))
cols = 4
sheet = Image.new("RGB", (480 * cols, 270 * ((len(ims) + cols - 1) // cols)))
for i, im in enumerate(ims):
    sheet.paste(im, ((i % cols) * 480, (i // cols) * 270))
sheet.save(sys.stdin.isatty() and "stills/sheet.png" or "stills/sheet.png")
