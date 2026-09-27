import sys
from PIL import Image
import render
for t in map(float, sys.argv[1:]):
    Image.fromarray(render.frame(t, int(t * 30))).save(f"stills/t{t:05.2f}.png")
