"""Cartes de profondeur des images sources (Depth Anything V2, ONNX, sur CPU).

Elles servent aux mouvements de caméra en 2,5D : le premier plan se déplace
plus que le fond, ce qui donne du relief à une image fixe.

    python3 depth.py chemin/vers/depth_anything_v2.onnx
Sortie : assets/<nom>_depth.png (16 bits, blanc = proche)
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).parent


def main(model):
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    for name in ("city", "android", "scream"):
        im = Image.open(HERE / "assets" / f"{name}.png").convert("RGB")
        w, h = 1022, 574                                        # multiples de 14, format 16:9
        x = np.asarray(im.resize((w, h), Image.BICUBIC), np.float32) / 255
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        d = sess.run(None, {"pixel_values": x})[0][0]
        d = (d - d.min()) / (d.max() - d.min())
        d = np.asarray(Image.fromarray(d.astype(np.float32), "F").resize(im.size, Image.BICUBIC))
        d = np.clip(gaussian_filter(d, 2), 0, 1)
        Image.fromarray((d * 65535).astype(np.uint16)).save(HERE / "assets" / f"{name}_depth.png")
        print(name, im.size, "ok")


if __name__ == "__main__":
    main(sys.argv[1])
