"""Agrandissement ×4 par IA (Real-ESRGAN, ONNX) d'une image source, par tuiles.

    python3 upscale.py realesrgan-x4.onnx assets/city_1k.png assets/city.png 2688
Le résultat ×4 est ramené à la largeur demandée (plus net qu'un simple agrandissement).
"""
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

TILE, PAD = 64, 8


def main(model, src, dst, width):
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    im = np.asarray(Image.open(src).convert("RGB"), np.float32) / 255
    h, w, _ = im.shape
    step = TILE - 2 * PAD
    ph, pw = (-h) % step + 2 * PAD, (-w) % step + 2 * PAD
    padded = np.pad(im, ((PAD, ph - PAD), (PAD, pw - PAD), (0, 0)), mode="reflect")
    out = np.zeros((h * 4, w * 4, 3), np.float32)
    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = padded[y:y + TILE, x:x + TILE].transpose(2, 0, 1)[None]
            res = sess.run(None, {name: np.ascontiguousarray(tile)})[0][0].transpose(1, 2, 0)
            core = res[PAD * 4:(PAD + step) * 4, PAD * 4:(PAD + step) * 4]
            hh, ww = min(step * 4, out.shape[0] - y * 4), min(step * 4, out.shape[1] - x * 4)
            out[y * 4:y * 4 + hh, x * 4:x * 4 + ww] = core[:hh, :ww]
        print(f"{y + step}/{h}", end="\r", flush=True)
    big = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
    big.resize((width, round(width * h / w)), Image.LANCZOS).save(dst)
    print("\nok", big.size, "->", width)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
