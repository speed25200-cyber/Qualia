"""Horodatage mot à mot de la voix off (faster-whisper), pour synchroniser l'image.

Produit build/words.json : pour chaque réplique, les mots affichés avec leur
instant de début/fin. Les mots reconnus sont alignés sur le texte affiché
(difflib) ; les mots non reconnus sont interpolés entre leurs voisins.
"""
import difflib
import json
import re
import unicodedata
from pathlib import Path

from faster_whisper import WhisperModel

BUILD = Path(__file__).parent / "build"


def norm(w):
    w = unicodedata.normalize("NFKD", w.lower())
    return re.sub(r"[^a-z]", "", w)


def caption_words(text):
    """Découpe l'affichage en mots ; la ponctuation isolée (« : », « ? ») reste collée au mot précédent."""
    out = []
    for tok in text.split():
        if out and not norm(tok):
            out[-1] += " " + tok
        else:
            out.append(tok)
    return out


def heard_words(segments):
    out = []
    for s in segments:
        for w in s.words:
            txt = w.word.strip()
            if out and (txt[:1] in "'-" or not norm(txt)):
                out[-1] = (out[-1][0] + txt, out[-1][1], w.end)
            else:
                out.append((txt, w.start, w.end))
    return out


def main():
    tl = json.loads((BUILD / "timeline.json").read_text())
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(str(BUILD / "voice.wav"), language="fr", word_timestamps=True)
    heard = heard_words(segs)
    out = []
    for line in tl["lines"]:
        words = caption_words(line["caption"])
        inside = [h for h in heard if line["start"] - 0.3 <= (h[1] + h[2]) / 2 <= line["end"] + 0.3]
        sm = difflib.SequenceMatcher(None, [norm(w) for w in words], [norm(h[0]) for h in inside], autojunk=False)
        times = [None] * len(words)
        for a, b, n in sm.get_matching_blocks():
            for k in range(n):
                times[a + k] = inside[b + k][1:]
        # interpolation des trous
        for i, t in enumerate(times):
            if t is None:
                prev_end = times[i - 1][1] if i and times[i - 1] else line["start"]
                j = next((j for j in range(i + 1, len(words)) if times[j]), None)
                next_start = times[j][0] if j is not None else line["end"]
                gap = j - i if j is not None else len(words) - i
                d = (next_start - prev_end) / (gap + 0)
                times[i] = (prev_end, prev_end + d)
        out.append({**line, "words": [{"w": w, "start": round(a, 3), "end": round(b, 3)} for w, (a, b) in zip(words, times)]})
    (BUILD / "words.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for l in out:
        print(l["id"], " ".join(f'{w["w"]}@{w["start"]:.2f}' for w in l["words"]))


if __name__ == "__main__":
    main()
