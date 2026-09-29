"""Analyse du morceau, image par image (30 i/s) : énergie, bandes, attaques, temps.

Entrée  : build/song.wav (non versionné : c'est le morceau de l'artiste)
Sortie  : build/features.npz
"""
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import find_peaks

BUILD = Path(__file__).parent / "build"
FPS = 30


def main():
    y, sr = librosa.load(BUILD / "song.wav", sr=22050, mono=True)
    dur = len(y) / sr
    n = int(np.ceil(dur * FPS))
    hop = 512
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    t_stft = librosa.times_like(S, sr=sr, hop_length=hop)
    tf = np.arange(n) / FPS

    def band(lo, hi):
        e = S[(f >= lo) & (f < hi)].mean(0)
        e = np.interp(tf, t_stft, e)
        return e / (np.percentile(e, 98) + 1e-9)

    rms = np.interp(tf, t_stft, librosa.feature.rms(S=S)[0])
    rms = rms / np.percentile(rms, 98)
    sub, bass, mid, high = band(20, 90), band(90, 300), band(300, 3000), band(3000, 11000)

    def onset(lo, hi):
        env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[(f >= lo) & (f < hi)]), sr=sr, hop_length=hop)
        return env / (np.percentile(env, 99) + 1e-9)

    low_on, mid_on = onset(30, 150), onset(1500, 6000)
    kicks = t_stft[find_peaks(low_on, height=0.55, distance=int(0.18 * sr / hop))[0]]
    snares = t_stft[find_peaks(mid_on, height=0.6, distance=int(0.18 * sr / hop))[0]]
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", hop_length=hop)

    np.savez(BUILD / "features.npz", duration=dur, rms=rms, sub=sub, bass=bass, mid=mid, high=high,
             kicks=kicks, snares=snares, beats=beats,
             low_on=np.interp(tf, t_stft, low_on), mid_on=np.interp(tf, t_stft, mid_on))
    print(f"{dur:.2f} s, {n} images, {len(beats)} temps, {len(kicks)} kicks, {len(snares)} caisses claires")


if __name__ == "__main__":
    main()
