"""
Autoencoder — Resume Image Denoising
------------------------------------
Files used:
    encoder.h5   Noisy resume tile (256x256x1) -> latent space (32x32x256) + skip features
    decoder.h5   Latent space + skip features -> clean resume tile (256x256x1)

High-resolution resumes are processed with overlapping 256x256 tiles that are
blended back together with a Hann window, so the full page keeps its resolution.
"""

import os

import numpy as np
import keras
from PIL import Image

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.h5")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder.h5")

TILE_SIZE = 256
OVERLAP = 32
STEP = TILE_SIZE - OVERLAP
BATCH_SIZE = 8


class ResumeDenoiser:
    def __init__(self):
        self.encoder = keras.models.load_model(ENCODER_PATH, compile=False)
        self.decoder = keras.models.load_model(DECODER_PATH, compile=False)

    @property
    def num_params(self):
        return int(self.encoder.count_params() + self.decoder.count_params())

    def _reconstruct(self, tiles):
        """Encoder -> latent space -> decoder."""
        latent = self.encoder.predict(tiles, batch_size=BATCH_SIZE, verbose=0)
        return self.decoder.predict(latent, batch_size=BATCH_SIZE, verbose=0)

    def denoise(self, image, progress=None):
        """
        Denoise a resume image of any size.
        Returns (original_grayscale_image, denoised_image).
        """
        image = image.convert("L")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        height, width = arr.shape

        pad_h = max(0, TILE_SIZE - height)
        pad_w = max(0, TILE_SIZE - width)
        if height > TILE_SIZE:
            pad_h += (STEP - ((height - TILE_SIZE) % STEP)) % STEP
        if width > TILE_SIZE:
            pad_w += (STEP - ((width - TILE_SIZE) % STEP)) % STEP
        padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")
        ph, pw = padded.shape

        positions = [
            (y, x)
            for y in range(0, ph - TILE_SIZE + 1, STEP)
            for x in range(0, pw - TILE_SIZE + 1, STEP)
        ]

        window = np.maximum(np.outer(np.hanning(TILE_SIZE), np.hanning(TILE_SIZE)), 0.05).astype(np.float32)
        output = np.zeros_like(padded)
        weights = np.zeros_like(padded)

        chunk = BATCH_SIZE * 4
        for start in range(0, len(positions), chunk):
            batch_pos = positions[start:start + chunk]
            tiles = np.stack([padded[y:y + TILE_SIZE, x:x + TILE_SIZE] for y, x in batch_pos])[..., None]
            clean = np.clip(self._reconstruct(tiles)[..., 0], 0.0, 1.0)
            for (y, x), tile in zip(batch_pos, clean):
                output[y:y + TILE_SIZE, x:x + TILE_SIZE] += tile * window
                weights[y:y + TILE_SIZE, x:x + TILE_SIZE] += window
            if progress:
                progress(min(1.0, (start + len(batch_pos)) / len(positions)))

        output = np.clip(output / np.maximum(weights, 1e-8), 0.0, 1.0)[:height, :width]
        return image, Image.fromarray((output * 255).astype(np.uint8), mode="L")


# ---------------- evaluation helpers (MSE / PSNR / SSIM) ----------------

def add_noise(image, amount=0.08, seed=0):
    """Simulate a noisy scan: gaussian noise + salt & pepper + slight stains."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    noisy = arr + rng.normal(0, amount * 1.5, arr.shape)
    mask = rng.random(arr.shape)
    noisy[mask < amount / 4] = 0.0
    noisy[mask > 1 - amount / 4] = 1.0
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(3):
        cy, cx, r = rng.integers(0, h), rng.integers(0, w), rng.integers(min(h, w) // 10 + 1, min(h, w) // 4 + 2)
        stain = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2)))
        noisy -= stain * amount * 2
    return Image.fromarray((np.clip(noisy, 0, 1) * 255).astype(np.uint8), mode="L")


def mse(a, b):
    a = np.asarray(a, dtype=np.float64) / 255.0
    b = np.asarray(b, dtype=np.float64) / 255.0
    return float(np.mean((a - b) ** 2))


def psnr(a, b):
    err = mse(a, b)
    return float("inf") if err == 0 else float(10 * np.log10(1.0 / err))


def ssim(a, b):
    """Global SSIM computed on 8x8 blocks (no extra dependencies)."""
    a = np.asarray(a, dtype=np.float64) / 255.0
    b = np.asarray(b, dtype=np.float64) / 255.0
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    h, w = (a.shape[0] // 8) * 8, (a.shape[1] // 8) * 8
    a = a[:h, :w].reshape(h // 8, 8, w // 8, 8)
    b = b[:h, :w].reshape(h // 8, 8, w // 8, 8)
    mu_a, mu_b = a.mean(axis=(1, 3)), b.mean(axis=(1, 3))
    var_a, var_b = a.var(axis=(1, 3)), b.var(axis=(1, 3))
    cov = ((a - mu_a[:, None, :, None]) * (b - mu_b[:, None, :, None])).mean(axis=(1, 3))
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(s.mean())
