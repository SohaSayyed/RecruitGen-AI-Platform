"""
Autoencoder — Resume Image Denoising
------------------------------------
Files used:
    encoder.h5   Noisy resume tile (256x256x1) -> latent space (32x32x256) + skip features
    decoder.h5   Latent space + skip features -> clean resume tile (256x256x1)
    encoder_v2.h5 / decoder_v2.h5 / v2_config.json
                 same U-Net fine-tuned on real resume pages (finetune.py); used instead of v1 only when the
                 held-out OCR evaluation (models/diffusion/evaluate.py) set "use_by_default" in v2_config.json

Inference pipeline (keeps text sharp on any input size):
    1. Scale matching   the network was trained on A4 pages ~1240 px wide, so small
                        screenshots / low-res scans are upscaled to that scale first
                        (tiny text otherwise falls below the network's receptive field
                        and comes out smeared).
    2. Noise estimate   robust (MAD) noise estimate; noisy scans get a 3x3 median
                        pre-filter at native resolution to kill salt & pepper speckle.
    3. Autoencoder      overlapping 256x256 tiles -> encoder -> latent -> decoder,
                        blended back together with a Hann window.
    4. Detail fusion    the network output decides background vs ink; on clean
                        inputs the ink strokes are taken from the sharp source,
                        on noisy inputs the network output is trusted.
    5. Finishing        unsharp mask + levels (white paper, deep ink).
"""

import os

import numpy as np
import keras
from PIL import Image, ImageFilter

from models import skip_tensorflow_probe

skip_tensorflow_probe()

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.h5")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder.h5")
# v2 = fine-tuned on real resume pages (models/autoencoder/finetune.py); used only when its evaluation enabled it
ENCODER_V2_PATH = os.path.join(MODEL_DIR, "encoder_v2.h5")
DECODER_V2_PATH = os.path.join(MODEL_DIR, "decoder_v2.h5")
CONFIG_V2_PATH = os.path.join(MODEL_DIR, "v2_config.json")


def v2_enabled():
    """True when the fine-tuned v2 files exist and their held-out OCR evaluation marked them as the default."""
    import json

    if not all(os.path.exists(p) for p in (ENCODER_V2_PATH, DECODER_V2_PATH, CONFIG_V2_PATH)):
        return False
    with open(CONFIG_V2_PATH) as f:
        return bool(json.load(f).get("use_by_default"))

TILE_SIZE = 256
OVERLAP = 32
STEP = TILE_SIZE - OVERLAP
BATCH_SIZE = 8

TARGET_WIDTH = 1240      # page width the network was trained on (A4 @ 150 dpi)
UPSCALE_BELOW = 1100     # only upscale pages narrower than this
MAX_UPSCALE = 3.0
NOISY_SIGMA = 0.015      # above this the input is treated as a noisy scan


def _to_float(img):
    return np.asarray(img, dtype=np.float32) / 255.0


def _to_image(arr):
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8), mode="L")


def estimate_noise(image):
    """Robust noise sigma (0-1 scale) from the median absolute deviation of a high-pass residual."""
    a = _to_float(image.convert("L"))
    if min(a.shape) < 3:
        return 0.0
    hp = a[1:, 1:] - 0.5 * (a[:-1, 1:] + a[1:, :-1])
    return float(np.median(np.abs(hp - np.median(hp))) * 1.4826 / 1.2247)


class ResumeDenoiser:
    def __init__(self, version="auto"):
        """version: "auto" (v2 when enabled, else v1), "v1" (original encoder.h5 / decoder.h5) or "v2"."""
        self.version = "v2" if version == "v2" or (version == "auto" and v2_enabled()) else "v1"
        enc, dec = (ENCODER_V2_PATH, DECODER_V2_PATH) if self.version == "v2" else (ENCODER_PATH, DECODER_PATH)
        self.encoder = keras.models.load_model(enc, compile=False)
        self.decoder = keras.models.load_model(dec, compile=False)

    @property
    def num_params(self):
        return int(self.encoder.count_params() + self.decoder.count_params())

    def _reconstruct(self, tiles):
        """Encoder -> latent space -> decoder."""
        latent = self.encoder.predict(tiles, batch_size=BATCH_SIZE, verbose=0)
        return self.decoder.predict(latent, batch_size=BATCH_SIZE, verbose=0)

    def _denoise_tiles(self, arr, progress=None):
        """Run the autoencoder over a float image of any size using overlapping blended tiles."""
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

        return np.clip(output / np.maximum(weights, 1e-8), 0.0, 1.0)[:height, :width]

    def denoise(self, image, progress=None):
        """
        Denoise a resume image of any size.
        Returns (original_grayscale_image, denoised_image). Small inputs are returned
        super-resolved at the network's working scale, so the text stays crisp.
        """
        image = image.convert("L")
        width, height = image.size
        if self.version == "v2":            # trained to output the clean page directly: no hand-tuned post-processing
            scale = min(MAX_UPSCALE, TARGET_WIDTH / width) if width < UPSCALE_BELOW else 1.0
            source = image.resize((round(width * scale), round(height * scale)), Image.LANCZOS) if scale > 1 else image
            return image, _to_image(self._denoise_tiles(_to_float(source), progress))
        sigma = estimate_noise(image)
        noisy = sigma > NOISY_SIGMA

        source = image.filter(ImageFilter.MedianFilter(3)) if noisy else image
        scale = min(MAX_UPSCALE, TARGET_WIDTH / width) if width < UPSCALE_BELOW else 1.0
        if scale > 1.0:
            source = source.resize((round(width * scale), round(height * scale)), Image.LANCZOS)

        den_arr = self._denoise_tiles(_to_float(source), progress)
        den = _to_image(den_arr)

        # Detail fusion: background tone comes from the network, ink ratio from the sharp source.
        ref = _to_float(source.filter(ImageFilter.MedianFilter(3)))
        bg_den = _to_float(den.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(5)))
        bg_ref = _to_float(_to_image(ref).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(5)))
        sharp = bg_den * np.clip(ref / np.maximum(bg_ref, 0.2), 0.0, 1.0)

        ink = (_to_float(den.filter(ImageFilter.MinFilter(3))) < 0.85).astype(np.float32)
        ink = _to_float(_to_image(ink).filter(ImageFilter.GaussianBlur(1)))
        trust = float(np.clip(1.0 - (sigma - 0.02) / 0.04, 0.0, 1.0))
        out = den_arr + trust * ink * (sharp - den_arr)

        result = _to_image(out)
        if noisy:
            result = result.filter(ImageFilter.MedianFilter(3))
        result = result.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
        result = _to_image((_to_float(result) - 0.04) / 0.90)
        return image, result


# ---------------- content safeguard ----------------

def ink_energy(image):
    """Total darkness of text/graphics relative to the local paper background (half resolution, 0-1 scale)."""
    g = image.convert("L")
    g = g.resize((max(1, g.width // 2), max(1, g.height // 2)), Image.BILINEAR)
    a = _to_float(g)
    background = _to_float(g.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(10)))
    return float(np.clip(background - a, 0.0, 1.0).sum())


def content_ratio(original, cleaned):
    """How much of the original's ink survives cleaning (1 = same amount, < 0.25 = most content erased)."""
    if cleaned.size != original.size:
        cleaned = cleaned.resize(original.size, Image.LANCZOS)
    return ink_energy(cleaned) / max(ink_energy(original), 1e-6)


# Below this the cleaned image is rejected and the original kept. Calibrated: a correct cleaning keeps 0.70-1.20 of the
# ink (noise / shadow / contrast / blur pages); a model erasing an unrecoverably blurred page scored 0.09.
CONTENT_MIN_RATIO = 0.30


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


def _match(a, b):
    """Resize b to a's size (denoised outputs may be super-resolved)."""
    if isinstance(a, Image.Image) and isinstance(b, Image.Image) and a.size != b.size:
        b = b.resize(a.size, Image.LANCZOS)
    return a, b


def mse(a, b):
    a, b = _match(a, b)
    a = np.asarray(a, dtype=np.float64) / 255.0
    b = np.asarray(b, dtype=np.float64) / 255.0
    return float(np.mean((a - b) ** 2))


def psnr(a, b):
    err = mse(a, b)
    return float("inf") if err == 0 else float(10 * np.log10(1.0 / err))


def ssim(a, b):
    """Global SSIM computed on 8x8 blocks (no extra dependencies)."""
    a, b = _match(a, b)
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
