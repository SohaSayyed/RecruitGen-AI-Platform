"""
Controlled resume-image degradations (clean page -> paired degraded page)
------------------------------------------------------------------------
noise · blur · shadows · uneven lighting · low contrast · JPEG artifacts · salt & pepper

Low-frequency effects (shadows, lighting) are defined in PAGE coordinates, so a training crop and the full page see the
same field: degrade(crop, origin=(y, x), page_shape=(H, W)) equals the matching crop of degrade(page).
Named PRESETS are used for the held-out evaluation so every method is scored on identical inputs.
"""

import io

import numpy as np
from PIL import Image, ImageFilter

PRESETS = {
    "noise": {"noise": (0.06, 0.09), "salt_pepper": (0.002, 0.006)},
    "blur": {"blur": (0.9, 1.3), "noise": (0.01, 0.02)},
    "shadow_lighting": {"shadow": (0.25, 0.45), "lighting": (0.55, 0.75), "noise": (0.01, 0.02)},
    "low_contrast_jpeg": {"contrast": (0.30, 0.45), "jpeg": (18, 30)},
    "mixed_scan": {"noise": (0.03, 0.05), "blur": (0.6, 0.9), "shadow": (0.15, 0.3), "lighting": (0.7, 0.85),
                   "contrast": (0.15, 0.25), "jpeg": (30, 45)},
}
# Training: every effect is applied independently with this probability, at a random strength from these ranges
TRAIN_RANGES = {"noise": (0.01, 0.10), "blur": (0.4, 1.4), "shadow": (0.1, 0.5), "lighting": (0.5, 0.95),
                "contrast": (0.1, 0.5), "jpeg": (15, 60), "salt_pepper": (0.001, 0.01)}
TRAIN_PROB = {"noise": 0.7, "blur": 0.45, "shadow": 0.4, "lighting": 0.45, "contrast": 0.4, "jpeg": 0.4,
              "salt_pepper": 0.25}


def _field_params(rng, spec, page_shape):
    """Random page-level parameters for the low-frequency effects (drawn once per page)."""
    h, w = page_shape
    p = {}
    if "shadow" in spec:
        p["shadow"] = [(rng.uniform(0, h), rng.uniform(0, w), rng.uniform(0.12, 0.4) * max(h, w),
                        rng.uniform(*spec["shadow"])) for _ in range(rng.integers(1, 4))]
        p["edge"] = (rng.integers(0, 4), rng.uniform(0.0, spec["shadow"][1]), rng.uniform(0.05, 0.2) * w)
    if "lighting" in spec:
        angle = rng.uniform(0, 2 * np.pi)
        p["lighting"] = (rng.uniform(*spec["lighting"]), np.cos(angle), np.sin(angle))
    return p


def _fields(shape, origin, page_shape, p):
    """Multiplicative illumination (lighting x shadows) evaluated on the crop grid."""
    h, w = shape
    H, W = page_shape
    yy, xx = np.mgrid[origin[0]:origin[0] + h, origin[1]:origin[1] + w].astype(np.float32)
    mult = np.ones(shape, np.float32)
    if "lighting" in p:
        low, cy, cx = p["lighting"]
        ramp = ((yy / H - 0.5) * cy + (xx / W - 0.5) * cx + 0.71) / 1.42      # 0..1 across the page
        mult *= 1.0 - (1.0 - low) * np.clip(ramp, 0, 1)
    for cy, cx, r, a in p.get("shadow", []):
        mult *= 1.0 - a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
    if "edge" in p:
        side, a, width = p["edge"]
        dist = [yy, H - yy, xx, W - xx][side]
        mult *= 1.0 - a * np.exp(-np.maximum(dist, 0) / max(width, 1.0))
    return mult


def degrade(clean, spec, rng, origin=(0, 0), page_shape=None, page_params=None):
    """clean: float32 array in [0, 1] (white paper = 1). spec: {effect: (lo, hi)}. Returns float32 [0, 1]."""
    x = clean.astype(np.float32).copy()
    page_shape = page_shape or x.shape
    params = page_params if page_params is not None else _field_params(rng, spec, page_shape)
    if "contrast" in spec:                          # grey ink on grey paper
        c = rng.uniform(*spec["contrast"])
        x = c * 0.6 + x * (1.0 - c)
    if "blur" in spec:
        x = np.asarray(Image.fromarray((x * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(rng.uniform(*spec["blur"]))), np.float32) / 255.0
    if params:
        x = x * _fields(x.shape, origin, page_shape, params)
    if "noise" in spec:
        x = x + rng.normal(0, rng.uniform(*spec["noise"]), x.shape).astype(np.float32)
    if "salt_pepper" in spec:
        m = rng.random(x.shape)
        amount = rng.uniform(*spec["salt_pepper"])
        x[m < amount / 2] = 0.0
        x[m > 1 - amount / 2] = 1.0
    x = np.clip(x, 0, 1)
    if "jpeg" in spec:
        buf = io.BytesIO()
        Image.fromarray((x * 255).round().astype(np.uint8)).save(buf, format="JPEG", quality=int(rng.uniform(*spec["jpeg"])))
        x = np.asarray(Image.open(buf), np.float32) / 255.0
    return x.astype(np.float32)


def random_spec(rng):
    """Random training degradation: each effect independently, at least one."""
    spec = {k: TRAIN_RANGES[k] for k, p in TRAIN_PROB.items() if rng.random() < p}
    return spec or {"noise": TRAIN_RANGES["noise"]}


def degrade_page(clean_image, preset, seed):
    """Full-page degradation with a named preset (deterministic for a given seed). PIL in, PIL out."""
    arr = np.asarray(clean_image.convert("L"), np.float32) / 255.0
    out = degrade(arr, PRESETS[preset], np.random.default_rng(seed))
    return Image.fromarray((out * 255).round().astype(np.uint8), mode="L")
