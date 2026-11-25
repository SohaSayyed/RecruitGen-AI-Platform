"""
Diffusion Model — Resume Image Enhancement (conditional DDPM, x0-prediction, DDIM sampling)
-------------------------------------------------------------------------------------------
Files used:
    diffusion_unet.safetensors   EMA weights of the conditional U-Net
    config.json                  architecture, noise schedule, sampling defaults, training info
    metrics.json                 held-out evaluation written by models/diffusion/evaluate.py

Training   x0 = clean page, c = degraded page, x_t = sqrt(ᾱ_t)·x0 + sqrt(1-ᾱ_t)·ε
           the U-Net predicts x0 from (x_t, c, t); loss = L1 weighted ×5 on ink pixels (text first)
Inference  deterministic DDIM from x_T ~ N(0, I) (one seeded noise image per page, so overlapping tiles agree),
           a few steps per tile; tiles without ink use a single step. Runs fully locally (CPU or GPU).
"""

import json
import os
import time

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from models.diffusion.unet import ConditionalUNet

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(MODEL_DIR, "diffusion_unet.safetensors")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

TARGET_WIDTH = 1240      # training scale (A4 @ 150 dpi), same as the Autoencoder
UPSCALE_BELOW = 1100
MAX_UPSCALE = 3.0
MAX_PIXELS = 12_000_000  # input validation: refuse absurdly large images


def get_device(preferred=None):
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_alphas_cumprod(T, s=0.008):
    steps = np.arange(T + 1, dtype=np.float64) / T
    f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    return np.clip(f[1:] / f[0], 1e-5, 0.9999).astype(np.float32)


class DiffusionSchedule:
    def __init__(self, T=1000):
        self.T = T
        self.alphas_cumprod = torch.tensor(cosine_alphas_cumprod(T))

    def q_sample(self, x0, t, noise):
        ab = self.alphas_cumprod.to(x0.device)[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def loss(self, model, clean, cond, ink_weight=5.0):
        t = torch.randint(0, self.T, (clean.shape[0],), device=clean.device)
        noise = torch.randn_like(clean)
        pred = model(self.q_sample(clean, t, noise), cond, t)
        weight = 1.0 + (ink_weight - 1.0) * (clean < 0.0).float()         # ink = darker than mid-grey
        return (weight * (pred - clean).abs()).mean()

    def timesteps(self, steps):
        return np.linspace(self.T - 1, 0, steps).round().astype(int).tolist()

    @torch.no_grad()
    def ddim(self, model, cond, noise, steps):
        """Deterministic DDIM (eta = 0) with x0-prediction. Returns the predicted clean image in [-1, 1]."""
        ab = self.alphas_cumprod.to(cond.device)
        x = noise
        ts = self.timesteps(steps)
        x0 = None
        for i, t in enumerate(ts):
            tt = torch.full((cond.shape[0],), t, device=cond.device, dtype=torch.long)
            x0 = model(x, cond, tt).clamp(-1, 1)
            if i == len(ts) - 1:
                break
            eps = (x - ab[t].sqrt() * x0) / (1 - ab[t]).sqrt()
            t_prev = ts[i + 1]
            x = ab[t_prev].sqrt() * x0 + (1 - ab[t_prev]).sqrt() * eps
        return x0


def save_model(model, config, out_dir=MODEL_DIR):
    os.makedirs(out_dir, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
              os.path.join(out_dir, "diffusion_unet.safetensors"), metadata={"format": "pt", "model": "ResumeDiffusion"})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def checkpoint_available(model_dir=MODEL_DIR):
    return all(os.path.exists(os.path.join(model_dir, n)) for n in ("diffusion_unet.safetensors", "config.json"))


def load_metrics(model_dir=MODEL_DIR):
    path = os.path.join(model_dir, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


class ResumeDiffusionEnhancer:
    """Loads the trained diffusion U-Net and enhances resume images of any size (tiled, locally)."""

    def __init__(self, model_dir=MODEL_DIR, device=None, model=None, config=None):
        if config is None:
            if not checkpoint_available(model_dir):
                raise FileNotFoundError("Diffusion checkpoint not found in models/diffusion/ — train it with "
                                        "Diffusion_Training.ipynb or `python -m models.diffusion.train`.")
            with open(os.path.join(model_dir, "config.json")) as f:
                config = json.load(f)
        self.config = config
        self.device = get_device(device)
        arch = config["architecture"]
        if model is None:
            model = ConditionalUNet(base=arch["base"], mults=tuple(arch["mults"]))
            model.load_state_dict(load_file(os.path.join(model_dir, "diffusion_unet.safetensors")), strict=True)
        self.model = model.to(self.device).eval()
        self.schedule = DiffusionSchedule(config["schedule"]["T"])
        s = config["sampling"]
        self.steps, self.tile, self.overlap, self.batch = s["steps"], s["tile"], s["overlap"], s["batch"]
        if self.device.type == "cpu":
            torch.set_num_threads(max(1, os.cpu_count() or 1))

    @property
    def num_params(self):
        return sum(p.numel() for p in self.model.parameters())

    @staticmethod
    def validate(image):
        if not isinstance(image, Image.Image):
            raise TypeError("enhance() expects a PIL image")
        w, h = image.size
        if w < 32 or h < 32:
            raise ValueError("Image is too small to enhance (minimum 32 x 32 px).")
        if w * h > MAX_PIXELS:
            raise ValueError(f"Image is too large ({w}x{h}); maximum is {MAX_PIXELS / 1e6:.0f} megapixels.")

    def _tiles(self, h, w):
        step = self.tile - self.overlap
        ys = list(range(0, max(1, h - self.tile) + 1, step))
        xs = list(range(0, max(1, w - self.tile) + 1, step))
        if ys[-1] + self.tile < h:
            ys.append(h - self.tile)
        if xs[-1] + self.tile < w:
            xs.append(w - self.tile)
        return [(y, x) for y in ys for x in xs]

    def enhance(self, image, steps=None, seed=0, progress=None):
        """Returns (original grayscale image, enhanced image, info)."""
        self.validate(image)
        start = time.time()
        image = image.convert("L")
        width, height = image.size
        source = image
        scale = min(MAX_UPSCALE, TARGET_WIDTH / width) if width < UPSCALE_BELOW else 1.0
        if scale > 1.0:
            source = image.resize((round(width * scale), round(height * scale)), Image.LANCZOS)
        arr = np.asarray(source, np.float32) / 255.0
        h0, w0 = arr.shape
        T = self.tile
        ph, pw = max(h0, T), max(w0, T)
        arr = np.pad(arr, ((0, ph - h0), (0, pw - w0)), mode="reflect" if min(h0, w0) > 1 else "edge")
        cond = torch.from_numpy(arr * 2 - 1)
        noise = torch.randn((ph, pw), generator=torch.Generator().manual_seed(int(seed)))
        positions = self._tiles(ph, pw)
        window = torch.from_numpy(np.maximum(np.outer(np.hanning(T), np.hanning(T)), 0.05).astype(np.float32))
        out, weights = torch.zeros((ph, pw)), torch.zeros((ph, pw))
        steps = int(steps or self.steps)

        # tiles with visible ink get the full step count, blank (background) tiles a single step
        inky = [bool(((arr[y:y + T, x:x + T] < 0.55).mean()) > 0.002) for y, x in positions]
        groups = [(steps, [p for p, k in zip(positions, inky) if k]), (1, [p for p, k in zip(positions, inky) if not k])]
        done = 0
        with torch.inference_mode():
            for n_steps, group in groups:
                for i in range(0, len(group), self.batch):
                    batch = group[i:i + self.batch]
                    c = torch.stack([cond[y:y + T, x:x + T] for y, x in batch])[:, None].to(self.device)
                    z = torch.stack([noise[y:y + T, x:x + T] for y, x in batch])[:, None].to(self.device)
                    pred = self.schedule.ddim(self.model, c, z, n_steps)[:, 0].float().cpu()
                    for (y, x), tile in zip(batch, pred):
                        out[y:y + T, x:x + T] += tile * window
                        weights[y:y + T, x:x + T] += window
                    done += len(batch)
                    if progress:
                        progress(done / len(positions))
        result = ((out / weights.clamp_min(1e-8)).numpy()[:h0, :w0] + 1) / 2
        enhanced = Image.fromarray((np.clip(result, 0, 1) * 255).round().astype(np.uint8), mode="L")
        info = {"seconds": time.time() - start, "steps": steps, "tiles": len(positions), "text_tiles": sum(inky),
                "device": self.device.type, "scale": scale}
        return image, enhanced, info
