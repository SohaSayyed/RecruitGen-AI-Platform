"""
Autoencoder v2 — fine-tune the shipped U-Net autoencoder on real resume pages
-----------------------------------------------------------------------------
    python -m models.autoencoder.finetune               # CPU default: 600 steps (~25-35 min)
    python -m models.autoencoder.finetune --steps 4000 --batch 16   # longer run (GPU recommended)

Why: the original encoder.h5 / decoder.h5 were trained on synthetic rendered pages; on real resume scans their output
often reads WORSE in OCR than the input (see models/diffusion/metrics.json). v2 starts from those weights and is
fine-tuned on the same real data as the diffusion model — first pages of the Kaggle resume PDFs with controlled
degradations (noise, blur, shadows, lighting, contrast, JPEG) — with an ink-weighted L1 loss (text first).

Writes encoder_v2.h5 + decoder_v2.h5 (+ v2_config.json). The original encoder.h5 / decoder.h5 are never modified.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
from keras import layers, ops

from models import skip_tensorflow_probe
from models.autoencoder.autoencoder import DECODER_PATH, ENCODER_PATH, MODEL_DIR
from models.diffusion import data as dd

skip_tensorflow_probe()

ENCODER_V2 = os.path.join(MODEL_DIR, "encoder_v2.h5")
DECODER_V2 = os.path.join(MODEL_DIR, "decoder_v2.h5")
CONFIG_V2 = os.path.join(MODEL_DIR, "v2_config.json")


def build_unet():
    """Same architecture and layer names as Autoencoder_Training.ipynb, but with flexible input size."""
    def conv(x, filters, name):
        return layers.Conv2D(filters, 3, padding="same", activation="relu", name=name)(x)

    inp = keras.Input((None, None, 1), name="dirty_resume")
    e1 = conv(conv(inp, 32, "enc1_a"), 32, "enc1_b")
    e2 = conv(conv(layers.MaxPooling2D(name="pool1")(e1), 64, "enc2_a"), 64, "enc2_b")
    e3 = conv(conv(layers.MaxPooling2D(name="pool2")(e2), 128, "enc3_a"), 128, "enc3_b")
    latent = conv(layers.MaxPooling2D(name="pool3")(e3), 256, "latent_representation")
    d = conv(latent, 256, "dec_bottleneck")
    d = layers.Concatenate(name="skip1")([layers.UpSampling2D(name="up1")(d), e3])
    d = conv(conv(d, 128, "dec1_a"), 128, "dec1_b")
    d = layers.Concatenate(name="skip2")([layers.UpSampling2D(name="up2")(d), e2])
    d = conv(conv(d, 64, "dec2_a"), 64, "dec2_b")
    d = layers.Concatenate(name="skip3")([layers.UpSampling2D(name="up3")(d), e1])
    d = conv(conv(d, 32, "dec3_a"), 32, "dec3_b")
    out = layers.Conv2D(1, 3, padding="same", activation="sigmoid", name="clean_resume")(d)
    full = keras.Model(inp, out, name="Resume_Denoising_Autoencoder_v2")
    return full, (inp, latent, e3, e2, e1)


def split(full, tensors):
    """Encoder (latent + skip maps) and decoder, exactly like the shipped encoder.h5 / decoder.h5."""
    inp, latent, e3, e2, e1 = tensors
    g = full.get_layer
    encoder = keras.Model(inp, [latent, e3, e2, e1], name="resume_encoder")
    lat_in = keras.Input((None, None, 256), name="latent")
    s3, s2, s1 = (keras.Input((None, None, c), name=n) for c, n in ((128, "skip_64"), (64, "skip_128"), (32, "skip_256")))
    x = g("dec_bottleneck")(lat_in)
    x = g("skip1")([g("up1")(x), s3]); x = g("dec1_b")(g("dec1_a")(x))
    x = g("skip2")([g("up2")(x), s2]); x = g("dec2_b")(g("dec2_a")(x))
    x = g("skip3")([g("up3")(x), s1]); x = g("dec3_b")(g("dec3_a")(x))
    decoder = keras.Model([lat_in, s3, s2, s1], g("clean_resume")(x), name="resume_decoder")
    return encoder, decoder


def load_v1_weights(full):
    """The shipped files use auto-generated layer names (conv2d_25 …) but the identical architecture, so the 15
    convolution layers are copied in order, with a strict shape check."""
    enc = keras.models.load_model(ENCODER_PATH, compile=False)
    dec = keras.models.load_model(DECODER_PATH, compile=False)
    source = [l for m in (enc, dec) for l in m.layers if l.weights]
    target = [l for l in full.layers if l.weights]
    if len(source) != len(target):
        raise ValueError(f"Layer count mismatch: shipped {len(source)} vs v2 {len(target)}")
    for src, dst in zip(source, target):
        w = src.get_weights()
        if [x.shape for x in w] != [tuple(v.shape) for v in dst.weights]:
            raise ValueError(f"Shape mismatch {src.name} -> {dst.name}")
        dst.set_weights(w)
    return len(target)


def ink_l1(y_true, y_pred):
    weight = 1.0 + 4.0 * ops.cast(y_true < 0.5, "float32")
    return ops.mean(weight * ops.abs(y_true - y_pred))


def finetune(steps=600, batch=8, patch=128, lr=2e-4, eval_every=100, seed=42, log=print):
    keras.utils.set_random_seed(seed)
    splits = dd.prepare_pages(seed=seed, log=None)
    sampler = dd.PatchSampler([dd.load_page(p) for p in splits["train"]], patch=patch, seed=seed)
    val_deg, val_clean = dd.PatchSampler([dd.load_page(p) for p in splits["val"]], patch=128, seed=seed + 1).sample(48)
    val_deg, val_clean = (val_deg[:, 0, :, :, None] + 1) / 2, (val_clean[:, 0, :, :, None] + 1) / 2

    full, tensors = build_unet()
    copied = load_v1_weights(full)
    full.compile(optimizer=keras.optimizers.Adam(lr), loss=ink_l1)
    v1_val = float(ink_l1(val_clean, full.predict(val_deg, verbose=0)))
    log(f"Fine-tuning from v1 ({copied} layers copied) · v1 validation ink-L1 {v1_val:.4f} · {steps} steps × {batch} × {patch}px")

    best = {"val": v1_val, "step": 0, "weights": full.get_weights()}
    history, t0, losses = [], time.time(), []
    for step in range(1, steps + 1):
        deg, clean = sampler.sample(batch)
        losses.append(float(full.train_on_batch((deg[:, 0, :, :, None] + 1) / 2, (clean[:, 0, :, :, None] + 1) / 2)))
        if step % eval_every == 0 or step == steps:
            val = float(ink_l1(val_clean, full.predict(val_deg, verbose=0)))
            history.append({"step": step, "train_loss": float(np.mean(losses)), "val_ink_l1": val,
                            "seconds": round(time.time() - t0, 1)})
            losses = []
            if val < best["val"]:
                best = {"val": val, "step": step, "weights": full.get_weights()}
            log(f"step {step:5d} | train {history[-1]['train_loss']:.4f} | val ink-L1 {val:.4f} (v1 {v1_val:.4f}) | "
                f"best {best['val']:.4f} @ {best['step']} | {time.time() - t0:.0f}s")

    full.set_weights(best["weights"])
    encoder, decoder = split(full, tensors)
    encoder.save(ENCODER_V2)
    decoder.save(DECODER_V2)
    config = {"model": "Resume_Denoising_Autoencoder_v2", "fine_tuned_from": ["encoder.h5", "decoder.h5"],
              "data": "Kaggle Resume Dataset PDFs (first page, 150 dpi) + controlled degradations",
              "loss": "L1, ink pixels weighted x5", "steps": steps, "batch": batch, "patch": patch,
              "learning_rate": lr, "seed": seed, "v1_val_ink_l1": v1_val, "best_val_ink_l1": best["val"],
              "best_step": best["step"], "seconds": round(time.time() - t0, 1),
              "trained_at": datetime.now().isoformat(timespec="seconds"), "history": history}
    with open(CONFIG_V2, "w") as f:
        json.dump(config, f, indent=2)
    log(f"Saved encoder_v2.h5 + decoder_v2.h5 · best val ink-L1 {best['val']:.4f} (v1 {v1_val:.4f})")
    return config


def main(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune the resume autoencoder (writes *_v2.h5, keeps the originals)")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if a.quick:
        a.steps, a.batch = 10, 4
    finetune(a.steps, a.batch, a.patch, a.lr, eval_every=5 if a.quick else 100, seed=a.seed)


if __name__ == "__main__":
    main()
