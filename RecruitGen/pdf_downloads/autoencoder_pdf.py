"""Autoencoder page — denoising report and the cleaned resume as a print-ready PDF."""

import io

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak, Spacer

from . import theme as t
from .components import MetricCards, SectionTitle, bullets, data_table, image_grid, pil_image
from .document import body_height, build_pdf, content_width

PIPELINE = [
    "**Scale matching** — small screenshots and low-resolution scans are upscaled to the page width the network "
    "was trained on (~1240 px), so small text is not smeared.",
    "**Noise estimate** — a robust noise estimate decides whether a median pre-filter removes salt & pepper "
    "speckle before the network runs.",
    "**Encoder → latent space → decoder** — the page is processed in overlapping 256×256 tiles that are "
    "blended back together with a Hann window.",
    "**Detail fusion** — the network decides what is paper and what is ink; on clean inputs the sharp strokes "
    "of the source are kept so text stays crisp.",
    "**Finishing** — unsharp mask and levels give white paper and deep, readable ink.",
]


def build_denoise_report_pdf(result, prepared_by="", source=""):
    width = content_width()
    den = result["den"]
    if result["mode"] == "eval":
        n, d = result["metrics"]["noisy"], result["metrics"]["den"]
        cards = [
            ("MSE", f"{d[0]:.5f}", f"Noisy input {n[0]:.5f}", t.SUCCESS if d[0] < n[0] else t.WARNING),
            ("PSNR", f"{d[1]:.2f} dB", f"Noisy input {n[1]:.2f} dB", t.SUCCESS if d[1] > n[1] else t.WARNING),
            ("SSIM", f"{d[2]:.4f}", f"Noisy input {n[2]:.4f}", t.SUCCESS if d[2] > n[2] else t.WARNING),
        ]
        images = [(result["clean"], "Clean (ground truth)"), (result["noisy"], "Noisy input"),
                  (den, f"Denoised · {den.size[0]}×{den.size[1]}")]
        reduction = (n[0] - d[0]) / max(n[0], 1e-9) * 100
        comparison = data_table(
            ["Metric", "Noisy input", "Denoised", "Change", "Better when"],
            [("MSE", f"{n[0]:.5f}", f"{d[0]:.5f}",
              f"{reduction:.1f}% less error" if reduction >= 0 else f"{-reduction:.1f}% more error", "lower"),
             ("PSNR", f"{n[1]:.2f} dB", f"{d[1]:.2f} dB", f"{d[1] - n[1]:+.2f} dB", "higher"),
             ("SSIM", f"{n[2]:.4f}", f"{d[2]:.4f}", f"{d[2] - n[2]:+.4f}", "higher")],
            [width * w for w in (0.16, 0.2, 0.2, 0.24, 0.2)], align=["LEFT", "RIGHT", "RIGHT", "RIGHT", "CENTER"])
        mode, subtitle = "Evaluation demo", "Synthetic noise added to a clean page, then removed by the Autoencoder"
    else:
        d = result["metrics"]["den"]
        o = result["orig"].size
        scale = den.size[0] / o[0]
        cards = [
            ("MSE", f"{d[0]:.5f}", "Original vs reconstructed", t.PRIMARY),
            ("PSNR", f"{d[1]:.2f} dB", "Peak signal-to-noise ratio", t.ACCENT_DARK),
            ("SSIM", f"{d[2]:.4f}", "Structural similarity", t.SUCCESS),
            ("Output", f"{den.size[0]}×{den.size[1]}", f"{scale:.1f}× super-resolved" if scale > 1.01 else "Same size",
             t.INK),
        ]
        images = [(result["orig"], f"Original · {o[0]}×{o[1]}"), (den, f"Denoised · {den.size[0]}×{den.size[1]}")]
        comparison = None
        mode, subtitle = "Denoise", "Noisy resume image → encoder → latent space → decoder → clean resume"

    story = [
        SectionTitle("Quality metrics", "MSE (lower is better) · PSNR and SSIM (higher is better)"),
        MetricCards(cards),
        Spacer(1, 6),
    ]
    if comparison is not None:
        story += [Spacer(1, 6), comparison]
    story += [
        SectionTitle("Before & after"),
        image_grid(images, width, max_h=300 if len(images) == 2 else 230),
        SectionTitle("How the image was cleaned"),
        *bullets(PIPELINE, numbered=True),
        PageBreak(),
        SectionTitle("Denoised resume · full page", f"{den.size[0]}×{den.size[1]} px"),
        pil_image(den, width, body_height() - 50, max_px=2600, lossless=True),
    ]
    return build_pdf(
        story, title="Resume Image Denoising", kind="Denoising Report", eyebrow=f"Autoencoder · {mode}",
        subtitle=subtitle, meta={"Prepared by": prepared_by, "Source": source, "Model": "U-Net Autoencoder"},
    )


def build_clean_resume_pdf(image, title="Denoised resume"):
    """The denoised page alone on an A4 page — ready to print, share or upload to an ATS."""
    buf = io.BytesIO()
    w, h = A4
    c = canvas.Canvas(buf, pagesize=A4)
    c.setTitle(t.clean(title))
    c.setAuthor("RecruitGen")
    c.setCreator("RecruitGen Autoencoder")
    img = image if image.mode in ("L", "RGB") else image.convert("RGB")
    margin = 7 * mm
    scale = min((w - 2 * margin) / img.size[0], (h - 2 * margin) / img.size[1])
    dw, dh = img.size[0] * scale, img.size[1] * scale
    c.drawImage(ImageReader(img), (w - dw) / 2, (h - dh) / 2, dw, dh)
    c.showPage()
    c.save()
    return buf.getvalue()
