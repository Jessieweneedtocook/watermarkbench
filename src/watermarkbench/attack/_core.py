
import io
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import subprocess
import tempfile
import shutil
from pathlib import Path
from torchvision.transforms import InterpolationMode
from PIL import Image
import torchvision.io as tvio

try:
    import kornia
    import kornia.filters as Kf
    import kornia.enhance as Ke
    _HAS_KORNIA = True
except Exception:
    _HAS_KORNIA = False

try:
    from compressai.zoo import cheng2020_anchor
    _HAS_COMPRESSAI = True
except Exception:
    _HAS_COMPRESSAI = False

# ============================================================
# 1) Shape + Range Adapters
# ============================================================
def _detect_layout(x: torch.Tensor) -> str:
    if x.dim() == 2:
        return "HW"
    if x.dim() == 3:
        return "CHW" if x.shape[0] <= 4 else "HWC"
    if x.dim() == 4:
        return "BCHW" if x.shape[1] <= 4 else "BHWC"
    raise ValueError(f"Unsupported tensor dim={x.dim()} with shape={tuple(x.shape)}")


def _to_bchw(x: torch.Tensor):

    layout = _detect_layout(x)
    meta = {
        "layout": layout,
        "dtype": x.dtype,
        "device": x.device,
    }

    if layout == "HW":
        x_bchw = x.unsqueeze(0).unsqueeze(0)               
    elif layout == "CHW":
        x_bchw = x.unsqueeze(0)                            
    elif layout == "HWC":
        x_bchw = x.permute(2, 0, 1).unsqueeze(0)          
    elif layout == "BCHW":
        x_bchw = x
    elif layout == "BHWC":
        x_bchw = x.permute(0, 3, 1, 2)                     
    else:
        raise ValueError(f"Unsupported layout={layout}")

    return x_bchw, meta


def _from_bchw(x_bchw: torch.Tensor, meta: dict) -> torch.Tensor:
    layout = meta["layout"]

    if layout == "HW":
        return x_bchw[0, 0]                               
    if layout == "CHW":
        return x_bchw[0]                                   
    if layout == "HWC":
        return x_bchw[0].permute(1, 2, 0)                  
    if layout == "BCHW":
        return x_bchw
    if layout == "BHWC":
        return x_bchw.permute(0, 2, 3, 1)                  

    raise ValueError(f"Unsupported layout={layout}")


def _detect_range_mode(x: torch.Tensor) -> str:

    if not torch.is_floating_point(x):
        return "uint8_255"

    safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_min = float(safe.min().item())
    x_max = float(safe.max().item())

    if x_min >= -1.0001 and x_max <= 1.0001:
        return "float_0_1" if x_min >= -0.0001 else "float_-1_1"

    if x_min >= -0.0001 and x_max <= 255.0001:
        return "float_0_1" if x_max <= 1.0001 else "float_0_255"

    return "float_-1_1"


def _to_minus1_1(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "uint8_255":
        x01 = (x.to(torch.float32) / 255.0).clamp(0.0, 1.0)
        return (x01 * 2.0 - 1.0).clamp(-1.0, 1.0)

    if mode == "float_0_1":
        x01 = x.to(torch.float32).clamp(0.0, 1.0)
        return (x01 * 2.0 - 1.0).clamp(-1.0, 1.0)

    if mode == "float_0_255":
        x01 = (x.to(torch.float32) / 255.0).clamp(0.0, 1.0)
        return (x01 * 2.0 - 1.0).clamp(-1.0, 1.0)

    return x.to(torch.float32).clamp(-1.0, 1.0)


def _from_minus1_1(x_m11: torch.Tensor, mode: str, orig_dtype: torch.dtype) -> torch.Tensor:
    x_m11 = x_m11.clamp(-1.0, 1.0)

    if mode == "uint8_255":
        x01 = (x_m11 + 1.0) / 2.0
        x255 = (x01 * 255.0).round().clamp(0.0, 255.0)
        return x255.to(torch.uint8)

    if mode == "float_0_1":
        x01 = (x_m11 + 1.0) / 2.0
        return x01.clamp(0.0, 1.0).to(orig_dtype)

    if mode == "float_0_255":
        x01 = (x_m11 + 1.0) / 2.0
        x255 = (x01 * 255.0).clamp(0.0, 255.0)
        return x255.to(orig_dtype)

    return x_m11.to(orig_dtype)


def _apply_attack_preserve(x: torch.Tensor, attack_fn, *args, **kwargs) -> torch.Tensor:

    x_bchw, meta = _to_bchw(x)
    range_mode = _detect_range_mode(x_bchw)

    x_m11 = _to_minus1_1(x_bchw, range_mode).to(meta["device"])
    y_m11 = attack_fn(x_m11, *args, **kwargs).clamp(-1.0, 1.0)

    y_bchw = _from_minus1_1(y_m11, range_mode, meta["dtype"]).to(meta["device"])
    y = _from_bchw(y_bchw, meta)
    return y


# ============================================================
# 2) Geometric Attacks
# ============================================================
def rotate_tensor(x: torch.Tensor, angle: float) -> torch.Tensor:
    def _core(z: torch.Tensor):
        return TF.rotate(
            z,
            angle=float(angle),
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=-1.0,
        )
    return _apply_attack_preserve(x, _core)


def crop(x: torch.Tensor, pct: float) -> torch.Tensor:

    def _core(z: torch.Tensor):
        _, _, H, W = z.shape

        dy = int(round(H * (float(pct) / 100.0)))
        dx = int(round(W * (float(pct) / 100.0)))

        dy = max(0, min(dy, (H - 2) // 2))
        dx = max(0, min(dx, (W - 2) // 2))

        if dy == 0 and dx == 0:
            return z

        new_h = H - 2 * dy
        new_w = W - 2 * dx

        cropped = TF.crop(z, top=dy, left=dx, height=new_h, width=new_w)
        resized = TF.resize(
            cropped,
            size=[H, W],
            interpolation=InterpolationMode.BILINEAR,
            antialias=False,
        )
        return resized

    return _apply_attack_preserve(x, _core)


def scaled(x: torch.Tensor, scale: float) -> torch.Tensor:

    def _core(z: torch.Tensor):
        _, _, H, W = z.shape
        s = float(scale)

        if s > 3.0:
            s = 1.0 + (s / 100.0)

        if s <= 0:
            return z

        new_h = max(1, int(round(H * s)))
        new_w = max(1, int(round(W * s)))

        bigger = TF.resize(z, [new_h, new_w], interpolation=InterpolationMode.BILINEAR, antialias=False)
        back   = TF.resize(bigger, [H, W], interpolation=InterpolationMode.BILINEAR, antialias=False)
        return back

    return _apply_attack_preserve(x, _core)


def flipping(x: torch.Tensor, mode: str) -> torch.Tensor:
    def _core(z: torch.Tensor):
        m = str(mode).upper()
        if m == "H":
            return torch.flip(z, dims=[3])
        if m == "V":
            return torch.flip(z, dims=[2])
        if m == "B":
            return torch.flip(z, dims=[2, 3])
        return z
    return _apply_attack_preserve(x, _core)


def resized(x: torch.Tensor, pct: int) -> torch.Tensor:
    def _core(z: torch.Tensor):
        _, _, H, W = z.shape
        level_ratio = int(pct) / 100.0
        down = max(0.2, 1.0 - level_ratio)

        new_h = max(1, int(round(H * down)))
        new_w = max(1, int(round(W * down)))

        small = TF.resize(z, size=[new_h, new_w], interpolation=InterpolationMode.BILINEAR, antialias=False)
        back  = TF.resize(small, size=[H, W], interpolation=InterpolationMode.BILINEAR, antialias=False)
        return back

    return _apply_attack_preserve(x, _core)


# ============================================================
# 3) Signal Processing Attacks
# ============================================================
def jpeg_compression(x: torch.Tensor, quality: int) -> torch.Tensor:
    def _core(z: torch.Tensor):
        device = z.device
        z_cpu = z.detach().cpu().clamp(-1.0, 1.0)

        B, C, H, W = z_cpu.shape
        outs = []
        q = int(quality)

        for i in range(B):
            x01 = (z_cpu[i] + 1.0) / 2.0
            img_u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8) 

            jpeg_bytes = tvio.encode_jpeg(img_u8, quality=q)
            dec_u8 = tvio.decode_jpeg(jpeg_bytes)  

            dec_f = dec_u8.to(torch.float32) / 255.0
            dec_m11 = (dec_f * 2.0 - 1.0).clamp(-1.0, 1.0)
            outs.append(dec_m11)

        return torch.stack(outs, dim=0).to(device)

    return _apply_attack_preserve(x, _core)


def jpeg2000_compression(
    x: torch.Tensor,
    quality_layers=(20,),
    quality_mode="rates",
    irreversible=True,
    ext="jp2",
) -> torch.Tensor:

    def _core(z: torch.Tensor):
        device = z.device
        z_cpu = z.detach().cpu().clamp(-1.0, 1.0)
        B, C, H, W = z_cpu.shape
        outs = []

        ql = quality_layers
        if isinstance(ql, (int, float, np.integer, np.floating)):
            ql = (int(ql),)

        for i in range(B):
            x01 = (z_cpu[i] + 1.0) / 2.0
            img_u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)

            if img_u8.shape[0] == 1:
                img_np = img_u8[0].numpy()
                img_pil = Image.fromarray(img_np, mode="L")
            else:
                img_np = img_u8.permute(1, 2, 0).numpy()
                img_pil = Image.fromarray(img_np, mode="RGB")

            buf = io.BytesIO()
            img_pil.save(
                buf,
                format="JPEG2000",
                quality_mode=quality_mode,
                quality_layers=list(ql),
                irreversible=bool(irreversible),
            )
            buf.seek(0)

            dec_pil = Image.open(buf)
            if img_u8.shape[0] == 1:
                dec_pil = dec_pil.convert("L")
                dec_np = np.array(dec_pil, dtype=np.uint8)  
                dec_u8 = torch.from_numpy(dec_np).unsqueeze(0)  
            else:
                dec_pil = dec_pil.convert("RGB")
                dec_np = np.array(dec_pil, dtype=np.uint8)  
                dec_u8 = torch.from_numpy(dec_np).permute(2, 0, 1).contiguous()  

            dec_f = dec_u8.to(torch.float32) / 255.0
            dec_m11 = (dec_f * 2.0 - 1.0).clamp(-1.0, 1.0)
            outs.append(dec_m11)

        return torch.stack(outs, dim=0).to(device)

    return _apply_attack_preserve(x, _core)

# ------------------------------------------------------------
# Helpers for JPEG-AI (CompressAI)
# ------------------------------------------------------------
_COMPRESSAI_MODEL_CACHE = {}

def _get_cheng2020_anchor(device: torch.device, quality: int):
    if not _HAS_COMPRESSAI:
        raise ImportError(
            "compressai is not installed. Install it to use jpegai_compression."
        )

    q = int(quality)
    if q < 1 or q > 8:
        raise ValueError("jpegai quality must be in [1..8] for cheng2020_anchor")

    key = (str(device), q)
    if key in _COMPRESSAI_MODEL_CACHE:
        return _COMPRESSAI_MODEL_CACHE[key]

    model = cheng2020_anchor(quality=q, pretrained=True).eval().to(device)
    _COMPRESSAI_MODEL_CACHE[key] = model
    return model

def _pad_to_multiple_of_64(x01_bchw: torch.Tensor) -> tuple[torch.Tensor, int, int]:

    _, _, h, w = x01_bchw.shape
    pad_h = (64 - (h % 64)) % 64
    pad_w = (64 - (w % 64)) % 64
    x_pad = F.pad(x01_bchw, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return x_pad, h, w
def jpegai_compression(x: torch.Tensor, quality: int = 4) -> torch.Tensor:

    def _core(z_m11: torch.Tensor):
        device = z_m11.device

        
        z01 = ((z_m11 + 1.0) / 2.0).clamp(0.0, 1.0)

        B, C, H, W = z01.shape

        if C == 1:
            z01_in = z01.repeat(1, 3, 1, 1)
            single_channel = True
        elif C == 3:
            z01_in = z01
            single_channel = False
        else:
            raise ValueError(f"jpegai_compression supports only C=1 or C=3, got C={C}")

        model = _get_cheng2020_anchor(device, int(quality))

        x_pad, orig_h, orig_w = _pad_to_multiple_of_64(z01_in)

        with torch.no_grad():
            out = model(x_pad)
            x_hat = out["x_hat"].clamp(0.0, 1.0)

        x_hat = x_hat[:, :, :orig_h, :orig_w]

        if single_channel:
            x_hat = x_hat[:, :1, :, :]

        y_m11 = (x_hat * 2.0 - 1.0).clamp(-1.0, 1.0)
        return y_m11

    return _apply_attack_preserve(x, _core)

def jpegxl_compression(x: torch.Tensor, quality: int = 50) -> torch.Tensor:

    def _core(z: torch.Tensor):
        device = z.device
        z_cpu = z.detach().cpu().clamp(-1.0, 1.0)

        B, C, H, W = z_cpu.shape
        q = int(quality)
        outs = []

        for i in range(B):
            x01 = (z_cpu[i] + 1.0) / 2.0
            img_u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)

            if img_u8.shape[0] == 1:
                img_np = img_u8[0].numpy()
                img_pil = Image.fromarray(img_np, mode="L")
            else:
                img_np = img_u8.permute(1, 2, 0).numpy()
                img_pil = Image.fromarray(img_np, mode="RGB")

            buf = io.BytesIO()

            try:
                img_pil.save(buf, format="JXL", quality=q)
            except Exception as e:
                raise RuntimeError(
                    "JPEG-XL save failed. Your Pillow likely lacks JXL support. "
                    "Install/build Pillow with JXL support (or install pillow-jxl), "
                    f"then retry. Original error: {e}"
                )

            buf.seek(0)

            dec_pil = Image.open(buf)
            if img_u8.shape[0] == 1:
                dec_pil = dec_pil.convert("L")
                dec_np = np.array(dec_pil, dtype=np.uint8)
                dec_u8 = torch.from_numpy(dec_np).unsqueeze(0)
            else:
                dec_pil = dec_pil.convert("RGB")
                dec_np = np.array(dec_pil, dtype=np.uint8)
                dec_u8 = torch.from_numpy(dec_np).permute(2, 0, 1).contiguous()

            dec_f = dec_u8.to(torch.float32) / 255.0
            dec_m11 = (dec_f * 2.0 - 1.0).clamp(-1.0, 1.0)
            outs.append(dec_m11)

        return torch.stack(outs, dim=0).to(device)

    return _apply_attack_preserve(x, _core)

def jpegxs_compression(
    x: torch.Tensor,
    bitrate: str = "40M",
    pix_fmt: str = "yuv444p10le",
) -> torch.Tensor:

    def _core(z: torch.Tensor):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH. Install ffmpeg built with libsvtjpegxs "
                "(--enable-libsvtjpegxs) to use jpegxs_compression."
            )

        device = z.device
        z_cpu = z.detach().cpu().clamp(-1.0, 1.0)

        B, C, H, W = z_cpu.shape
        outs = []

        br = bitrate
        if isinstance(br, (int, float, np.integer, np.floating)):
            br = f"{int(br)}M"
        br = str(br)

        for i in range(B):
            x01 = (z_cpu[i] + 1.0) / 2.0
            img_u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)

            if img_u8.shape[0] == 1:
                img_np = img_u8[0].numpy()
                img_pil = Image.fromarray(img_np, mode="L")
            else:
                img_np = img_u8.permute(1, 2, 0).numpy()
                img_pil = Image.fromarray(img_np, mode="RGB")

            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                in_png = td / "in.png"
                out_jxs = td / "out.jxs"
                out_png = td / "out.png"

                img_pil.save(in_png)

                encode_cmd = [
                    "ffmpeg",
                    "-y",
                    "-loop", "1",
                    "-i", str(in_png),
                    "-frames:v", "1",
                    "-c:v", "libsvtjpegxs",
                    "-pix_fmt", str(pix_fmt),
                    "-b:v", br,
                    str(out_jxs),
                ]

                enc = subprocess.run(encode_cmd, capture_output=True, text=True)
                if enc.returncode != 0:
                    raise RuntimeError(
                        "JPEG-XS encode failed. Ensure ffmpeg supports libsvtjpegxs.\n"
                        f"Command: {' '.join(encode_cmd)}\n"
                        f"stderr:\n{enc.stderr}"
                    )

                decode_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i", str(out_jxs),
                    "-frames:v", "1",
                    str(out_png),
                ]

                dec = subprocess.run(decode_cmd, capture_output=True, text=True)
                if dec.returncode != 0:
                    raise RuntimeError(
                        "JPEG-XS decode failed.\n"
                        f"Command: {' '.join(decode_cmd)}\n"
                        f"stderr:\n{dec.stderr}"
                    )

                dec_pil = Image.open(out_png)
                if img_u8.shape[0] == 1:
                    dec_pil = dec_pil.convert("L")
                    dec_np = np.array(dec_pil, dtype=np.uint8)
                    dec_u8 = torch.from_numpy(dec_np).unsqueeze(0)
                else:
                    dec_pil = dec_pil.convert("RGB")
                    dec_np = np.array(dec_pil, dtype=np.uint8)
                    dec_u8 = torch.from_numpy(dec_np).permute(2, 0, 1).contiguous()

            dec_f = dec_u8.to(torch.float32) / 255.0
            dec_m11 = (dec_f * 2.0 - 1.0).clamp(-1.0, 1.0)
            outs.append(dec_m11)

        return torch.stack(outs, dim=0).to(device)

    return _apply_attack_preserve(x, _core)


def gaussian_noise(x: torch.Tensor, var: float = 0.01) -> torch.Tensor:

    def _core(z: torch.Tensor):
        z = z.clamp(-1.0, 1.0)
        v = float(var)

        if v > 1.0:
            sigma01 = v / 255.0
            v = sigma01 * sigma01

        sigma01 = math.sqrt(max(v, 0.0))
        sigma_m11 = 2.0 * sigma01  

        noise = torch.normal(
            mean=0.0,
            std=sigma_m11,
            size=z.shape,
            device=z.device,
            dtype=z.dtype,
        )
        return (z + noise).clamp(-1.0, 1.0)

    return _apply_attack_preserve(x, _core)


def speckle_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    def _core(z: torch.Tensor):
        z = z.clamp(-1.0, 1.0)
        noise = torch.randn_like(z) * float(sigma)
        return (z + z * noise).clamp(-1.0, 1.0)
    return _apply_attack_preserve(x, _core)


def blurring(x: torch.Tensor, k: int) -> torch.Tensor:
    def _core(z: torch.Tensor):
        kk = int(k)
        if kk % 2 == 0:
            kk += 1

        if _HAS_KORNIA:
            sigma = float(kk) / 6.0
            out = Kf.gaussian_blur2d(z, (kk, kk), (sigma, sigma))
            return out.clamp(-1.0, 1.0)

        sigma = float(kk) / 6.0
        out = TF.gaussian_blur(z, kernel_size=[kk, kk], sigma=[sigma, sigma])
        return out.clamp(-1.0, 1.0)

    return _apply_attack_preserve(x, _core)


def brightness(x: torch.Tensor, factor: float) -> torch.Tensor:
    def _core(z: torch.Tensor) -> torch.Tensor:
        z = z.clamp(-1.0, 1.0)
        z01 = (z + 1.0) / 2.0
        out01 = (z01 * float(factor)).clamp(0.0, 1.0)
        return (out01 * 2.0 - 1.0).clamp(-1.0, 1.0)
    return _apply_attack_preserve(x, _core)


def sharpness(x: torch.Tensor, amount: float = 1.0) -> torch.Tensor:
    def _core(z: torch.Tensor) -> torch.Tensor:
        z = z.clamp(-1.0, 1.0)
        blur = F.avg_pool2d(z, kernel_size=3, stride=1, padding=1)
        out = z + float(amount) * (z - blur)
        return out.clamp(-1.0, 1.0)
    return _apply_attack_preserve(x, _core)


def median_filtering(x: torch.Tensor, k: int) -> torch.Tensor:
    def _core(z: torch.Tensor):
        kk = int(k)
        if kk % 2 == 0:
            kk += 1

        z = z.clamp(-1.0, 1.0)

        if _HAS_KORNIA:
            out = Kf.median_blur(z, (kk, kk))
            return out.clamp(-1.0, 1.0)

        B, C, H, W = z.shape
        pad = kk // 2
        z_pad = F.pad(z, (pad, pad, pad, pad), mode="reflect")
        patches = z_pad.unfold(2, kk, 1).unfold(3, kk, 1)
        patches = patches.contiguous().view(B, C, H, W, kk * kk)
        median_vals = patches.median(dim=-1).values
        return median_vals.clamp(-1.0, 1.0)

    return _apply_attack_preserve(x, _core)


__all__ = [
    "rotate_tensor", "crop", "scaled", "flipping", "resized",
    "jpeg_compression", "jpeg2000_compression",
    "jpegai_compression", "jpegxl_compression", "jpegxs_compression",
    "gaussian_noise", "speckle_noise",
    "blurring", "brightness", "sharpness", "median_filtering",
]


