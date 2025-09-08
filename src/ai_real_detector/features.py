
import numpy as np
from PIL import Image, ImageStat, ImageOps, ImageFilter, ExifTags

def _to_rgb(img):
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def exif_flags(img):
    flags = {"exif_has_software": 0.0, "exif_ai_hint": 0.0}
    try:
        exif = img.getexif()
        if exif:
            exif_named = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
            software = str(exif_named.get("Software", "")).lower()
            flags["exif_has_software"] = 1.0 if software else 0.0
            ai_words = ["midjourney", "stable diffusion", "dalle", "ai", "generative"]
            flags["exif_ai_hint"] = 1.0 if any(w in software for w in ai_words) else 0.0
    except Exception:
        pass
    return flags

def hist_features(img):
    img = _to_rgb(img)
    arr = np.array(img)
    feats = {}
    for i, ch in enumerate("rgb"):
        hist, _ = np.histogram(arr[:,:,i], bins=32, range=(0,255), density=True)
        eps = 1e-9
        feats[f"hist_mean_{ch}"] = float((hist*np.arange(32)).mean())
        feats[f"hist_std_{ch}"] = float(hist.std())
        feats[f"hist_entropy_{ch}"] = float(-(hist*np.log2(hist+eps)).sum())
    return feats

def edge_features(img):
    img = _to_rgb(img)
    gray = ImageOps.grayscale(img)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.array(edges, dtype=np.float32)
    return {"edges_mean": float(arr.mean()), "edges_std": float(arr.std()), "edges_norm": float((arr/255.0).mean())}

def fft_features(img):
    img = _to_rgb(img)
    gray = ImageOps.grayscale(img).resize((256,256))
    arr = np.array(gray, dtype=np.float32)/255.0
    F = np.fft.fft2(arr)
    Fshift = np.fft.fftshift(F)
    magnitude = np.abs(Fshift)
    h, w = magnitude.shape
    center = magnitude[h//2-5:h//2+5, w//2-5:w//2+5].mean()
    high = np.delete(magnitude, np.s_[h//2-10:h//2+10], axis=0).mean()
    return {"fft_center_energy": float(center), "fft_high_energy": float(high), "fft_ratio_high_center": float(high/(center+1e-6))}

def color_stats(img):
    img = _to_rgb(img)
    stat = ImageStat.Stat(img)
    means = stat.mean; stddev = stat.stddev
    feats = {"mean_r": float(means[0]), "mean_g": float(means[1]), "mean_b": float(means[2]),
             "std_r": float(stddev[0]), "std_g": float(stddev[1]), "std_b": float(stddev[2])}
    return feats

def basic_geom(img):
    w, h = img.size
    ratio = w/(h+1e-9)
    return {"width": float(w), "height": float(h), "aspect_ratio": float(ratio)}

def extract_features(pil_image):
    image = pil_image.copy()
    feats = {}
    feats.update(basic_geom(image))
    feats.update(color_stats(image))
    feats.update(hist_features(image))
    feats.update(edge_features(image))
    feats.update(fft_features(image))
    feats.update(exif_flags(image))
    return feats
