import os
import glob
import numpy as np
import cv2

from scipy.special import lambertw
from scipy.sparse import coo_matrix

# ============================================================
# OpenCV speed flags
# ============================================================
cv2.setUseOptimized(True)
cv2.setNumThreads(0)

# ============================================================
# Retinal density model parameters for Ganglion
# These constants come from the paper
# fi(r): integrated density function (cells out to eccentricity r)
# fii(r): inverse mapping from cell-count space back to degrees
# ============================================================
A = 0.98
R2 = 1.05
DG = 33162
C = 3.4820e+04 + 0.1


def fi(r: np.ndarray | float) -> np.ndarray | float:
    """
    Integrated ganglion cell count as a function of eccentricity (in degrees).
    How many ganglion cells are there within radius r?
    """
    return C - (DG * (R2 ** 2)) / (r + R2)


def fii(r: np.ndarray) -> np.ndarray:
    """
    Inverse of the ganglion integrated function.
    Takes a cell radius (cell-count space) and returns an eccentricity in degrees.
    """
    return (DG * (R2 ** 2)) / (C - r) - R2


# ============================================================
# Retinal density model functions for Cones
# cones(r): density curve
# cones_i(r): integrated density
# cones_ii(r): inverse integrated (via LambertW)
# ============================================================
def cones(r: np.ndarray) -> np.ndarray:
    """Cone density as a function of eccentricity (degrees)."""
    return 200 * np.exp(-0.75 * r) + 11.5


def cones_i(r: np.ndarray) -> np.ndarray:
    """Integrated cone density as a function of eccentricity."""
    return 11.5 * r - 266.666666666667 * np.exp(-0.75 * r) + 266.666666666667


def cones_ii(r: np.ndarray) -> np.ndarray:
    """
    Inverse of cones_i using LambertW.
    LambertW can produce complex numbers, so later we take real() where needed.
    """
    return (2 * r) / 23 + (4 * lambertw((400 * np.exp(400 / 23 - (3 * r) / 46)) / 23, k=0)) / 3 - 1600 / 69


# ============================================================
# Helper Functions
# ============================================================
def pad_to_square_reflect(img_bgr: np.ndarray) -> np.ndarray:
    """
    Pads a non-square image to a square using reflection padding.
    Input:  BGR image of shape (H, W, 3)
    Output: square BGR image of shape (S, S, 3), where S=max(H,W)
    """
    h, w, _ = img_bgr.shape
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(img_bgr, top, bottom, left, right, borderType=cv2.BORDER_REFLECT101)


def build_pyramid(image_bgr: np.ndarray, levels: int = 6) -> list[np.ndarray]:
    """
    Builds a Gaussian pyramid: level 0 = original, level 1 = downsampled,
    level 2 = downsampled again, etc.

    Each higher level is blurrier and lower resolution.
    """
    pyr = [image_bgr]
    cur = image_bgr
    for _ in range(1, levels):
        cur = cv2.pyrDown(cur)
        pyr.append(cur)
    return pyr


# ============================================================
# Remap + Pyramid foveation
# ============================================================
def retinal_distort(
    image_bgr: np.ndarray,
    fov: float = 30,
    out_size: int = 512,
    fovea_deg: float = 2.0,
    pyramid_levels: int = 6,
    max_blur_level: int | None = 3,
    blur_gamma: float = 1.4,
) -> np.ndarray:
    """
    Creates a foveated image using:
      1) a retinal distortion (nonlinear sampling map)
      2) a Gaussian pyramid (different blur levels)
      3) blending blur level based on eccentricity

    Important:
    - The 'fovea' region is forced to blur level 0 so the center stays sharp.
    - Periphery gradually shifts toward higher blur levels.
    """

    # --- Sanity: must be square for radial mapping
    if image_bgr.shape[0] != image_bgr.shape[1]:
        raise ValueError("Input must be square (pad first).")

    in_size = image_bgr.shape[0]
    e = fov / 2.0                 # half-angle (degrees)
    in_radius = in_size / 2.0     # radius in pixels
    deg_per_pix = e / in_radius   # conversion between degrees and pixels

    # total "cells" out to edge eccentricity
    n_cells = float(fi(e))

    # Create an output grid in "cell space" spanning [-n_cells, n_cells]
    t = np.linspace(-n_cells, n_cells, num=out_size, dtype=np.float32)
    x, y = np.meshgrid(t, t)

    rad_cells = np.sqrt(x * x + y * y)            # radius in cell-space
    ang = np.arctan2(y, x).astype(np.float32)     # angle

    # mask: valid retina circle
    inside = rad_cells <= n_cells

    # Convert cell radius -> eccentricity degrees using the inverse model
    rad_deg_model = fii(rad_cells)

    # Near the center we use a linear mapping so the fovea stays stable
    fovea_cells = max(float(fi(fovea_deg)), 1e-6)
    rad_deg_linear = (rad_cells / fovea_cells) * fovea_deg

    # Blend linear center and nonlinear model outside (LINEAR BLEND, no smoothstep)
    w = np.clip((rad_deg_linear - fovea_deg) / fovea_deg, 0.0, 1.0)
    rad_deg = (1 - w) * rad_deg_linear + w * rad_deg_model
    rad_deg = np.clip(rad_deg, 0.0, e).astype(np.float32)

    # Convert degrees -> input pixels
    rad_pix = np.clip(rad_deg / deg_per_pix, 0.0, in_radius - 1.001)

    # Build sampling maps for cv2.remap
    map_x = (np.cos(ang) * rad_pix + in_radius).astype(np.float32)
    map_y = (np.sin(ang) * rad_pix + in_radius).astype(np.float32)

    # Build multi-scale blur pyramid
    pyr = build_pyramid(image_bgr, levels=pyramid_levels)

    # Determine how many blur levels are allowed
    if max_blur_level is None:
        max_blur_level = pyramid_levels - 1
    max_blur_level = int(np.clip(max_blur_level, 0, pyramid_levels - 1))

    # Choose blur level as function of eccentricity
    ecc = np.clip(rad_deg / e, 0.0, 1.0)
    level_map = (ecc ** blur_gamma) * max_blur_level
    level_map = np.clip(level_map, 0, max_blur_level)

    # Force fovea to level 0 (sharp)
    level_map = np.where(rad_deg <= fovea_deg, 0.0, level_map).astype(np.float32)

    # Accumulate weighted blend from pyramid levels
    out_acc = np.zeros((out_size, out_size, 3), dtype=np.float32)
    w_sum = np.zeros((out_size, out_size), dtype=np.float32)

    for l in range(max_blur_level + 1):
        # weight for this level based on distance to chosen level_map
        d = np.abs(level_map - l)
        w_l = np.clip(1.0 - d, 0.0, 1.0).astype(np.float32)  # no smoothstep

        # remap from pyramid level l (needs scaled coordinates)
        scale = 1.0 / (2**l)
        sampled_l = cv2.remap(
            pyr[l],
            map_x * scale,
            map_y * scale,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT101,
        ).astype(np.float32)

        out_acc += sampled_l * w_l[..., None]
        w_sum += w_l

    out = out_acc / np.maximum(w_sum[..., None], 1e-6)

    # Fill outside retina with mid-gray
    out = np.where(inside[..., None], out, 127.0)

    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================
# Sparse matrix retinal warp (Ganglion or Cones)
# ============================================================
class SparseRetinalWarp:
    """
    Builds a sparse matrix W such that:
        output_pixels = W @ input_pixels

    Each output pixel is bilinearly sampled from the input, but the mapping
    is precomputed into a sparse matrix for speed (reusable per image size).

    model: "ganglion" or "cones"
    """

    def __init__(self, fov=30, out_size=512, model="ganglion", inv=0, reuse_matrix=True):
        self.fov = float(fov)
        self.out_size = int(out_size)
        self.model = str(model).lower()
        self.inv = int(inv)
        self.reuse_matrix = bool(reuse_matrix)

        self.W = None
        self.msk = None
        self.inS = None

    def _cells_at_e(self, e):
        return float(cones_i(e)) if self.model == "cones" else float(fi(e))

    def _inv_integrated(self, rad_cells):
        if self.model == "cones":
            rad_cells = np.maximum(rad_cells, 0.0)
            rr = cones_ii(rad_cells)
            rr = np.real(rr).astype(np.float32)
            return np.maximum(rr, 0.0)
        return fii(rad_cells).astype(np.float32)

    def _integrated(self, rad_deg):
        return cones_i(rad_deg).astype(np.float32) if self.model == "cones" else fi(rad_deg).astype(np.float32)

    def build_matrix(self, in_size):
        """
        Precompute sparse mapping matrix W for a given input size.
        Only needs rebuilding if input resolution changes.
        """
        self.inS = int(in_size)
        out_size = self.out_size

        e = self.fov / 2.0
        in_radius = self.inS / 2.0
        n_cells = self._cells_at_e(e)

        if self.inv == 0:
            deg_per_pix = e / in_radius
            t = np.linspace(-n_cells, n_cells, num=out_size, dtype=np.float32)
        else:
            cell_per_pix = n_cells / in_radius
            t = np.linspace(-e, e, num=out_size, dtype=np.float32)

        xx, yy = np.meshgrid(t, t)
        x = xx.reshape(-1)
        y = yy.reshape(-1)

        ang = np.arctan2(y, x).astype(np.float32)
        rad = np.sqrt(x * x + y * y).astype(np.float32)

        if self.inv == 0:
            msk = (rad <= n_cells)
            new_r_pix = self._inv_integrated(rad) / deg_per_pix
        else:
            msk = (rad <= e)
            cell_per_pix = n_cells / in_radius
            new_r_pix = self._integrated(rad) / cell_per_pix

        col_f = (np.cos(ang) * new_r_pix + in_radius).astype(np.float32)
        row_f = (np.sin(ang) * new_r_pix + in_radius).astype(np.float32)

        # clamp to valid sampling area
        col_f = np.clip(col_f, 0.0, self.inS - 1.001)
        row_f = np.clip(row_f, 0.0, self.inS - 1.001)

        # bilinear neighbors
        c0 = np.floor(col_f).astype(np.int32)
        r0 = np.floor(row_f).astype(np.int32)
        c1 = np.clip(c0 + 1, 0, self.inS - 1)
        r1 = np.clip(r0 + 1, 0, self.inS - 1)

        dc = col_f - c0
        dr = row_f - r0

        w00 = (1 - dc) * (1 - dr)
        w01 = (dc) * (1 - dr)
        w10 = (1 - dc) * (dr)
        w11 = (dc) * (dr)

        idx00 = r0 * self.inS + c0
        idx01 = r0 * self.inS + c1
        idx10 = r1 * self.inS + c0
        idx11 = r1 * self.inS + c1

        rows = np.arange(out_size * out_size, dtype=np.int32)

        m = msk.astype(np.float32)
        w00 *= m; w01 *= m; w10 *= m; w11 *= m

        coo_r = np.concatenate([rows, rows, rows, rows])
        coo_c = np.concatenate([idx00, idx01, idx10, idx11]).astype(np.int64)
        coo_d = np.concatenate([w00, w01, w10, w11]).astype(np.float32)

        W = coo_matrix(
            (coo_d, (coo_r, coo_c)),
            shape=(out_size * out_size, self.inS * self.inS),
            dtype=np.float32,
        ).tocsr()
        W.sum_duplicates()

        self.W = W
        self.msk = msk.reshape(out_size, out_size)
        return self.W, self.msk

    def __call__(self, img_bgr_uint8):
        """
        Apply the sparse warp to a square input image.
        """
        if img_bgr_uint8.ndim != 3 or img_bgr_uint8.shape[0] != img_bgr_uint8.shape[1]:
            raise ValueError("Input must be square (pad first) and BGR (H,W,3).")

        inS = img_bgr_uint8.shape[0]
        if (self.W is None) or (not self.reuse_matrix) or (self.inS != inS):
            self.build_matrix(inS)

        vec = img_bgr_uint8.reshape(inS * inS, 3).astype(np.float32)
        out = self.W.dot(vec).reshape(self.out_size, self.out_size, 3)
        out = np.clip(out, 0, 255).astype(np.uint8)

        out = np.where(self.msk[..., None], out, 127)
        return out


# ============================================================
# Press-run exporter
# ============================================================
if __name__ == "__main__":
    input_dir = r"C:\Users\Salina Maharjan\Desktop\Internship\images"

    out_fov_dir = r"C:\Users\Salina Maharjan\Desktop\Internship\images_fov2"
    out_gang_dir = r"C:\Users\Salina Maharjan\Desktop\Internship\images_ganglion2"
    out_cone_dir = r"C:\Users\Salina Maharjan\Desktop\Internship\images_cones2"
    os.makedirs(out_fov_dir, exist_ok=True)
    os.makedirs(out_gang_dir, exist_ok=True)
    os.makedirs(out_cone_dir, exist_ok=True)

    # Output resolution
    OUT_SIZE = 512

    # Blur controls
    MAX_BLUR_LEVEL = 3
    BLUR_GAMMA = 1.4

    # Working size to keep the center sharp:
    TARGET_WORK = 1536  # try 1024 if slow

    warp_g = SparseRetinalWarp(fov=30, out_size=OUT_SIZE, model="ganglion", inv=0, reuse_matrix=True)
    warp_c = SparseRetinalWarp(fov=30, out_size=OUT_SIZE, model="cones", inv=0, reuse_matrix=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = [p for p in glob.glob(os.path.join(input_dir, "*.*"))
             if os.path.splitext(p)[1].lower() in exts]

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue

        img_sq = pad_to_square_reflect(img)

        h = img_sq.shape[0]
        if h > TARGET_WORK:
            img_sq = cv2.resize(img_sq, (TARGET_WORK, TARGET_WORK), interpolation=cv2.INTER_AREA)
        elif h < TARGET_WORK:
            img_sq = cv2.resize(img_sq, (TARGET_WORK, TARGET_WORK), interpolation=cv2.INTER_CUBIC)

        base = os.path.basename(p)

        out_fov = retinal_distort(
            img_sq,
            fov=30,
            out_size=OUT_SIZE,
            fovea_deg=2.0,
            pyramid_levels=6,
            max_blur_level=MAX_BLUR_LEVEL,
            blur_gamma=BLUR_GAMMA,
        )

        out_gang = warp_g(img_sq)
        out_cone = warp_c(img_sq)

        cv2.imwrite(os.path.join(out_fov_dir, base), out_fov)
        cv2.imwrite(os.path.join(out_gang_dir, base), out_gang)
        cv2.imwrite(os.path.join(out_cone_dir, base), out_cone)

    print("Saved remap foveated images to:", out_fov_dir)
    print("Saved ganglion sparse images to:", out_gang_dir)
    print("Saved cones sparse images to:", out_cone_dir)