#!/usr/bin/env python3
"""unwrapper.py

Flatten the surface texture of a 3D model (OBJ format) into a clean 2D image.
Designed for museum / cultural-heritage use with cylindrical objects such as
cylinder seals, carved gems, and cameos.

HOW IT WORKS
  Photogrammetry tools (Agisoft Metashape etc.) produce a "UV atlas" texture
  where the surface is divided into many small patches scattered across the
  texture image.  Simply outputting that texture would look like a puzzle.

  Instead, this tool:
    1. Finds the cylinder's main axis using PCA of the vertex cloud.
    2. Projects every vertex onto cylindrical coordinates: (angle θ, height z).
    3. For each vertex, samples its true colour from the atlas texture at
       the original UV coordinates.
    4. Also samples edge midpoints and face centroids for denser coverage.
    5. Scatters all those colours into a flat output image using cylindrical
       coordinates as (x, y) positions.
    6. Fills sub-pixel gaps with a fast inward-dilation pass, leaving large
       structural voids (e.g. the hollow interior of a vessel) as white.

  The result is an unwarped, proportionally correct strip showing the full
  360° surface from left to right.

USAGE
  python unwrapper.py seal.obj
  python unwrapper.py seal.obj --output result.png --invert
  python unwrapper.py seal.obj --width 4096 --flip
  python unwrapper.py seal.obj --seam 180    # rotate seam 180° to reposition join
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# 1. Load mesh and texture
# ---------------------------------------------------------------------------

def load_mesh(obj_path: str, texture_override: str = None):
    """Return (mesh, uv_per_vertex, texture_bgr) or exit with a message."""
    path = Path(obj_path)
    if not path.exists():
        sys.exit(f"ERROR: file not found: {obj_path}")

    print(f"Loading mesh: {path.name} …", flush=True)
    scene_or_mesh = trimesh.load(str(path), process=False)

    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = list(scene_or_mesh.geometry.values())
        if not meshes:
            sys.exit("ERROR: no geometry found.")
        if len(meshes) > 1:
            print(f"  Scene contains {len(meshes)} meshes – merging …")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene_or_mesh

    print(f"  {len(mesh.vertices):,} vertices  {len(mesh.faces):,} triangles")

    if not hasattr(mesh.visual, "uv") or mesh.visual.uv is None:
        sys.exit("ERROR: mesh has no UV coordinates (vt lines in the OBJ).")
    uv = np.asarray(mesh.visual.uv, dtype=np.float32)

    texture = _load_texture(mesh, path, texture_override)
    h, w = texture.shape[:2]
    print(f"  Texture: {w}×{h} px")

    return mesh, uv, texture


def _load_texture(mesh, obj_path: Path, texture_override: str = None) -> np.ndarray:
    """Return the diffuse texture as a BGR uint8 numpy array."""
    obj_dir = obj_path.parent

    # Explicit override from --texture flag
    if texture_override:
        tp = Path(texture_override)
        if not tp.is_absolute():
            tp = obj_dir / tp
        if not tp.exists():
            sys.exit(f"ERROR: texture file not found: {tp}")
        print(f"  Texture file: {tp.name} (--texture override)")
        img = cv2.imread(str(tp), cv2.IMREAD_COLOR)
        if img is None:
            sys.exit(f"ERROR: could not read texture: {tp}")
        return img

    try:
        mat = mesh.visual.material
        if hasattr(mat, "image") and mat.image is not None:
            w, h = mat.image.size
            if w > 4 and h > 4:
                arr = np.array(mat.image.convert("RGB"), dtype=np.uint8)
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    tex_path = _find_texture_from_mtl(obj_dir)

    if tex_path is None or not tex_path.exists():
        stem = obj_path.stem
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
            candidate = obj_dir / (stem + ext)
            if candidate.exists():
                tex_path = candidate
                break

    if tex_path is None or not tex_path.exists():
        sys.exit(f"ERROR: cannot locate texture image for {obj_path.name}.")

    print(f"  Texture file: {tex_path.name}")
    img = cv2.imread(str(tex_path), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"ERROR: could not read texture image: {tex_path}")
    return img


def _normalise_stem(name: str) -> str:
    """Lower-case and collapse separators for fuzzy filename matching."""
    import re
    return re.sub(r'[-_\s]+', '_', name.lower())


def _find_texture_from_mtl(obj_dir: Path):
    """Scan all MTL files in obj_dir for the first map_Kd entry."""
    # Map: exact name, lower-case name, and normalised stem → real Path
    dir_exact   = {p.name:                        p for p in obj_dir.iterdir() if p.is_file()}
    dir_lower   = {p.name.lower():                p for p in obj_dir.iterdir() if p.is_file()}
    dir_normstem = {_normalise_stem(p.stem) + p.suffix.lower(): p
                    for p in obj_dir.iterdir() if p.is_file()}

    for mtl_file in sorted(obj_dir.glob("*.mtl")):
        with open(mtl_file, "r", errors="ignore") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped.lower().startswith("map_kd"):
                    continue
                rest = stripped[6:].strip()
                if rest.startswith('"') or rest.startswith("'"):
                    tex_name = rest[1:].split(rest[0])[0]
                else:
                    tex_name = rest.split()[0]
                p = Path(tex_name)
                # 1. Exact path
                candidate = obj_dir / tex_name
                if candidate.exists():
                    return candidate
                # 2. Case-insensitive
                ci = dir_lower.get(p.name.lower())
                if ci:
                    return ci
                # 3. Normalised separators (e.g. underscores vs hyphens)
                key = _normalise_stem(p.stem) + p.suffix.lower()
                ci = dir_normstem.get(key)
                if ci:
                    return ci
    return None


# ---------------------------------------------------------------------------
# 2. Cylindrical projection
# ---------------------------------------------------------------------------

def cylindrical_projection(vertices: np.ndarray, faces: np.ndarray,
                            seam_offset_deg: float = 0.0):
    """
    Project 3D vertices onto a perfect cylinder.

    Axis detection selects the PCA eigenvector with minimum radial-distance
    variance — robust to both tall and short vessels regardless of aspect ratio.
    The projection maps every vertex by (angle around axis, height along axis)
    only; radial distance from the axis is discarded.

    Inward-facing faces (inner walls and base of hollow vessels) are identified
    by computing each face normal's dot product with the local radial outward
    direction.  Faces with a significantly negative outward component are
    excluded via the returned mask so that interior texture does not bleed into
    the unwrapped output.

    Parameters
    ----------
    vertices        : (N, 3) vertex positions
    faces           : (F, 3) face indices
    seam_offset_deg : rotate seam location by this many degrees

    Returns
    -------
    angle_norm   : (N,) float32  normalised angle  [0, 1]
    height_norm  : (N,) float32  normalised height [0, 1]
    aspect_ratio : float         circumference / height
    outward_mask : (F,) bool     True = face should be rendered
    """
    verts = np.asarray(vertices, dtype=np.float64)
    center = verts.mean(axis=0)
    v = verts - center

    _, eigvecs = np.linalg.eigh(np.cov(v.T))
    main_axis, best_var = None, np.inf
    for i in range(3):
        ax = eigvecs[:, i]
        along = v @ ax
        radial = np.linalg.norm(v - np.outer(along, ax), axis=1)
        var = radial.var()
        if var < best_var:
            best_var, main_axis = var, ax

    print(f"  Axis (PCA min-radial-var): "
          f"[{main_axis[0]:.3f}, {main_axis[1]:.3f}, {main_axis[2]:.3f}]  "
          f"(radial var {best_var:.5f})", flush=True)

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, main_axis)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    perp1 = ref - np.dot(ref, main_axis) * main_axis
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(main_axis, perp1)

    height = v @ main_axis
    rad_x  = v @ perp1
    rad_y  = v @ perp2

    angle_rad  = np.arctan2(rad_y, rad_x) + np.radians(seam_offset_deg)
    angle_norm = ((angle_rad % (2.0 * np.pi)) / (2.0 * np.pi)).astype(np.float32)

    h_min, h_max = height.min(), height.max()
    height_norm = ((height - h_min) / max(h_max - h_min, 1e-9)).astype(np.float32)

    mid = (h_min + h_max) / 2.0
    near_mid = np.abs(height - mid) < (h_max - h_min) * 0.1
    radius = (np.sqrt(rad_x[near_mid]**2 + rad_y[near_mid]**2).mean()
              if near_mid.any() else 1.0)
    circumference = 2.0 * np.pi * radius
    phys_height   = h_max - h_min
    aspect_ratio  = circumference / max(phys_height, 1e-9)

    print(f"  Circumference ≈ {circumference:.3f}  Height ≈ {phys_height:.3f}  "
          f"Aspect ≈ {aspect_ratio:.2f}:1", flush=True)

    # ── Outward-facing face filter ──────────────────────────────────────────
    # Photogrammetry meshes of hollow vessels contain inner-surface faces
    # (interior walls, base) whose normals point toward the axis.  Projecting
    # those would mix interior texture into the output and create ragged edges
    # at the rim and base.  We exclude them by checking the dot product of each
    # face normal with the local radial outward direction.
    fa, fb, fc = faces[:, 0], faces[:, 1], faces[:, 2]

    # Edge vectors in the local (perp1, perp2, main_axis) frame
    e1 = np.column_stack([rad_x[fb] - rad_x[fa],
                          rad_y[fb] - rad_y[fa],
                          height[fb] - height[fa]])
    e2 = np.column_stack([rad_x[fc] - rad_x[fa],
                          rad_y[fc] - rad_y[fa],
                          height[fc] - height[fa]])

    fn = np.cross(e1, e2)
    fn_len = np.linalg.norm(fn, axis=1, keepdims=True).clip(1e-9)
    fn /= fn_len                                              # unit normals (F, 3)

    # Radial outward unit vector at each face centre
    cx = (rad_x[fa] + rad_x[fb] + rad_x[fc]) / 3.0
    cy = (rad_y[fa] + rad_y[fb] + rad_y[fc]) / 3.0
    cr = np.sqrt(cx ** 2 + cy ** 2).clip(1e-9)
    outward = fn[:, 0] * (cx / cr) + fn[:, 1] * (cy / cr)

    # Sanity-check winding: if the majority of faces appear to point inward
    # the mesh's winding is reversed — flip the filter so it still works.
    if outward.mean() < -0.3:
        print("  Note: mesh normals appear inverted — adjusting filter.", flush=True)
        outward = -outward

    # Keep faces pointing outward or at most ~84° off outward (dot > -0.1).
    # This retains the outer body, the outer rim, and slightly undercut surfaces
    # while reliably excluding inner walls and the hollow interior base.
    outward_mask = outward > -0.1

    n_excl = int((~outward_mask).sum())
    if n_excl > 0:
        print(f"  Outward filter: {int(outward_mask.sum()):,} faces kept, "
              f"{n_excl:,} inner-surface faces excluded", flush=True)

    return angle_norm, height_norm, aspect_ratio, outward_mask


# ---------------------------------------------------------------------------
# 3. Render
# ---------------------------------------------------------------------------

def render(angle_norm: np.ndarray, height_norm: np.ndarray,
           faces: np.ndarray, uv: np.ndarray,
           texture: np.ndarray,
           out_w: int, out_h: int,
           scale: int = 1) -> tuple:
    """
    Produce the flat cylindrical unwrap.

    For every vertex, edge midpoint, and face centroid in the mesh:
      • Sample colour from the atlas texture at the original UV coords.
      • Place that colour at the (angle, height) position in the output.

    Sub-pixel gaps (mesh density < output resolution) are filled by inward
    dilation.  Dilation stops at a maximum of ~5 px so that large structural
    voids (the hollow inside of a cup, gaps between handles, etc.) stay white
    rather than being smeared over.
    """
    tex_h, tex_w = texture.shape[:2]
    F = len(faces)

    # ── Step 1: Build sample positions (vertices + barycentric oversampling) ──
    print("  Building sample set …", flush=True)

    # Build vertex mask from the pre-filtered face set.
    # This excludes inner-surface vertices (e.g. inside a cup) so that only
    # vertices belonging to outward-facing faces contribute to the scatter.
    vert_mask = np.zeros(len(angle_norm), dtype=bool)
    vert_mask[faces.ravel()] = True
    N = int(vert_mask.sum())

    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
    a0, a1, a2 = angle_norm[f0], angle_norm[f1], angle_norm[f2]

    # Detect seam-crossing faces (angle span > 0.5 wraps the 0/1 boundary)
    a_span = np.maximum(np.maximum(a0, a1), a2) - np.minimum(np.minimum(a0, a1), a2)
    not_seam = a_span < 0.5

    # Adaptive barycentric oversampling for non-seam faces.
    # Target ≥ 0.10 samples per render pixel so that sparse low-poly meshes
    # (e.g. 150k-face vases) reach the same scatter density as the high-poly
    # meshes (1–3M faces) that naturally produce clean, smooth output.
    # We generate a regular (i/N, j/N, k/N) grid over each face, excluding
    # the 3 corner vertices (already covered by the global vertex pass).
    n_ns        = int(not_seam.sum())
    n_nc_target = max(3, int((0.10 * out_w * out_h - N) / max(n_ns, 1)))
    N_sub       = max(2, min(15, int((-3 + np.sqrt(8 * n_nc_target + 25)) / 2)))

    bary_nc = np.array(
        [(i / N_sub, j / N_sub, (N_sub - i - j) / N_sub)
         for i in range(N_sub + 1) for j in range(N_sub + 1 - i)
         if not (i == N_sub or j == N_sub or N_sub - i - j == N_sub)],
        dtype=np.float32)           # (n_nc, 3)
    n_nc = len(bary_nc)

    ns0 = faces[not_seam, 0]; ns1 = faces[not_seam, 1]; ns2 = faces[not_seam, 2]

    def _bary(field):
        return (np.outer(field[ns0], bary_nc[:, 0]) +
                np.outer(field[ns1], bary_nc[:, 1]) +
                np.outer(field[ns2], bary_nc[:, 2])).ravel()

    all_a = np.concatenate([angle_norm[vert_mask], _bary(angle_norm)])
    all_h = np.concatenate([height_norm[vert_mask], _bary(height_norm)])
    all_u = np.concatenate([uv[vert_mask, 0],       _bary(uv[:, 0])])
    all_v = np.concatenate([uv[vert_mask, 1],       _bary(uv[:, 1])])

    print(f"  {len(all_a):,} samples  "
          f"({N:,} verts + {n_ns:,} faces × {n_nc} bary pts, N_sub={N_sub})",
          flush=True)

    # ── Step 2: Sample colours with bilinear interpolation ───────────────────
    fx = (all_u * (tex_w - 1)).astype(np.float32)
    fy = ((1.0 - all_v) * (tex_h - 1)).astype(np.float32)

    x0 = np.floor(fx).astype(np.int32).clip(0, tex_w - 1)
    x1 = (x0 + 1).clip(0, tex_w - 1)
    y0 = np.floor(fy).astype(np.int32).clip(0, tex_h - 1)
    y1 = (y0 + 1).clip(0, tex_h - 1)

    wx = (fx - x0)[:, np.newaxis]   # weight toward x1
    wy = (fy - y0)[:, np.newaxis]   # weight toward y1

    c00 = texture[y0, x0].astype(np.float32)
    c01 = texture[y0, x1].astype(np.float32)
    c10 = texture[y1, x0].astype(np.float32)
    c11 = texture[y1, x1].astype(np.float32)

    colors = ((1 - wy) * ((1 - wx) * c00 + wx * c01) +
                   wy  * ((1 - wx) * c10 + wx * c11))

    # ── Step 3: Scatter into output image ─────────────────────────────────────
    print("  Scattering colours …", flush=True)
    ox = (all_a * out_w).clip(0, out_w - 1).astype(np.int32)
    oy = ((1.0 - all_h) * out_h).clip(0, out_h - 1).astype(np.int32)

    color_acc = np.zeros((out_h, out_w, 3), dtype=np.float32)
    count_acc = np.zeros((out_h, out_w),    dtype=np.int32)
    np.add.at(color_acc, (oy, ox), colors.astype(np.float32))
    np.add.at(count_acc, (oy, ox), 1)

    has_data = count_acc > 0
    avg = np.where(has_data[:, :, np.newaxis],
                   color_acc / count_acc[:, :, np.newaxis].clip(1), 0.0
                   ).astype(np.uint8)
    mask = has_data.astype(np.uint8) * 255

    coverage = has_data.mean()
    print(f"  Scatter coverage: {coverage*100:.1f}%  Gap-filling …", flush=True)

    # ── Step 4: Fill sub-pixel gaps only (not large structural voids) ─────────
    # Strategy: dilate in small steps; stop once dilation fills all connected
    # components of empty pixels that touch data within ~5 px.
    # We track the ORIGINAL mask to never "grow" into already-white pixels.
    filled       = avg.copy()
    filled_mask  = mask.copy()
    orig_mask    = mask.copy()

    for ksize in (3, 3, 3, 5, 5):
        ks = max(3, ksize * scale) | 1   # scale up, keep odd
        k = np.ones((ks, ks), np.uint8)
        dilated      = cv2.dilate(filled, k)
        dilated_mask = cv2.dilate(filled_mask, k)
        filled       = np.where(filled_mask[:, :, np.newaxis] > 0, filled, dilated)
        filled_mask  = dilated_mask

    # White background for areas never covered by the surface
    bg_mask = filled_mask == 0          # True = pure background (never covered)
    result = np.where(filled_mask[:, :, np.newaxis] > 0, filled,
                      np.full_like(filled, 255))

    final_coverage = (~bg_mask).mean()
    print(f"  Final coverage:   {final_coverage*100:.1f}%", flush=True)

    return result, bg_mask


# ---------------------------------------------------------------------------
# 4. Post-processing helpers
# ---------------------------------------------------------------------------

def inpaint_holes(result: np.ndarray, bg_mask: np.ndarray) -> tuple:
    """
    Fill small isolated surface holes (mesh gaps) by inpainting from neighbours.
    Holes larger than 0.1% of the image are left untouched so that structural
    voids (hollow cup interior, etc.) remain white.
    Returns (inpainted_result, updated_bg_mask).
    """
    h, w = result.shape[:2]
    max_hole_px = max(100, int(0.001 * h * w))

    mask_u8 = bg_mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8)

    inpaint_mask = np.zeros((h, w), dtype=np.uint8)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] <= max_hole_px:
            inpaint_mask[labels == lbl] = 1

    n_px = int(inpaint_mask.sum())
    if n_px:
        print(f"  Inpainting {n_px:,} hole pixels ({n_labels-1} regions, "
              f"threshold {max_hole_px:,} px) …", flush=True)
        result   = cv2.inpaint(result, inpaint_mask, 3, cv2.INPAINT_TELEA)
        bg_mask  = bg_mask & (inpaint_mask == 0)

    return result, bg_mask


def autocrop(result: np.ndarray, bg_mask: np.ndarray,
             threshold: float) -> tuple:
    """
    Trim top and bottom rows whose surface coverage is below *threshold* (0–1).
    threshold=0 disables cropping.
    Returns (cropped_result, cropped_bg_mask).
    """
    if threshold <= 0:
        return result, bg_mask

    row_coverage = (~bg_mask).mean(axis=1)
    kept = np.where(row_coverage >= threshold)[0]
    if len(kept) == 0:
        return result, bg_mask

    top, bottom = int(kept[0]), int(kept[-1]) + 1
    original_h = result.shape[0]
    print(f"  Crop: rows {top}–{bottom-1} kept of {original_h}  "
          f"(threshold {threshold*100:.1f}% row coverage)", flush=True)
    return result[top:bottom], bg_mask[top:bottom]


def invert_image(img: np.ndarray) -> np.ndarray:
    """Invert tones – makes intaglio read as raised (wax impression look)."""
    return 255 - img


def flip_horizontal(img: np.ndarray) -> np.ndarray:
    """Flip left-right (corrects mirror-image seam orientation)."""
    return cv2.flip(img, 1)


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Flatten a cylindrical 3D model texture to a 2D image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("obj",  help="Path to the OBJ file")
    ap.add_argument("--output", "-o",
                    help="Output filename (default: <stem>_unwrapped.png)")
    ap.add_argument("--width",  type=int, default=None,
                    help="Output width in pixels  (default: texture width)")
    ap.add_argument("--height", type=int, default=None,
                    help="Output height in pixels (default: auto from aspect ratio)")
    ap.add_argument("--invert", action="store_true",
                    help="Invert tones (intaglio → impression)")
    ap.add_argument("--flip",   action="store_true",
                    help="Flip output horizontally")
    ap.add_argument("--seam",   type=float, default=0.0,
                    help="Rotate seam position by this many degrees (default: 0)")
    ap.add_argument("--texture", default=None,
                    help="Path to texture image (overrides MTL lookup)")
    ap.add_argument("--crop",   type=float, default=0.0, metavar="FRAC",
                    help="Min row coverage fraction to keep after unwrap "
                         "(0 = no crop, default 0).  Higher = tighter crop.")
    return ap.parse_args()


def main():
    args = parse_args()

    mesh, uv, texture = load_mesh(args.obj, texture_override=args.texture)
    tex_h, tex_w = texture.shape[:2]

    # ── Cylindrical projection (done once) ───────────────────────────────────
    faces_all = np.asarray(mesh.faces, dtype=np.int32)
    angle_norm, height_norm, aspect, outward_mask = cylindrical_projection(
        np.asarray(mesh.vertices, dtype=np.float64),
        faces_all,
        seam_offset_deg=args.seam)

    # Only render outward-facing faces; inner surfaces are excluded so they
    # don't mix interior texture into the unwrapped output.
    faces_render = faces_all[outward_mask]

    # ── Determine output dimensions ──────────────────────────────────────────
    if args.width is not None and args.height is not None:
        out_w, out_h = args.width, args.height
    elif args.width is not None:
        out_w = args.width
        out_h = max(1, int(round(out_w / aspect)))
    elif args.height is not None:
        out_h = args.height
        out_w = max(1, int(round(out_h * aspect)))
    else:
        out_w = max(tex_w, tex_h)
        out_h = max(1, int(round(out_w / aspect)))

    print(f"Output: {out_w}×{out_h} px  (rendering at 2× for supersampling)", flush=True)

    # ── Render at 2× then downscale (supersampling) ──────────────────────────
    result, bg_mask = render(angle_norm, height_norm,
                             faces_render,
                             uv, texture, out_w * 2, out_h * 2, scale=2)

    result  = cv2.resize(result,  (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    bg_mask = cv2.resize(bg_mask.astype(np.uint8) * 255, (out_w, out_h),
                         interpolation=cv2.INTER_AREA) > 127
    print(f"  Downscaled to {out_w}×{out_h} px", flush=True)

    # ── Inpaint small mesh holes ─────────────────────────────────────────────
    result, bg_mask = inpaint_holes(result, bg_mask)

    # ── Crop ragged top/bottom edges ─────────────────────────────────────────
    result, bg_mask = autocrop(result, bg_mask, threshold=args.crop)

    # ── Mild Gaussian blur to smooth aliasing without losing detail ──────────
    result = cv2.GaussianBlur(result, (0, 0), sigmaX=0.7, sigmaY=0.7)
    result[bg_mask] = 255   # restore clean white background after blur

    if args.flip:
        result  = flip_horizontal(result)
        bg_mask = cv2.flip(bg_mask.astype(np.uint8), 1).astype(bool)
        print("  Applied horizontal flip")

    # ── Determine output path (always PNG) ──────────────────────────────────
    if args.output:
        out_path = Path(args.output).with_suffix(".png")
    else:
        stem = Path(args.obj).stem
        out_path = Path(args.obj).parent / f"{stem}_unwrapped.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    h_final, w_final = result.shape[:2]
    print(f"Final size: {w_final}×{h_final} px", flush=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.invert:
        inv = invert_image(result)
        inv[bg_mask] = 255          # keep structural background white after inversion
        _save(inv, out_path)
        print(f"\nSaved (inverted):  {out_path}")
        normal_path = out_path.with_stem(out_path.stem + "_normal")
        _save(result, normal_path)
        print(f"Saved (normal):    {normal_path}")
    else:
        _save(result, out_path)
        print(f"\nSaved: {out_path}")


def _save(img: np.ndarray, path: Path):
    ok = cv2.imwrite(str(path), img)
    if not ok:
        sys.exit(f"ERROR: could not write {path}")


if __name__ == "__main__":
    main()
