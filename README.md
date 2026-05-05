# Unwrapper

Flattens the surface texture of a cylindrical 3D model into a clean 2D image. Designed for museum and cultural-heritage use with objects such as incense burners, carved gems, cylinder seals, and cameos.

## How it works

Photogrammetry tools (Agisoft Metashape, etc.) produce a UV atlas texture — the surface is cut into many small patches scattered across the texture image. Simply outputting that texture looks like a puzzle.

Unwrapper instead:
1. Finds the cylinder's main axis automatically using PCA
2. Projects every surface vertex to cylindrical coordinates (angle θ, height z)
3. Samples each vertex's colour from the original atlas texture
4. Scatters those colours into a flat output image using (angle, height) as (x, y)
5. Fills sub-pixel gaps with inward dilation, leaving large structural voids (hollow interiors) as white

Inner-surface faces (the inside walls and base of hollow vessels) are automatically detected and excluded so they don't bleed into the output.

The result is a proportionally correct strip showing the full 360° decorated surface from left to right.

## Requirements

```
pip install trimesh opencv-python numpy
```

## Basic usage

```bash
python unwrapper.py model.obj
```

Output is saved as `model_unwrapped.png` in the same folder as the OBJ file.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output FILE` / `-o FILE` | `<stem>_unwrapped.png` | Output file path |
| `--width PX` | texture width | Output width in pixels |
| `--height PX` | auto from aspect ratio | Output height in pixels |
| `--seam DEG` | `0` | Rotate the seam position by this many degrees — useful to move the join away from a feature |
| `--flip` | off | Flip output horizontally (corrects mirror-image seam orientation) |
| `--invert` | off | Invert tones — makes intaglio read as a raised impression. Saves both inverted and normal versions |
| `--texture FILE` | auto from MTL | Override the texture file (useful if the MTL path is broken) |
| `--crop FRAC` | `0` | Trim top and bottom rows whose surface coverage falls below this fraction (0 = no crop). Example: `--crop 0.05` |

## Examples

```bash
# Basic unwrap at default resolution
python unwrapper.py seal.obj

# Specify output path and width
python unwrapper.py seal.obj --output result.png --width 4096

# Move the seam to the back of the object
python unwrapper.py vase.obj --seam 180

# Intaglio seal — save both inverted (impression) and normal views
python unwrapper.py seal.obj --invert

# Manually supply a texture if auto-detection fails
python unwrapper.py model.obj --texture diffuse.jpg
```

## Input format

- OBJ file with UV coordinates (`vt` lines)
- MTL file referencing a diffuse texture (`map_Kd`), or use `--texture` to specify directly
- Texture image alongside the OBJ (JPG, PNG, TIFF, BMP)

Quoted MTL filenames (e.g. `map_Kd "My Texture.png"`) and filenames with spaces are supported.

## Notes

- Output width defaults to the texture's larger dimension; height is calculated from the physical circumference-to-height ratio of the mesh
- Both width and height can be set together to force exact dimensions
- Large structural voids (hollow cup interior, gaps in openwork) are left as white rather than being filled
- Typical processing time: ~15 s for a 2.2M-face mesh at 8K output
