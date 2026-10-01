# Blended Chart Surfaces

Code for **"Blended Chart Surfaces: A Seamless Explicit Representation for Smooth Surface Fitting"**, Romy Williamson and Niloy Mitra. Journal paper at Pacific Graphics 2026 / Computer Graphics Forum.

A Blended Chart Surface (BCS) is built on a coarse triangle mesh: each vertex has a polynomial patch associated with it, and the patches are blended across each triangle into one surface with smoothness guarantees. Code is available here to fit a BCS to a target shape given as an implicit function (like an sdf, but it doesn't have to be a strict sdf, either analytic or a pre-trained neural), by optimising the polynomial coefficients.

We provide a network structure in "deepsdf_models.py" and weights in the "BCS-neuralSDF-data" release, this is not part of our method but can be used as input. The available implicit shapes for testing (analytic and neural) are listed in "implicit_reps.py". You can add your own implicit function definitions (analytic or neural) to this file if you wish.

It also contains the toy 2D (curve) version of the method (used for the Shark illustration in the paper) and an implementation of the relevant part of the Djuren et al. (2025) method (just the setting with quadratic polynomial vertex-functions) that we compare against.

## Installation

```bash
git clone https://github.com/romyjw/BlendedChartSurfaces.git
cd BlendedChartSurfaces
conda env create -f environment.yml
conda activate bcs
```

The environment uses Python 3.8 and PyTorch 2.4. The code has been tested on macOS with Apple Silicon (MPS).

### Data

The coarse meshes, curves and sample triangles needed by the configs are included in the repository. The pre-trained neural SDFs of the 3D target shapes (~205 MB each, ~2.7 GB in total) are attached to the [`BCS-neuralSDF-data` release](https://github.com/romyjw/BlendedChartSurfaces/releases/tag/v1.0). Download them into `sdf_weights/surfaces/` with:

```bash
python scripts/download_weights.py            # all 13 shapes
python scripts/download_weights.py bob arm    # only some shapes
```

Shapes with an analytic SDF (e.g. `urchin`, `twisted_torus`, `wobbly_torus`, `mobius`) don't need any download.

## Usage

All notebooks can be run from anywhere inside the repository: their first cell switches to the repository root.

| Notebook | What it does |
|---|---|
| [`notebooks/surface_fitting.ipynb`](notebooks/surface_fitting.ipynb) | **Main method.** Fits a BCS to a 3D target SDF, given a config from `configs/surfaces/`, then visualises and saves the result. Instructions are in the notebook. |
| [`notebooks/curve_fitting_poly.ipynb`](notebooks/curve_fitting_poly.ipynb) | The toy 2D version: fits a blended curve to a 2D neural SDF, with interactive visualisations and verification of the equivariance properties. |
| [`notebooks/sdf_2d_fitting.ipynb`](notebooks/sdf_2d_fitting.ipynb) | Fits a small neural SDF to a 2D polyline (`data/curves/`), giving target shapes for the curve notebook. |
| [`notebooks/vis_sdf.ipynb`](notebooks/vis_sdf.ipynb) | Visualises a 3D target SDF: interactive slices, gradient magnitudes and a marching-cubes mesh. To make a coarse mesh for a new BCS, you can use the marching cubes result from here and then do quadric edge decimation in e.g. Meshlab. |

### Fitting a surface

1. Open `notebooks/surface_fitting.ipynb` and set `config_filepath` to a config in `configs/surfaces/` (e.g. `arm500-inv-exp.json`).
2. Run all cells. Section 4 runs `scripts/precomputation.py` to precompute the training samples, which takes a few minutes for a 500-face mesh.

Outputs:
- `results/<shape>/<settings>__<date>/`: the config, the proxy mesh and, if `testing = False`, per-epoch checkpoints and plots
- `models/surfaces/<proxy>.pth`: the trained polynomial coefficients
- `rendering/rendering_results/<shape>/`: renders of the final surface

The visualisations draw the BCS on subdivided sample triangles (`data/high_precision_subdiv_triangles/triangle_<mesh_res>.obj`). Levels 0–8 are included; for denser visualisations (levels 9–11), generate them first with [`scripts/make_subdiv_triangles.ipynb`](scripts/make_subdiv_triangles.ipynb).

### Configs

Each config in `configs/surfaces/` has two parts. Main fields:

| Field | Meaning |
|---|---|
| `surface-config.coarse_patches_id` | Coarse proxy mesh, `data/surfaces/<id>.obj` |
| `surface-config.blend_type` | Blend function, e.g. `pou_inv_exp` or `pou_trig` (others are listed in `BCS_fast.blend_weight` in `bcs/surface.py`) |
| `surface-config.overlap_param` | Overlap of the blend functions, between 0.1547 and 0.7321 |
| `surface-config.degree` | Polynomial degree of the vertex charts |
| `surface-config.global_scale`, `local_scales` | Scaling of the polynomials; with `local_scales: true` it is also proportional to the local edge length |
| `training-config.sdf_id` | Target SDF: an analytic shape, or `deep3d_<shape>` for a neural SDF |
| `training-config.sdf_weights_path`, `model_variant` | Checkpoint in `sdf_weights/surfaces/` and its network architecture (see `bcs/deepsdf_models.py`) |
| `training-config.sdf_transition_width` | Blends the neural SDF into a radial function far from the shape, to suppress spurious values there |
| `training-config.num_samples_per_face`, `per_face_batch_size`, `max_epochs`, `initial_lr`, `min_lr` | Optimisation settings (AdamW with a cosine learning-rate schedule) |
| `training-config.normals_reg_coeff`, `distortion_reg_coeff`, `area_weighting` | Optional regularisers (off in the provided configs) |

`fertility500-djuren-mix.json` is an experimental mix of BCS with Djuren et al.'s blend weights and one-ring coordinates. It is **not** the Djuren et al. method; see the note in `BCS_fast.__init__`.

## Repository layout

```
bcs/                  the BCS library
  surface.py            BCS_fast: the blended chart surface (charts, blending, forward pass)
  utils.py              blend functions, one-ring utilities, config naming
  differential.py       normals, curvatures and distortion via autograd
  implicit_reps.py      target SDFs: analytic shapes and neural SDF loading
  deepsdf_models.py     neural SDF architectures
  visuals.py            BCS_visualiser: rendering, colourings and mesh export
  mesh_processing.py    half-edge mesh helpers
  sdf_2d.py             2D neural SDF fitting
notebooks/            the notebooks above
  djuren/               Djuren et al. baseline (djuren.py and its notebook)
scripts/
  precomputation.py     precomputes training data (called by surface_fitting.ipynb)
  download_weights.py   downloads the neural SDFs from the GitHub release
  make_subdiv_triangles.ipynb   generates the subdivided sample triangles
  polyline_from_img.py  extracts a curve polyline from a silhouette image
configs/              surface and curve configs
data/                 coarse meshes, curves and sample triangles
sdf_weights/          neural SDFs (curves included; surfaces via download_weights.py)
```

## Djuren et al. baseline

[`notebooks/djuren/djuren.py`](notebooks/djuren/djuren.py) is our implementation of the quadratic polynomial setting from Tobias Djuren et al. (2025), *Interpolating splines over triangulated surfaces by blending vertex-centric local geometries*. It is independent of the BCS code: each vertex gets a quadratic polynomial fitted by least squares to its one-ring neighbours, and these are blended over each triangle. As discussed in the paper, their surface interpolates the coarse mesh and needs no optimisation, whereas a BCS is optimised to fit a given implicit surface.

## Citation

```bibtex
@article{williamson2026blendedchartsurfaces,
  title   = {Blended Chart Surfaces: A Seamless Explicit Representation for Smooth Surface Fitting},
  author  = {Williamson, Romy and Mitra, Niloy},
  journal = {Computer Graphics Forum},
  year    = {2026},
  note    = {Pacific Graphics 2026}
}
```
