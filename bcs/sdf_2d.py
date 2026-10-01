import os
import math
import numpy as np
import matplotlib.pyplot as plt


def load_polyline_obj(path):
    """Load a polyline from OBJ and automatically close it (wrap last→first)."""
    verts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 3:
                    verts.append((float(parts[1]), float(parts[2])))
    if len(verts) < 2:
        raise ValueError("OBJ file must contain at least 2 vertices.")
    verts = np.array(verts, dtype=float)
    # Always close by wrapping (no proximity test)
    verts = np.vstack([verts, verts[0]])
    return verts


    

# --- Helper: create demo polyline (circle) ---
def make_circle_polyline(radius=1.0, center=(0.0, 0.0), n=128):
    cx, cy = center
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([cx + radius * np.cos(t), cy + radius * np.sin(t)], axis=1)
    return pts

# --- Distance from points to line segments ---
def point_segment_distance(p, a, b):
    # p: (...,2) ; a: (2,) ; b: (2,)
    # returns distances (...)
    ap = p - a
    ab = b - a
    ab_len2 = (ab ** 2).sum()
    if ab_len2 == 0.0:
        # degenerate segment
        proj = 0.0
    else:
        proj = np.clip((ap * ab).sum(axis=-1) / ab_len2, 0.0, 1.0)
    closest = a + proj[..., None] * ab
    return np.linalg.norm(p - closest, axis=-1)

def unsigned_sdf_to_polyline(points, polyline):
    # points: (N,2)
    # polyline: (M,2)
    N = points.shape[0]
    M = polyline.shape[0]
    dists = np.full((N, M - 1), np.inf)
    for i in range(M - 1):
        a = polyline[i]
        b = polyline[i + 1]
        dists[:, i] = point_segment_distance(points, a, b)
    # if closed (first ~ last), include last segment
    if np.allclose(polyline[0], polyline[-1]):
        # ensure we didn't accidentally create two identical points at ends; also handle segments wrapping
        pass
    return dists.min(axis=1)

# --- Point-in-polygon (ray casting) to determine sign for closed polyline ---
def point_in_polygon(points, poly):
    # points: (N,2), poly: (M,2) (closed or open; we'll treat as closed by connecting last->first)
    x = points[:, 0]
    y = points[:, 1]
    poly_x = poly[:, 0]
    poly_y = poly[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    M = poly.shape[0]
    for i in range(M):
        j = (i + 1) % M
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        intersect = ((yi > y) != (yj > y)) & \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-20) + xi)
        inside ^= intersect  # toggle where ray crosses
    return inside

# --- Signed SDF wrapper ---
def signed_sdf(points, polyline):
    unsigned = unsigned_sdf_to_polyline(points, polyline)
    # determine if closed: first and last identical or very close
    closed = np.allclose(polyline[0], polyline[-1])
    if not closed:
        # If not closed, return unsigned distances (positive). User can still get sign by providing closed curve.
        return unsigned
    inside = point_in_polygon(points, polyline)
    return unsigned * (np.where(inside, -1.0, 1.0))

# --- Build training samples ---
def sample_training_points(polyline, n_outside=2000, n_oncurve=800, padding=0.2, ribbon_offset=0.01):
    # bounding box for sampling
    minxy = polyline.min(axis=0)
    maxxy = polyline.max(axis=0)
    size = maxxy - minxy
    minxy = minxy - padding * size - padding
    maxxy = maxxy + padding * size + padding
    # uniform samples in box
    outside = np.random.rand(n_outside, 2) * (maxxy - minxy)[None, :] + minxy[None, :]
    # sample points on curve (interpolate polyline segments) + small random normal offsets to get near-zero sdf
    M = polyline.shape[0]
    oncurve_idxs = np.random.randint(0, M - 1 if M>1 else M, size=n_oncurve)
    t = np.random.rand(n_oncurve)
    a = polyline[oncurve_idxs]
    b = polyline[oncurve_idxs + 1]
    oncurve = a * (1 - t)[:, None] + b * t[:, None]
    # small offsets both sides to give signed information (if closed) and gradient info
    normals = np.zeros_like(oncurve)
    # simple normals from segment direction rotated by 90 deg
    segs = b - a
    segs_len = np.linalg.norm(segs, axis=1, keepdims=True) + 1e-8
    unit = segs / segs_len
    normals[:, 0] = -unit[:, 1]
    normals[:, 1] = unit[:, 0]
    offsets = (np.random.randn(n_oncurve, 1) * ribbon_offset) * normals
    oncurve = oncurve + offsets
    pts = np.vstack([outside, oncurve])
    sdf_vals = signed_sdf(pts, polyline)
    return pts, sdf_vals

# --- Simple MLP using PyTorch to fit SDF ---
def fit_sdf_network(points, sdf_vals, hidden=64, layers=3, lr=1e-3, epochs=1500, device=None, print_every=200):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception as e:
        raise ImportError("PyTorch is required for training the network. Please install torch. Error: " + str(e))
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    X = torch.tensor(points, dtype=torch.float32, device=device)
    y = torch.tensor(sdf_vals, dtype=torch.float32, device=device).unsqueeze(-1)
    # MLP
    layers_list = []
    in_ch = 2
    for i in range(layers):
        layers_list.append(nn.Linear(in_ch, hidden))
        layers_list.append(nn.ReLU())
        in_ch = hidden
    layers_list.append(nn.Linear(in_ch, 1))
    model = nn.Sequential(*layers_list).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    for e in range(1, epochs + 1):
        opt.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        opt.step()
        if (e % print_every == 0) or (e == 1) or (e == epochs):
            print(f"Epoch {e}/{epochs}  loss: {loss.item():.6f}")
    # return model and device for inference
    return model, device

# --- Predict SDF on grid for visualization ---
def predict_on_grid(model, device, bb_min, bb_max, res=200):
    import torch
    xs = np.linspace(bb_min[0], bb_max[0], res)
    ys = np.linspace(bb_min[1], bb_max[1], res)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    with torch.no_grad():
        X = torch.tensor(pts, dtype=torch.float32, device=device)
        pred = model(X).cpu().numpy().reshape((res, res))
    return xx, yy, pred




import matplotlib.pyplot as plt
import numpy as np


def plot_results(
    sample_pts,
    sample_sdf,
    model=None,
    device=None,
    grid_res=600,
    polyline=None,
    out_prefix=None,
    shape_name=None,
):
    """
    Plot SDF samples and optionally the predicted field and/or polyline.
    Saves each figure as a separate PDF, by default to
    output/sdf_2d/<shape_name>/<shape_name>_*.pdf (or output/sdf_2d/results_*.pdf
    without a shape name). Pass `out_prefix` to choose another path prefix.
    `shape_name`, if given, is shown in the plot titles.
    """

    title_prefix = f"{shape_name}: " if shape_name else ""

    if out_prefix is None:
        out_prefix = (os.path.join("output", "sdf_2d", shape_name, shape_name) if shape_name
                      else os.path.join("output", "sdf_2d", "results"))
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    pad = 0.2

    # -------------------------------------------------------------------------
    # Bounding box
    # -------------------------------------------------------------------------
    if polyline is not None:
        minxy = polyline.min(axis=0)
        maxxy = polyline.max(axis=0)
    else:
        minxy = sample_pts.min(axis=0)
        maxxy = sample_pts.max(axis=0)

    size = maxxy - minxy
    bb_min = minxy - pad * size - pad
    bb_max = maxxy + pad * size + pad

    # -------------------------------------------------------------------------
    # Ground-truth SDF samples
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    sc = ax.scatter(
        sample_pts[:, 0],
        sample_pts[:, 1],
        c=sample_sdf,
        s=6,
    )
    fig.colorbar(sc, ax=ax)

    if polyline is not None:
        ax.plot(polyline[:, 0], polyline[:, 1], "k-", linewidth=1.5)
        ax.set_title(f"{title_prefix}Training samples and polyline (color = ground-truth SDF)")
        fname = f"{out_prefix}_samples_polyline.pdf"
    else:
        ax.set_title(f"{title_prefix}Training samples (color = ground-truth SDF)")
        fname = f"{out_prefix}_samples.pdf"

    ax.set_aspect("equal", adjustable="box")

    plt.savefig(fname, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Model prediction
    # -------------------------------------------------------------------------
    if model is not None:
        xx, yy, pred = predict_on_grid(
            model, device, bb_min, bb_max, res=grid_res
        )

        fig, ax = plt.subplots(figsize=(8, 8))

        CS = ax.contour(xx, yy, pred, levels=40)
        #ax.clabel(CS, inline=1, fontsize=8)

        # Zero level-set
        ax.contour(
            xx,
            yy,
            pred,
            levels=[0.0],
            linewidths=2.0,
            colors="red",
        )

        if polyline is not None:
            ax.plot(polyline[:, 0], polyline[:, 1], "k--", linewidth=1.5)

        ax.set_title(f"{shape_name} Overfitted SDF" if shape_name else "Overfitted SDF")
        ax.set_aspect("equal", adjustable="box")

        plt.savefig(f"{out_prefix}_prediction.pdf", bbox_inches="tight")
        plt.show()
        plt.close(fig)
