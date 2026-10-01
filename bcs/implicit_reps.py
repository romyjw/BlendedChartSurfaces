import torch
import os
import yaml
import numpy as np

from . import sdf_2d


import torch
from pathlib import Path
from .deepsdf_models import (
    Model, ModelNoPosenc, ModelNoPosenc_relu,
    ModelV3Gelu, ModelSoft1 as ModelV3Soft1, ModelSoft2 as ModelV3Soft2,
)


# Every architecture that a DeepSDF checkpoint might have been trained
# with. `load_deepsdf_model`'s `model_variant` argument picks one of
# these keys directly, instead of juggling separate posenc/activation/v3
# flags whose combinations don't all make sense together.
DEEPSDF_MODEL_VARIANTS = {
    'posenc':         Model,           # FourierEmbedder (posenc) + GELU querier
    'no_posenc_gelu': ModelNoPosenc,   # SimpleEmbedder (ReLU) + GELU querier
    'no_posenc_relu': ModelNoPosenc_relu,  # SimpleEmbedder (ReLU) + ReLU querier
    'v3_gelu':        ModelV3Gelu,     # SimpleEmbedder (ReLU) + GELU querier, default (ReLU) decoder FFN
    'v3_soft1':       ModelV3Soft1,    # SoftEmbedder (GELU) + GELU querier + GELU decoder FFN
    'v3_soft2':       ModelV3Soft2,    # SoftEmbedder (Softplus) + Softplus querier + Softplus decoder FFN
}


def check_for_nans(tensor, name='tensor'):
    if torch.isnan(tensor).any():
        print(f"\n⚠️ NaNs detected in {name}!\n")
        print(f"{name} stats -- min: {tensor.min().item()}, max: {tensor.max().item()}, mean: {tensor.mean().item()}")
        raise ValueError(f"NaNs found in {name}")


def sdf(selected_pts, sdf_id, squared=False, model=None, transition_width=0.1):
    
    if sdf_id=='analytic_torus':
        return torus_sdf(selected_pts, squared=squared)
        
    elif sdf_id=='analytic_sphere':
        return sphere_sdf(selected_pts, squared=squared)

    elif sdf_id=='urchin':
        return urchin_sdf(selected_pts, squared=squared)

    elif sdf_id=='rippley':
        return rippley_sdf(selected_pts, squared=squared)

    elif sdf_id=='wobbly_torus':
        return wobbly_torus_sdf(selected_pts, squared=squared)
        
    elif sdf_id[:6]=='deep3d':
        return deepsdf(selected_pts, model=model, squared=squared, transition_width=transition_width)
        
    elif sdf_id[:6]=='deep2d':
        return deepsdf_curve(selected_pts, model=model, squared=squared)

    elif sdf_id=='circle':
        return circle_sdf(selected_pts, squared=squared)

    elif sdf_id=='square':
        return square_sdf(selected_pts, squared=squared)
        
    elif sdf_id=='cylinder':
        return cylinder_sdf(selected_pts, squared=squared)
        
    elif sdf_id=='twisted_torus':
        return twisted_torus_sdf(selected_pts, squared=squared)

    elif sdf_id=='bumpy_torus':
        return bumpy_torus_sdf(selected_pts, squared=squared)

    elif sdf_id=='mobius':
        return mobius_udf_robust(selected_pts, squared=squared)

    elif sdf_id=='mobius_bdry':
        return mobius_bdry_udf(selected_pts, squared=squared)

    
    else:
        raise ValueError("Sorry, couldn't find an sdf with the id", sdf_id)

    

weights_path="sdf_weights/curves/shark.pth"




def udf(selected_pts, sdf_id):
    #unsigned distance function of a shape with given shape id
    return abs(sdf(selected_pts, sdf_id))




def twisted_torus_sdf(selected_pts, squared=False, k=10.0, R=0.35, r=0.13):
    """
    Twisted torus SDF (like opTwist in GLSL)
    
    selected_pts: (B, N, 3)
    squared: return squared distance if True
    k: twist factor (rad per unit Y)
    R: major radius
    r: minor radius
    """
    y = selected_pts[:, :, 1]
    x = selected_pts[:, :, 0]
    z = selected_pts[:, :, 2]

    # ---- Twist the points around Z axis ----
    c = torch.cos(k * y)
    s = torch.sin(k * y)
    x_twist = c * x - s * z
    z_twist = s * x + c * z

    q = torch.stack([x_twist, y, z_twist], dim=-1)

    # ---- Torus SDF in twisted space ----
    # radial distance in XZ plane minus major radius
    xz_dist = torch.sqrt(q[:, :, 1]**2 + q[:, :, 0]**2) - R
    torus_f = xz_dist**2 + q[:, :, 2]**2  # squared distance to minor radius

    if squared:
        sdf = (torus_f.sqrt() - r)**2
    else:
        sdf = torus_f.sqrt() - r

    return sdf




def bumpy_torus_sdf(selected_pts, squared=False, R=0.35, r=0.13, bump_ampl=0.03,
                     bump_freq_theta=8.0, bump_freq_phi=6.0):
    """
    Pseudo-SDF for a torus whose tube radius is modulated by a bump
    pattern defined in proper toroidal coordinates:
      - theta: angle around the major (central) axis
      - phi:   angle around the tube's circular cross-section

    selected_pts: (B, N, 3)
    squared: return squared distance if True
    R: major radius
    r: minor (tube) radius
    bump_ampl: amplitude of the bump pattern
    bump_freq_theta: number of bumps around the major circle
    bump_freq_phi: number of bumps around the tube cross-section
    """
    x, y, z = selected_pts[:, :, 0], selected_pts[:, :, 1], selected_pts[:, :, 2]

    # ---- Angle around the major axis ----
    theta = torch.atan2(z, x)

    # ---- Local tube cross-section coordinates ----
    # radial_offset: signed offset from the central ring, outward in the
    # XZ plane; tube_y: offset along the tube's "up" (Y) axis.
    radial_offset = torch.sqrt(x**2 + z**2) - R
    tube_y = y

    # ---- Angle around the tube cross-section ----
    phi = torch.atan2(tube_y, radial_offset)

    # ---- Distance from the tube's centerline (plain torus SDF) ----
    tube_dist = torch.sqrt(radial_offset**2 + tube_y**2)

    # ---- Bump pattern mapped onto (theta, phi) ----
    bumps = bump_ampl * torch.sin(bump_freq_theta * theta) * torch.sin(bump_freq_phi * phi)

    if squared:
        sdf = (tube_dist - (r + bumps))**2
    else:
        sdf = tube_dist - (r + bumps)

    return sdf

    
def mobius_udf_robust(
    selected_pts,
    R=1.0,
    width=0.25,
    squared=False,
    eps=1e-8
):
    """
    Numerically robust UDF for the exact Möbius band.
    Zero level set is unchanged.
    """

    x = selected_pts[..., 0]
    y = selected_pts[..., 1]
    z = selected_pts[..., 2]

    # --------------------------------------------------
    # Safe angle computation
    # --------------------------------------------------
    # Prevent undefined gradients near origin
    r2 = x*x + y*y
    safe_r = torch.sqrt(r2 + eps)

    cos_t = x / safe_r
    sin_t = y / safe_r
    theta = torch.atan2(sin_t, cos_t)   # equivalent but safer

    phi = 0.5 * theta

    # --------------------------------------------------
    # Center circle
    # --------------------------------------------------
    c = torch.stack([
        R * cos_t,
        R * sin_t,
        torch.zeros_like(theta)
    ], dim=-1)

    # --------------------------------------------------
    # Twisted normal
    # --------------------------------------------------
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    n = torch.stack([
        cos_phi * cos_t,
        cos_phi * sin_t,
        sin_phi
    ], dim=-1)

    # Tangent
    t = torch.stack([
        -sin_t,
         cos_t,
         torch.zeros_like(theta)
    ], dim=-1)

    # --------------------------------------------------
    # Local coordinates
    # --------------------------------------------------
    v = selected_pts - c

    u = (v * n).sum(dim=-1)
    s = (v * t).sum(dim=-1)

    vv = (v * v).sum(dim=-1)

    # Robust orthogonal distance
    h2 = vv - u*u - s*s
    h2 = torch.maximum(h2, torch.zeros_like(h2))

    # --------------------------------------------------
    # Width clamp (geometry exact)
    # --------------------------------------------------
    u_clamped = torch.clamp(u, -width, width)

    dist2 = h2 + (u - u_clamped)**2
    dist2 = torch.maximum(dist2, torch.zeros_like(dist2))

    if squared:
        return dist2
    else:
        return torch.sqrt(dist2 + eps)





def mobius_bdry_udf(
    selected_pts,
    R=1.0,
    width=0.25,
    squared=False,
    eps=1e-8
):
    """
    UDF for the single closed loop forming the boundary of a Möbius band.
    Matches the logic and local frame of mobius_udf_robust.
    """
    x = selected_pts[..., 0]
    y = selected_pts[..., 1]
    z = selected_pts[..., 2]

    # 1. Safe angle and local frame (as in mobius_udf_robust)
    r2 = x*x + y*y
    safe_r = torch.sqrt(r2 + eps)

    cos_t = x / safe_r
    sin_t = y / safe_r
    theta = torch.atan2(sin_t, cos_t)
    phi = 0.5 * theta

    # Center circle path
    c = torch.stack([
        R * cos_t,
        R * sin_t,
        torch.zeros_like(theta)
    ], dim=-1)

    # Twisted normal (u-axis)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    n = torch.stack([
        cos_phi * cos_t,
        cos_phi * sin_t,
        sin_phi
    ], dim=-1)

    # Tangent (s-axis)
    t = torch.stack([
        -sin_t,
        cos_t,
        torch.zeros_like(theta)
    ], dim=-1)

    # 2. Project point into local coordinates
    v = selected_pts - c
    u = (v * n).sum(dim=-1)  # Distance along the "width" axis
    s = (v * t).sum(dim=-1)  # Distance along the tangent (should be near 0)
    
    # Square of total displacement
    vv = (v * v).sum(dim=-1)

    # h2 is the squared distance perpendicular to the (n, t) plane
    h2 = vv - u*u - s*s
    h2 = torch.maximum(h2, torch.zeros_like(h2))

    # 3. Boundary Calculation
    # On the boundary, |u| = width and s = 0 and h = 0.
    # We find the squared distance to the nearest of the two edge points in the cross-section.
    dist_u_plus = (u - width)**2
    dist_u_minus = (u + width)**2
    
    # Distance to the closer edge point in the local u-plane
    u_boundary_dist2 = torch.minimum(dist_u_plus, dist_u_minus)

    # Total squared distance: radial deviation (h2), tangential deviation (s2), 
    # and distance to the edge point (u_boundary_dist2)
    dist2 = h2 + s*s + u_boundary_dist2
    dist2 = torch.maximum(dist2, torch.zeros_like(dist2))

    if squared:
        return dist2
    else:
        return torch.sqrt(dist2 + eps)









'''
def mobius_udf(selected_pts, R=1.0, width=0.2, squared=False):
    """
    Unsigned distance field to a Möbius strip
    matching the parametric construction used in the OBJ generator.

    selected_pts: (..., 3)
    R: center circle radius
    width: half-width of the strip
    squared: return squared distance if True
    """

    # ----------------------------------------
    # Closest angle on center circle
    # ----------------------------------------
    theta = torch.atan2(selected_pts[..., 1], selected_pts[..., 0])
    phi = 0.5 * theta

    # ----------------------------------------
    # Center circle
    # ----------------------------------------
    central_ring = R * torch.stack([
        torch.cos(theta),
        torch.sin(theta),
        torch.zeros_like(theta)
    ], dim=-1)

    # ----------------------------------------
    # Local Möbius frame
    # ----------------------------------------
    n = torch.stack([
        torch.cos(phi) * torch.cos(theta),
        torch.cos(phi) * torch.sin(theta),
        torch.sin(phi)
    ], dim=-1)

    t = torch.stack([
        -torch.sin(theta),
         torch.cos(theta),
         torch.zeros_like(theta)
    ], dim=-1)

    # ----------------------------------------
    # Local coordinates
    # ----------------------------------------
    v = selected_pts - central_ring

    u = (v * n).sum(dim=-1)   # across width
    s = (v * t).sum(dim=-1)   # along circle

    # squared height off the ribbon surface
    h2 = (v * v).sum(dim=-1) - u**2 - s**2
    h2 = torch.clamp(h2, min=0.0)

    # ----------------------------------------
    # Clamp to finite width
    # ----------------------------------------
    u_clamped = torch.clamp(u, -width, width)

    dist2 = h2 + (u - u_clamped)**2

    if (dist2 < 0).any():
        print("⚠️ Negative values detected in Möbius dist2 before sqrt!")

    if not squared:
        return torch.sqrt(dist2)
    else:
        return dist2

'''
















def torus_sdf(selected_pts, squared=False):
    #sdf for a torus with major radius 0.35 and minor radius 0.13, centre origin
    
    theta = torch.atan2(selected_pts[:, :, 2], selected_pts[:, :, 0])

    central_ring = 0.35 * torch.stack([
        torch.cos(theta),
        torch.zeros_like(theta),
        torch.sin(theta)
    ], dim=-1)

    torus_f = (selected_pts - central_ring).pow(2).sum(-1)
    if (torus_f < 0).any():
        print("⚠️ Negative values detected in torus_f before sqrt!")

    check_for_nans(torus_f, "torus_f")

    if squared==False:
        sdf = torus_f.sqrt() - 0.13
        return sdf
    else:
        squared_sdf = ( torus_f.sqrt() - 0.13 )**2
        return squared_sdf
    

def wobbly_torus_sdf(selected_pts, squared=False):
    #sdf for a torus with major radius 0.35 and minor radius 0.13, centre origin
    
    theta = torch.atan2(selected_pts[:, :, 2], selected_pts[:, :, 0])

    central_ring = 0.35 * torch.stack([
        torch.cos(theta),
        torch.zeros_like(theta),
        torch.sin(theta)
    ], dim=-1)

    torus_f = (selected_pts - central_ring).pow(2).sum(-1)
    if (torus_f < 0).any():
        print("⚠️ Negative values detected in torus_f before sqrt!")

    check_for_nans(torus_f, "torus_f")
    x = selected_pts[:,:,0]

    if squared==False:
        sdf = torus_f.sqrt() - 0.13 + 0.02*torch.sin(20*theta)*(x-1)**2
        return sdf
    else:
        squared_sdf = ( torus_f.sqrt() - 0.13 )**2
        return squared_sdf



def sphere_sdf(selected_pts, squared=False):
    #sdf for a sphere with radius 0.5, centre the origin
    
    sphere_f = selected_pts.pow(2).sum(-1)
    if (sphere_f < 0).any():
        print("⚠️ Negative values detected in sphere_f before sqrt!")
        
    check_for_nans(sphere_f, "sphere_f")

    if squared==False:
        sdf = sphere_f.sqrt() - 0.5
        return sdf
    else:
        squared_sdf = ( sphere_f.sqrt() - 0.5 )**2
        return squared_sdf


def urchin_sdf(selected_pts, squared=False):
    x,y,z = selected_pts[...,0], selected_pts[...,1], selected_pts[...,2]
    theta = torch.arctan2(y,x)
    sdf = (x**2 + y**2 + z**2).sqrt() - 0.1* torch.sin(5*theta) * (abs(z**2)-1) - 1.0

    if squared==False:
        return sdf
    else:
        return sdf**2


def rippley_sdf(selected_pts, squared=False):
    x,y,z = selected_pts[...,0], selected_pts[...,1], selected_pts[...,2]
    theta = torch.arctan2(y,x)
    sdf = (x**2 + y**2 + z**2).sqrt() - 0.1* torch.sin(15*theta+5*z) * (abs(z**2)-1) - 1.0

    if squared==False:
        return sdf
    else:
        return sdf**2

        

def cylinder_sdf(selected_pts, R=0.8, h=0.8, squared=False):
    """
    SDF of a cylinder of radius R and half-height h, centered at origin, 
    axis along z.
    
    Args:
        selected_pts: (...,3) tensor of points
        R: radius of the cylinder
        h: half-height of the cylinder
        squared: if True, return squared SDF
    
    Returns:
        sdf: (...,) tensor of signed distances
    """
    x, y, z = selected_pts[..., 0], selected_pts[..., 1], selected_pts[..., 2]
    
    # radial distance from z-axis minus radius
    d_r = torch.sqrt(x**2 + y**2) - R
    # vertical distance from caps
    d_z = torch.abs(z) - h

    # combine distances using the SDF intersection formula
    outside = torch.sqrt(torch.clamp(d_r, min=0)**2 + torch.clamp(d_z, min=0)**2)
    inside  = torch.minimum(torch.maximum(d_r, d_z), torch.zeros_like(d_r))
    sdf = outside + inside

    if squared:
        return sdf**2
    else:
        return sdf



def deepsdf_curve(selected_pts, model=None, squared=False, device=None):
    """
    Compute the SDF for 2D curves using a pretrained network.

    Args:
        selected_pts: torch.Tensor or np.ndarray of shape (N, 2)
        model: pretrained SDF network (required)
        squared: if True, return squared SDF
        device: torch.device (optional, if not None, ensures selected_pts is on this device)

    Returns:
        torch.Tensor of SDF values
    """
    import torch
    import numpy as np

    if model is None:
        raise ValueError("You must provide a pretrained 'model'.")

    # Ensure input is a tensor
    if isinstance(selected_pts, np.ndarray):
        selected_pts = torch.from_numpy(selected_pts).float()

    # Move to correct device
    if device is not None:
        selected_pts = selected_pts.to(device)
        model.to(device)

    model.eval()  # ensure inference mode

    # Compute SDF
    sdf_vals = model(selected_pts)
    if squared:
        sdf_vals = sdf_vals**2

    return sdf_vals





'''
def sdf_curve(selected_pts, weights_path='sdf_weights/curves/shark.pth', squared=False):

    pts = np.zeros((10,2))
    sdf_vals = np.zeros(10)
    
    model, device = sdf_2d.fit_sdf_network(
    pts[:1], sdf_vals[:1],  # dummy tiny dataset, just to build structure
    hidden=128, layers=4, epochs=0  # zero epochs, we’ll load weights
    )
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()  # set to inference mode
    #print("Model weights loaded.")

    if squared==False:
        return model(selected_pts)
        
    else:
        return model(selected_pts)**2
'''




'''
def sdf_curve(selected_pts, weights_path='sdf_weights/curves/shark.pth', squared=False, device=None):
    """
    Compute the SDF for 2D curves using a pretrained network.

    Args:
        selected_pts: torch.Tensor or np.ndarray of shape (N, 2)
        weights_path: path to model weights
        squared: if True, return squared SDF
        device: torch.device (e.g., 'cpu', 'mps', 'cuda'); if None, defaults to 'cpu'

    Returns:
        torch.Tensor of SDF values
    """
    import torch
    import numpy as np

    if device is None:
        device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    # ensure input is a tensor on the correct device
    if isinstance(selected_pts, np.ndarray):
        selected_pts = torch.from_numpy(selected_pts).float()
    selected_pts = selected_pts.to(device)

    # dummy points to build network structure
    pts = torch.zeros((10, 2), device=device)
    sdf_vals = torch.zeros(10, device=device)

    model, _ = sdf_2d.fit_sdf_network(
        pts[:1], sdf_vals[:1],  # dummy tiny dataset
        hidden=128, layers=4, epochs=0  # zero epochs
    )

    # load pretrained weights to the correct device
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # compute SDF
    if not squared:
        return model(selected_pts)
    else:
        return model(selected_pts)**2

'''

def load_deepsdf_curve_model(weights_path='sdf_weights/curves/shark.pth', device=None):
    """
    Initializes and loads a pretrained 2D curve SDF network once.

    Args:
        weights_path: path to the pretrained weights
        device: torch.device ('cpu', 'mps', 'cuda'); required

    Returns:
        model: loaded SDF network ready for inference on the specified device
    """
    import torch
    from . import sdf_2d
    from pathlib import Path

    if device is None:
        raise ValueError('Please specify the device to load the model on.')

    device = torch.device(device) if isinstance(device, str) else device
    print(f"Loading 2D curve SDF model onto {device}...")

    # Build a dummy network to define architecture
    dummy_pts = torch.zeros((1, 2), device=device)
    dummy_sdf = torch.zeros(1, device=device)
    model, _ = sdf_2d.fit_sdf_network(
        dummy_pts, dummy_sdf,
        hidden=128, layers=4, epochs=0  # zero epochs, just to create the network
    )

    # Load pretrained weights onto CPU first
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Curve weights not found: {weights_path}")
    
    state_dict = torch.load(str(weights_path), map_location='cpu')  # always load onto CPU first
    model.load_state_dict(state_dict)  # load weights
    model.to(device)                    # then move the model to the desired device
    model.eval()

    # Freeze parameters (important if using in BCS or other optimization)
    for param in model.parameters():
        param.requires_grad = False

    print(f"2D Curve SDF Model loaded successfully on device: {device}")
    return model






def load_deepsdf_model(ckpt_path=None, device=None, model_variant=None,
                        posenc=True, activation='gelu', v3=False):
    """
    Initializes and loads a DeepSDF-family model once.

    The architecture is chosen directly via `model_variant`, one of the
    keys of DEEPSDF_MODEL_VARIANTS ('posenc', 'no_posenc_gelu',
    'no_posenc_relu', 'v3_gelu', 'v3_soft1', 'v3_soft2').

    `posenc`/`activation`/`v3` are only used to derive a variant when
    `model_variant` is left as None, for older call sites that haven't
    been switched over yet. New code should pass `model_variant`
    directly instead.
    """
    if device is None:
        raise ValueError('Please specify the device to load the model on.')

    if model_variant is None:
        if v3:
            model_variant = 'v3_soft2'
        elif posenc:
            model_variant = 'posenc'
        elif activation == 'relu':
            model_variant = 'no_posenc_relu'
        else:
            model_variant = 'no_posenc_gelu'

    if model_variant not in DEEPSDF_MODEL_VARIANTS:
        raise ValueError(
            f"Unknown model_variant '{model_variant}'. "
            f"Choose one of: {sorted(DEEPSDF_MODEL_VARIANTS)}"
        )

    print(f"Loading deepsdf model onto {device} (variant='{model_variant}')")

    model = DEEPSDF_MODEL_VARIANTS[model_variant]()
    model.to(device)
    
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    
    # Freeze parameters (Crucial for BCS training against this fixed SDF)
    for param in model.parameters():
        param.requires_grad = False
        
    model.eval()
    
    print(f"DeepSDF Model loaded successfully on device: {device}")
    return model




def deepsdf(selected_pts, model, squared=False, transition_width=0.0, safety_radius = 2.0):
    """
    Computes a blended SDF:
    - Uses original DeepSDF inside the unit sphere
    - Gradually transitions to sphere SDF outside the unit sphere
    """

    pts = selected_pts.float()

    # Original DeepSDF
    batch_dict = {"query_surface_points": pts}
    sdf_orig = model(batch_dict)[..., 0]  # [B, N]

    # Radius
    r = torch.linalg.norm(pts, dim=-1)

    # Sphere SDF (unit sphere)
    sdf_sphere = r 

    # Smooth blending weight
    # w = 0 inside, smoothly increases outside
    w = torch.clamp((r - safety_radius) / transition_width, min=0.0, max=1.0)

    # Optional smoother transition (comment out if not needed)
    #w = w * w * (3.0 - 2.0 * w)  # smoothstep

    # Blend SDFs
    sdf = (1.0 - w) * sdf_orig + w * sdf_sphere

    return sdf**2 if squared else sdf



def circle_sdf(x, squared=False):
    # Signed distance from the unit circle centered at origin
    if squared==False:
        return torch.sqrt( (torch.sqrt(x[:, :, 0]**2 + x[:, :, 1]**2) - 1)**2 )
    else:
        return (torch.sqrt(x[:, :, 0]**2 + x[:, :, 1]**2) - 1)**2


def square_sdf(x, squared=False):
    # Signed distance from a square of side length 2 centered at origin
    # The SDF for a square can be approximated as follows:
    q = torch.abs(x) - 1  # 1 = half side length
    outside_dist = torch.sqrt(torch.clamp(q[:, :, 0], min=0)**2 + torch.clamp(q[:, :, 1], min=0)**2)
    inside_dist = torch.clamp(torch.max(q[:, :, 0], q[:, :, 1]), max=0)
    if squared==False:
        return outside_dist + inside_dist
    else:
        return (outside_dist + inside_dist)**2

