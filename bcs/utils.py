''' Functions and classes used by the BCS class (BCS_fast in surface.py), including MLP definition, blend function definitions, custom onering data, and utils like barycentric coords/ distance from a fixed point, is a point on a segment. '''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np

import torch.nn.functional as F


import open3d as o3d
import numpy as np

import tempfile
import os

two_pi = 2*torch.pi




USE_TENSOR = False  # Toggle this to switch APIs

# --- Geometry conversions ---
def Vector3dVector(np_array):
    if USE_TENSOR:
        # Convert numpy array to Open3D tensor
        return o3d.core.Tensor(np_array, dtype=o3d.core.Dtype.Float32)
    else:
        # Legacy API
        return o3d.utility.Vector3dVector(np_array)

def Vector2iVector(np_array):
    if USE_TENSOR:
        return o3d.core.Tensor(np_array, dtype=o3d.core.Dtype.Int32)
    else:
        return o3d.utility.Vector2iVector(np_array)


def Vector3iVector(np_array):
    if USE_TENSOR:
        return o3d.core.Tensor(np_array, dtype=o3d.core.Dtype.Int32)
    else:
        return o3d.utility.Vector3iVector(np_array)

        

# --- I/O functions ---
def read_triangle_mesh(filename):
    if USE_TENSOR:
        return o3d.t.io.read_triangle_mesh(filename)
    else:
        return o3d.io.read_triangle_mesh(filename)

def write_triangle_mesh(filename, mesh):
    if USE_TENSOR:
        return o3d.t.io.write_triangle_mesh(filename, mesh)
    else:
        return o3d.io.write_triangle_mesh(filename, mesh)

# --- Add more wrappers as needed ---




def rotation_matrix_axis_angle(axis, theta):
    """
    Create a 3x3 rotation matrix for rotation of theta radians around an axis.

    Parameters
    ----------
    axis : torch.Tensor, shape (3,)
        Rotation axis (does not need to be unit length)
    theta : torch.Tensor or float
        Rotation angle in radians

    Returns
    -------
    R : torch.Tensor, shape (3, 3)
        Rotation matrix
    """
    axis = axis / torch.linalg.norm(axis)

    x, y, z = axis
    c = torch.cos(theta)
    s = torch.sin(theta)
    C = 1.0 - c

    R = torch.stack([
        torch.stack([c + x*x*C,     x*y*C - z*s, x*z*C + y*s]),
        torch.stack([y*x*C + z*s,   c + y*y*C,   y*z*C - x*s]),
        torch.stack([z*x*C - y*s,   z*y*C + x*s, c + z*z*C])
    ])

    return R





    
#def b(x,v=0.7):

#    result = torch.zeros_like(x)  
#    mask = torch.abs(x) < (1+v)/2
#    temp = (1+v)**2
#    result[mask] = torch.exp(1 + (temp / (4*x[mask]**2 - temp)))
    
#    return result


#def B1(x, v=0.7):
    
#    result = torch.zeros_like(x)
#    mask = torch.abs(x) < (1+v)/2
#    b_res = b(x,v=v)
#    b_minus_res = b(1-abs(x), v=v )

#    result[mask] = ( b_res / (b_res+b_minus_res))[mask]

#    return result




'''
class InvExpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        res = torch.zeros_like(x)
        mask = x > 0
        res[mask] = torch.exp(-1.0 / x[mask])
        return res

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        
        # 1. Handle the division safely to avoid NaNs for x <= 0 
        # (even if they'll be masked out later)
        safe_x = torch.where(x > 0, x, torch.ones_like(x))
        
        # 2. Compute the derivative: exp(-1/x) * (1/x^2)
        # We compute this for all values, but x <= 0 will be suppressed
        fx = torch.exp(-1.0 / safe_x)
        grad_val = fx * (1.0 / (safe_x ** 2))
        
        # 3. Use torch.where instead of in-place mask assignment
        # This is functional and compatible with vmap/jacobian calls
        grad_input = torch.where(x > 0, grad_output * grad_val, torch.zeros_like(x))
        
        return grad_input


def f_optimized(x):
    return InvExpFunction.apply(x)

# g(x) using the optimized f
def g_optimized(x):
    f1 = f_optimized(x)
    f2 = f_optimized(-x + 1)
    return f1 / (f1 + f2)

def B2_inv_exp_optimized(x, v=0.7):
    a = 0.5 - v/2
    return g_optimized((x - a) / (-2 * a + 1))








class InvExpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # We perform the math but suppress zeros to avoid log(0) errors
        mask = x > 0
        res = torch.zeros_like(x)
        res[mask] = torch.exp(-1.0 / x[mask])
        ctx.save_for_backward(x, res)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        x, res = ctx.saved_tensors
        # f'(x) = f(x) * (1 / x^2)
        # We use the 'res' (f(x)) from forward to save a whole exp() calculation
        grad_val = torch.where(x > 0, res / (x**2), torch.zeros_like(x))
        return grad_output * grad_val





def g_poly(x):
    """ 
    A 7th-degree polynomial approximation of the smooth transition.
    It has zero 1st, 2nd, and 3rd derivatives at the endpoints (0 and 1), 
    mimicking the 'flatness' of the exponential version.
    """
    x = x.clamp(0, 1)
    # -20x^7 + 70x^6 - 84x^5 + 35x^4
    return x**4 * (x * (x * (x * -20 + 70) - 84) + 35)

def B2_poly_optimized(x, v=0.7):
    a = 0.5 - v/2
    return g_poly((x - a) / (-2 * a + 1))
'''



import torch

def f(x):
    mask = x > 0
    
    # 1. Create a safe version of x to prevent division-by-zero or NaNs 
    # during the forward and backward pass for elements where x <= 0.
    x_safe = torch.where(mask, x, torch.ones_like(x))
    
    # 2. Compute the exponential globally, but safely.
    y = torch.exp(-1.0 / x_safe)
    
    # 3. Use torch.where to apply the mask element-wise. 
    # This completely avoids memory-heavy masked_scatter operations in autograd.
    return torch.where(mask, y, torch.zeros_like(x))

def g(x):
    # 4. Cache the results of the functions. 
    # This builds the autograd subgraph for f(-x + 1) only once instead of twice.
    fx = f(x)
    f_inv = f(-x + 1.0)
    
    return f_inv / (fx + f_inv)

def B0_linear(x, v=None):
    return (-torch.abs(x) + 1.0).clamp(0.0, 1.0)

def B2_inv_exp(x, v=0.7):
    a = 0.5 - v / 2.0
    return g((x - a) / (-2.0 * a + 1.0))

def B2_inv_exp_even(x, v=0.7):
    return B2_inv_exp(torch.abs(x), v=v)
    
def trig_interp(x):
    x_clamped = x.clamp(0.0, 1.0)
    return torch.cos(x_clamped * (torch.pi / 2.0)) ** 2

def B3_trig(x, v=0.7):
    a = 0.5 - v / 2.0
    return trig_interp((x - a) / (-2.0 * a + 1.0))














    
'''
def f(x):
    result = torch.zeros_like(x)
    mask = x > 0
    result[mask] = torch.exp(-1 / x[mask])

    return result

def g(x):
    return f(-x + 1) / (f(x) + f(-x + 1))

def B0_linear(x, v=None):
    return (-torch.abs(x) + 1.0).clamp(0.0,1.0)

def B2_inv_exp(x, v=0.7):
    a = 0.5 - v/2
    return g( (x-a) / (-2*a + 1) )

def B2_inv_exp_even(x, v=0.7):
    return B2_inv_exp(abs(x), v=v)
    
def trig_interp(x):
    # Clamp x so it stays between 0 and 1
    # If x < 0, it becomes 0 -> cos(0) = 1.0
    # If x > 1, it becomes 1 -> cos(pi/2) = 0.0
    x_clamped = x.clamp(0, 1)
    
    return torch.cos(x_clamped * (torch.pi / 2)) ** 2


def B3_trig(x, v=0.7):
    a = 0.5 - v/2
    return trig_interp( (x-a) / (-2*a + 1) )
'''



def B_djuren( bary, V_face, verts, i, exponent=2, dist_exponent=2):
        # Blend weight of Djuren et al. (2025), "Interpolating splines over triangulated
        # surfaces by blending vertex-centric local geometries". Used by BCS_fast when
        # blend_type='djuren' (an experimental mix of their blending with BCS; see the
        # note in BCS_fast.__init__ in surface.py).

        Xb = torch.stack(bary).transpose(1,0) @ V_face    
    
        eps = 1e-12
        tol = 1e-12
                    
        A = (bary[(i+1)%3] * bary[(i+2)%3])**exponent
        B = (bary[(i+0)%3] * bary[(i+2)%3])**exponent
        C = (bary[(i+0)%3] * bary[(i+1)%3])**exponent
        
        den = A + B + C + eps
        
        W_E_0 = A / den
        W_E_1 = B / den
        W_E_2 = C / den
        
        # exact edge masks
        opp_v_i   = bary[i] < tol
        opp_v_ip1 = bary[ (i+1)%3] < tol
        opp_v_ip2 = bary[ (i+2)%3] < tol
        
        W_E_0[opp_v_i]   = 1.0
        W_E_1[opp_v_i]   = 0.0
        W_E_2[opp_v_i]   = 0.0
        
        W_E_0[opp_v_ip1] = 0.0
        W_E_1[opp_v_ip1] = 1.0
        W_E_2[opp_v_ip1] = 0.0
        
        W_E_0[opp_v_ip2] = 0.0
        W_E_1[opp_v_ip2] = 0.0
        W_E_2[opp_v_ip2] = 1.0
        
        # singular vertex fallback
        singular = (A + B + C) < tol
        W_E_0[singular] = 1/3
        W_E_1[singular] = 1/3
        W_E_2[singular] = 1/3
        
        v_i = torch.tensor(V_face[i, :], dtype=torch.float32)
        v_ip1 = torch.tensor(V_face[(i+1)%3, :], dtype=torch.float32)
        v_ip2 = torch.tensor(V_face[(i+2)%3, :], dtype=torch.float32)


        e_01 = ((Xb - v_ip1)**dist_exponent).sum(-1)  /  ( ((Xb - v_i)**dist_exponent).sum(-1) + ((Xb - v_ip1)**dist_exponent).sum(-1) )  
        e_02 = ((Xb - v_ip2)**dist_exponent).sum(-1)  /  ( ((Xb - v_i)**dist_exponent).sum(-1) + ((Xb - v_ip2)**dist_exponent).sum(-1) )  


        
        return e_01 * W_E_2 + e_02 * W_E_1






def circular_interp(x):
    result = torch.zeros_like(x)

    # Outside region
    result[x < 0] = 1.0
    result[x > 1] = 0.0

    # Left half: [0, 0.5]
    mask_left = (x >= 0) & (x <= 0.5)
    result[mask_left] = torch.sqrt(0.25 - x[mask_left]**2) + 0.5

    # Right half: (0.5, 1]
    mask_right = (x > 0.5) & (x <= 1)
    result[mask_right] = -torch.sqrt(0.25 - (x[mask_right] - 1)**2) + 0.5

    return result



def B4_circular(x, v=0.7):
    a = 0.5 - v/2
    return circular_interp( (x-a) / (-2*a + 1) )

'''
def B5_simple_exp(x, v=0.7):
    #alpha is where it should drop to 1 percent.
    #alpha = (1+v)/2 
    #v is beta in the paper.
    alpha = (1+v)/2
    
    k = -torch.log(torch.tensor(0.01)) / alpha
    return torch.exp(-k*x)
'''

def B5_simple_exp(x, v=0.7):
    
    # decay to 1% at t=1
    k = -torch.log(torch.tensor(0.01))
    return torch.exp(-k * x)
    


def dist(x,i):
    # x: position, i: dist from which of the 3 vertices
    
    vert = [ torch.tensor([0.5, np.sqrt(3)/2]), 
              torch.tensor([0.0, 0.0]), 
             torch.tensor([1.0, 0.0]) ] [i]
        
    return torch.sqrt((x-vert).pow(2).sum(-1))


def G(theta):
    return theta + torch.sin(theta/2)

def H(theta):
    return G(theta - torch.floor(theta/two_pi)*two_pi)

def add_discontinuity(theta, shift=0.5):
    
    return H(theta-shift) + torch.floor((theta-shift)/two_pi)*two_pi + shift
    
import torch
import math

two_pi = 2.0 * math.pi


def add_discontinuities(theta, disc_index=None, valence=None):
    """
    Tile H : [0, 2π] → [0, 2π] using integer discontinuity indices.

    disc_index contains integers in [0, valence-1].
    Sector size = 2π / valence.
    """

    # ensure tensor
    theta = torch.as_tensor(theta, dtype=torch.float32)
    device = theta.device

    # ------------------------------------------------------------
    # Compute jumps (integer sector counts)
    # ------------------------------------------------------------
    if disc_index is None:
        if valence is None:
            raise ValueError("Either disc_index or valence must be provided.")
        jumps = torch.ones(valence, dtype=torch.long, device=device)

    else:
        if valence is None:
            raise ValueError("valence must be provided when using disc_index.")

        disc_index = torch.tensor(disc_index, dtype=torch.long, device=device)
        disc_index = torch.sort(disc_index % valence).values

        # cyclic integer jumps
        jumps = torch.diff(
            torch.cat([disc_index, disc_index[:1] + valence])
        )

    # ------------------------------------------------------------
    # Sector geometry
    # ------------------------------------------------------------
    sector = two_pi / valence                       # scalar
    weights = jumps.float() / valence               # w_k
    boundaries = torch.cat([
        torch.zeros(1, device=device),
        sector * torch.cumsum(jumps, dim=0)
    ])

    # ------------------------------------------------------------
    # Piecewise tiling
    # ------------------------------------------------------------
    out = torch.zeros_like(theta)

    for k, w in enumerate(weights):
        left = boundaries[k]
        right = boundaries[k + 1]

        mask = (theta >= left) & (theta < right)

        # rescale local coordinate to [0, 2π]
        local_theta = (theta[mask] - left) / w
        out[mask] = left + w * H(local_theta)

    # handle endpoint exactly
    out = torch.where(
        theta == two_pi,
        torch.full_like(theta, two_pi),
        out
    )

    return out

#import matplotlib.pyplot as plt
#x = np.linspace(0, two_pi, 1000)
#plt.plot(x, add_discontinuities(x, disc_index=[0,1,3], valence=6).numpy())
#plt.show()


def barycentric_coordinates(x, y):
    # Triangle vertices
    A = torch.tensor([0.0, 0.0])
    B = torch.tensor([1.0, 0.0])
    C = torch.tensor([0.5, np.sqrt(3)/2])

    # Compute area of the triangle
    area = (np.sqrt(3) / 4)

    # Compute lambda values using determinant method
    lambda1 = ((B[1] - C[1]) * (x - C[0]) + (C[0] - B[0]) * (y - C[1])) / (2 * area)
    lambda2 = ((C[1] - A[1]) * (x - A[0]) + (A[0] - C[0]) * (y - A[1])) / (2 * area)
    lambda3 = 1 - lambda1 - lambda2

    return torch.stack([lambda1, lambda2, lambda3]).transpose(1,0)






def compute_onering_data(he_mesh, V_he, vtx_normals=None, sharp_halfedges=[]):
    V = np.asarray(he_mesh.vertices)
    he = he_mesh.half_edges
    
    onerings = []

    for i in range(V.shape[0]):

        v_pos_central = V[i,:]
        
        onering = {'V_indices':[], 'V_pos':[], 'triangles':[], 'valence':0, 'cumulative_angles':[], 'central_pos': v_pos_central, 'sharp_halfedges':[], 'is_boundary': False }
        cur_he = he[V_he[i]]
        
        vtx_0 = cur_he.vertex_indices[1]

        vtx = vtx_0

        idx = 0
        while (vtx!=vtx_0 or idx==0) :
            if not vtx in onering['V_indices']:
                onering['V_indices'].append(vtx)
                
            onering['triangles'].append(cur_he.triangle_index)

            v_pos_cur = V[vtx,:]

            onering['V_pos'].append(v_pos_cur)

            

            temp_he = he[cur_he.next].next
            onering['V_indices'].append(he[temp_he].vertex_indices[0])


            
            new_he_index = he[temp_he].twin
            idx+=1

            if new_he_index == -1:
                onering['is_boundary'] = True
                break
            else:
                
                cur_he = he[ new_he_index ]
                vtx = cur_he.vertex_indices[1]
                
            

            if new_he_index in sharp_halfedges:
                onering['sharp_halfedges'].append(idx)
            
        valence=idx

        if onering['is_boundary']==True:
            while len(onering['triangles']) < 6:
                #onering['triangles'].append(-1)
                onering['triangles'].insert(0, -1)

            #onering['triangles'] = onering['triangles'][::-1]
            onering['triangles'] = onering['triangles'][-1:] + onering['triangles'][:-1]
            onering['triangles'] = onering['triangles'][-1:] + onering['triangles'][:-1]
            onering['triangles'] = onering['triangles'][-1:] + onering['triangles'][:-1]
            
            valence = 6


        onering['angle'] = 2*torch.pi / valence
        #onering['angle_stretch'] = 6 / valence

        onering['V_indices'] = onering['V_indices'][::-1]
        onering['triangles'] = onering['triangles'][::-1]
        onering['triangles'] = onering['triangles'][-1:] + onering['triangles'][:-1]

        #if onering['is_boundary']==True:
            #while len(onering['triangles']) < 6:
            #    #onering['triangles'].append(-1)
            #    onering['triangles'].insert(0, -1)

            #onering['triangles'] = onering['triangles'][::-1]
            #valence = 6

        onering['valence'] = valence

        center_idx = i   
        v_center = V[center_idx]
        
        cumulative_angle = 0.0
        onering['cumulative_angles'] = []
        try:
            for i in range(valence):
                v1_idx = onering['V_indices'][i]
                v2_idx = onering['V_indices'][(i+1) % valence]
            
                v_1 = torch.tensor(V[v1_idx] - v_center)   # edge vector
                v_2 = torch.tensor(V[v2_idx] - v_center)   # next edge
            
                # Normalize (important for stability)
                v_1 = v_1 / (torch.norm(v_1) + 1e-8)
                v_2 = v_2 / (torch.norm(v_2) + 1e-8)
            
                # Dot product → angle
                dot = torch.clamp(torch.dot(v_1, v_2), -1.0, 1.0)
                cur_angle = torch.acos(dot)
            
                cumulative_angle += cur_angle.item()
                onering['cumulative_angles'].append(cumulative_angle)
        except:
            #print('Could not compute cumulative onering angles, maybe because mesh has a boundary. Ignore if this is not needed.')
            onering['cumulative_angles'] = None

        
        onerings.append(onering)

    return onerings




def compute_spline_coeffs(x, y):
    """
    Computes coefficients for a cubic natural spline interpolator.

    Args:
        x (Tensor): 1D points of shape (N,) in strictly increasing order.
        y (Tensor): Values at those points, shape (N,).

    Prints:
        a, b, c, d coefficients for each polynomial piece.

    """

    N = x.shape[0]
    h = x[1:] - x[:-1]  # (N-1,)

    # Prepare tridiagonal coefficients
    l = torch.ones(N, dtype=x.dtype, device=x.device)
    mu = torch.zeros(N-1, dtype=x.dtype, device=x.device)
    z = torch.zeros(N, dtype=x.dtype, device=x.device)

    # Right side of the linear system
    alpha = (3 / h[1:] ) * (y[2:] - y[1:-1]) - (3 / h[:-1]) * (y[1:-1] - y[:-2])

    l[0] = 1
    for i in range(1, N-1):
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i-1] - h[i-1] * z[i-1]) / l[i]

    l[-1] = 1
    z[-1] = 0

    # Backtrack to find coefficients
    c = torch.zeros(N, dtype=x.dtype, device=x.device)
    b = torch.zeros(N-1, dtype=x.dtype, device=x.device)
    d = torch.zeros(N-1, dtype=x.dtype, device=x.device)
    a = y[:-1]

    for j in reversed(range(0, N-1)):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (y[j+1] - y[j]) / h[j] - (c[j+1] + 2 * c[j]) * h[j] / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])

    return a, b, c[:-1], d


def cubic_spline_eval(x_vals, coeffs, x):
    """
    Evaluates cubic polynomial segments at points x.

    Args:
        x_vals (Tensor): 1D array of knot points, shape (N,).
        coeffs (tuple of Tensors): Coefficients (a, b, c, d) for segments.
        x (Tensor): points to evaluate at, 1D.

    """

    a, b, c, d = coeffs

    idx = torch.searchsorted(x_vals, x) - 1
    idx = idx.clamp(0, len(a)-1)

    dx = x - x_vals[idx]
    return a[idx] + b[idx] * dx + c[idx] * dx**2 + d[idx] * dx**3










    
class MLP(nn.Module):
	def __init__(self, layer_sizes):
		super(MLP, self).__init__()
		layers = []
		for i in range(len(layer_sizes) - 1):
            
			layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
			if i < len(layer_sizes) - 2:  # No activation on last layer
				layers.append(nn.Tanh())
		self.network = nn.Sequential(*layers)

        
	
	def forward(self, x):
		return self.network(x)



class ResidualMLP(nn.Module):
    def __init__(self, layer_sizes, init_type="zero"):
        super(ResidualMLP, self).__init__()
        
        assert layer_sizes[0] <= layer_sizes[-1], "Input dim must be ≤ output dim for residual connection"
        self.input_dim = layer_sizes[0]
        self.output_dim = layer_sizes[-1]
        self.init_type = init_type.lower()

        layers = []
        self.linear_layers = []  # To apply custom init later

        for i in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.linear_layers.append(linear)
            layers.append(linear)
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.linear_layers:
            if self.init_type == "kaiming":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="tanh")
                if layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = 1 / fan_in**0.5
                    nn.init.uniform_(layer.bias, -bound, bound)
            elif self.init_type == "xavier":
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif self.init_type == "zero":
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            else:
                raise ValueError(f"Unknown init_type: {self.init_type}")

    def forward(self, x):
        out = self.network(x)
        
        # Pad input if needed
        if self.input_dim < self.output_dim:
            pad_size = self.output_dim - self.input_dim
            x = F.pad(x, (0, pad_size), "constant", 0)

        return out + x


def project_to_torus(coarse):
    if len(coarse.shape)==2:
        coarse = coarse.unsqueeze(0)
        
    theta = torch.atan2(coarse[:,:,2], coarse[:,:,0])

    central_ring = 0.32 * torch.stack([torch.cos(theta), 0*torch.cos(theta), torch.sin(theta)]).transpose(1,0).transpose(2,1)
    #print('cent rting', central_ring.shape)

    vec = coarse - central_ring

    norms = torch.norm(vec, dim=2, keepdim=True)  # shape: (10000, 1)
    unit_vectors = vec / norms
    projection = ( central_ring + 0.1*unit_vectors).squeeze()

    #print('proj', projection.shape)
    normals=unit_vectors.squeeze()

    return projection, normals


def project_to_sphere(coarse):
    if len(coarse.shape)==2:
        coarse = coarse.unsqueeze(0)
        
    theta = torch.atan2(coarse[:,:,2], coarse[:,:,0])

    norms = torch.norm(coarse, dim=2, keepdim=True)  # shape: (10000, 1)
    unit_vectors = coarse / norms
    
    projection = unit_vectors.squeeze()
    normals=projection

    return projection, normals



import re

import re

def stringify_config(cfg, float_precision=3):
    """
    Convert a nested experiment config dict into a readable,
    filesystem-safe string with visual gaps.
    """

    def sanitize(s):
        return re.sub(r"[^a-zA-Z0-9._-]+", "", s)

    def fmt_value(v):
        if isinstance(v, bool):
            return "T" if v else "F"
        if isinstance(v, float):
            return f"{v:.{float_precision}g}"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, str):
            return sanitize(v)
        return sanitize(str(v))

    def stringify_block(block, key_order, abbreviations):
        parts = []
        for k in key_order:
            if k not in block:
                continue
            key = abbreviations.get(k, k)
            parts.append(f"{key}={fmt_value(block[k])}")
        return "__".join(parts)   # 👈 visual gaps

    shape = sanitize(cfg.get("shape-name", "shape"))

    surface = stringify_block(
        cfg["surface-config"],
        key_order=[
            "blend_type",
            "coarse_patches_id",
            "degree",
            "overlap_param",
            "global_scale",
            "local_scales",
        ],
        abbreviations={
            "blend_type": "",
            "coarse_patches_id": "",
            "overlap_param": "overlap",
            "degree": "deg",
            "global_scale": "global-scale",
            "local_scales": "local-scales",
        },
    )

    training = stringify_block(
        cfg["training-config"],
        key_order=[
            "num_samples_per_face",
            "max_epochs",
            "per_face_batch_size",
            "initial_lr",
            "min_lr",
            "normals_reg_coeff",
            "distortion_reg_coeff",
            "area_weighting",
        ],
        abbreviations={
            "sdf_id": "sdf-id",
            "num_samples_per_face": "num-per-face",
            "normals_reg_coeff" : "Nreg",
            "distortion_reg_coeff" : "Dreg", 
            "max_epochs": "maxepochs",
            "per_face_batch_size": "b-size",
            "initial_lr": "initlr",
            "min_lr": "minlr",
            "area_weighting": "AreaW",
        },
    )

    return (
        f"{shape}"
        f"___surf__{surface}"
        f"___train__{training}"
    )


def export_proxy_mesh(obj_path, out_dir, name="proxy"):
    """
    Copy the proxy (coarse control) mesh into out_dir as "<name>.obj".
    """
    import shutil
    from pathlib import Path

    obj_path = Path(obj_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_out = out_dir / f"{name}.obj"
    shutil.copy2(obj_path, obj_out)

    return obj_out

