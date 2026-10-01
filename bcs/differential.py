import torch
from torch.autograd import grad as Grad 
from torch.nn import functional as F
from torch.nn import Module
import numpy as np
import time
import warnings
from torch.func import jacrev, vmap

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

if torch.cuda.is_available():
    try:
        from torch_batch_svd import svd
    except ImportError:
        from torch import svd
else:
    from torch import svd

eps = 1.0e-8


class DifferentialModule(Module):

    # ================================================================== #
    # =================== Compute the gradient ========================= #
    '''
    def gradient(self, out, wrt, allow_unused=False):
        """
        out: [B, N, R] — output tensor
        wrt: [B, N, C] — input tensor we differentiate with respect to
        Returns: Jacobian [B, N, R, C]
        """
        B, N, R = out.shape
        _, _, C = wrt.shape

        # Output: one Jacobian per batch
        gradients = []
        for r in range(R):
            grad_out = torch.ones_like(out[:, :, r])

            grad_r = Grad(
                outputs=out[:, :, r],
                inputs=wrt,
                grad_outputs=grad_out,
                create_graph=True,
                allow_unused=allow_unused,
            )[0]  # Shape: [B, N, C]

            gradients.append(grad_r.unsqueeze(2))  # [B, N, 1, C]

        jacobian = torch.cat(gradients, dim=2)  # [B, N, R, C]
        return jacobian
    '''

    '''
    def gradient(self, out, wrt, allow_unused=False):
        """
        out: [..., N, R]
        wrt: [..., N, C, *extra_dims]
        
        Returns Jacobian: [..., N, R, C, *extra_dims]
        """
    
        # Shapes
        *batch_dims, N, R = out.shape
        wrt_shape = wrt.shape
        _, _, C, *extra = wrt_shape
    
        jac_list = []
    
        for r in range(R):
            # Seed for the r-th output component
            grad_out = torch.zeros_like(out)
            grad_out[..., r] = 1.0
    
            # Gradient wrt wrt
            grad_r = torch.autograd.grad(
                outputs=out,
                inputs=wrt,
                grad_outputs=grad_out,
                create_graph=True,     #UPDATED!
                retain_graph=True,
                allow_unused=allow_unused,
            )[0]      # shape [..., N, C, *extra_dims]
    
            # Insert R dimension AFTER N, BEFORE C
            # Current: [..., N, C, *extra]
            # Want:   [..., N, (R), C, *extra]
            grad_r = grad_r.unsqueeze(-len(extra)-2)
    
            jac_list.append(grad_r)
    
        # Concatenate along the inserted R dimension
        jac = torch.cat(jac_list, dim=-len(extra)-2)
    
        return jac
    '''


    def gradient(self, out, wrt, allow_unused=False):
        """
        out: [B, N, R]  (e.g., [200, 15, 3])
        wrt: [B, N, C]  (e.g., [200, 15, 2])
        """
        *batch_dims, N, R = out.shape
        
        # 1. Create the basis (identity matrix)
        # shape: [R, R]
        identity = torch.eye(R, device=out.device, dtype=out.dtype)
        
        # 2. Reshape identity to [R, 1, 1, R] so it can broadcast over [B, N, R]
        # The first 'R' is the "batch" for is_grads_batched=True
        # The last 'R' is the one that dots against the 'out' dimension
        basis_shape = [R] + [1] * len(batch_dims) + [1, R]
        grad_outputs = identity.view(*basis_shape).expand(R, *batch_dims, N, R)
    
        # 3. Compute the gradient
        # Since grad_outputs is [3, 200, 15, 3], jac will be [3, 200, 15, 2]
        jac = torch.autograd.grad(
            outputs=out,
            inputs=wrt,
            grad_outputs=grad_outputs,
            is_grads_batched=True, 
            create_graph=True,
            retain_graph=True,
            allow_unused=allow_unused
        )[0]
    
        # 4. Reshape to the desired format: [B, N, R, C]
        # Current jac: [R, B, N, C]
        # Target: [B, N, R, C]
        
        # We move the first dimension (R) to the position after N
        # Position index: R(0), B(1), N(2), C(3) -> move 0 to position 2
        jac = jac.movedim(0, len(batch_dims) + 1)
    
        return jac



    
    def backprop(self, out, wrt):
        select = torch.ones(out.size(), dtype=torch.float32).to(out.device)
        J = Grad.grad(outputs=out, inputs=wrt, grad_outputs=select, create_graph=True)[0]
        J = J.view(wrt.size())
        return J

    # ================================================================== #
    # ================ Compute normals ================================= #
    def compute_normals(self, jacobian=None, out=None, wrt=None, return_grad=False):
        if jacobian is None:
            #with torch.autograd.set_detect_anomaly(True):
            jacobian = self.gradient(out=out, wrt=wrt)  # 3x3 matrix
        jacobian3x2 = jacobian

        dx_du = jacobian3x2[:, :, :, 0]
        dx_dv = jacobian3x2[:, :, :, 1]


        
        cross_prod = torch.cross(dx_du, dx_dv, dim=-1)
        norm = cross_prod.norm(p=2, dim=-1, keepdim=True)
        norm = norm.clamp(min=1e-4)

        
        normals = cross_prod / (norm)
        

        if return_grad==True:
            return normals, jacobian
        else:
            return normals

    # ================================================================== #
    # ================ Compute first fundamental form ================== #
    def compute_FFF(self, jacobian=None, normals=None, out=None, wrt=None, return_grad=False):
        if jacobian is None or normals is None:
            normals, jacobian = self.compute_normals(out=out, wrt=wrt, return_grad=True)

        FFF = jacobian.transpose(-1, -2) @ jacobian  # FFF as a batched matrix

        # Extract E, F, G terms
        I_E = FFF[:,:, 0, 0]
        I_F = FFF[:,:, 0, 1]
        I_G = FFF[:,:, 1, 1]

        if return_grad==True:
            return I_E, I_F, I_G, jacobian, normals
        else:
            return I_E, I_F, I_G

    # ================================================================== #
    # ================ Compute second fundamental form ================= #
    def compute_SFF(self, jacobian=None, out=None, wrt=None, return_grad=False, return_normals=False):
        normals, jacobian = self.compute_normals(out=out, wrt=wrt, return_grad=True)

        ra = jacobian[..., 0]
        rb = jacobian[..., 1]

        # Find derivatives of ra, rb with respect to sphere positions
        ra_deriv = self.gradient(out=ra, wrt=wrt)
        rb_deriv = self.gradient(out=rb, wrt=wrt)

        # Do the chain rule to find the derivatives with respect to the R^2 positions
        raa = ra_deriv[:, :, :, 0].squeeze()
        rab = ra_deriv[:, :, :, 1].squeeze()
        rbb = rb_deriv[:, :, :, 1].squeeze()

        #print('raa',raa.shape, 'normals', normals.shape)

        # Compute second fundamental form coefficients
        L = (raa * normals).sum(-1)
        M = (rab * normals).sum(-1)
        N = (rbb * normals).sum(-1)

        if return_grad and return_normals:
            return L, M, N, jacobian, normals
        if return_grad:
            return L, M, N, jacobian
        return L, M, N


    def compute_area_distortion(self, out=None, wrt=None):
        """Compute local area distortion from the sphere to the surface using FFF"""
        E, F, G = self.compute_FFF(out=out, wrt=wrt)
        distortion = torch.sqrt(E * G - F.pow(2))
        return distortion

    def elastic_energy_density(self, out1=None, out2=None, wrt=None, grads1=None, grads2=None, alpha=1.0, beta=1.0, return_total_energy=False):


        if grads1 is None or grads2 is None:
            # Compute gradients in batch
            grads1 = self.gradient(
                out=out1, wrt=wrt
            )
        
            grads2 = self.gradient(
                out=out2, wrt=wrt
            )
        
    
        
        # Construct rotation R (P, N, 3, 2)
        Su, Sv = grads1[..., 0], grads1[..., 1]

        cross_prod = torch.cross(Su, Sv)
        norm = cross_prod.norm(p=2, dim=-1, keepdim=True)
        area_stretch = norm
        normal = cross_prod / norm

        Su = F.normalize(Su, p=2, dim=-1)
        
        
        R = torch.stack([Su, torch.cross(Su, normal)], dim=-1)  # (P, N, 3, 2)
        # orthonormal frame for surface 1 (rest surface)


        inner = ( R.transpose(2,3) @ grads1 )     # (P, N, 2, 2)
        J = torch.matmul(grads2, torch.inverse(inner))   # (P, N, 3, 2)

    
        # Green-Lagrange strain tensor E = 0.5 * (F^T F - I)
        I = torch.eye(2, device=J.device).unsqueeze(0).unsqueeze(0)  # (1,1,2,2)
        S = 0.5 * (torch.matmul(J.transpose(2, 3), J) - I)  # (P, N, 2, 2)


        FFF1 = torch.matmul(grads1, grads1.transpose(2,3))
        FFF2 = torch.matmul(grads2, grads2.transpose(2,3))
        FFF = torch.matmul(J.transpose(2, 3), J)

        
    

        # Strain invariants
        trS = S[:,:,0,0] + S[:,:,1,1]   # trace(S)

        S2 = S @ S
        trS2 = S2[:,:,0,0] + S2[:,:,1,1]
    

        W = alpha * trS2 + 0.5 * beta * (trS ** 2)  # (P, N)

        total_energy = (W*(area_stretch.squeeze())).mean() #maybe times by 4/sqrt3. For now it's just proportional

        return W, total_energy, FFF
        
            

        
            

    # ================================================================== #
    # ================ Compute curvature from FFF, SFF ================= #
    def compute_curvature(self, jacobian=None, out=None, wrt=None, compute_principal_directions=False, prevent_nans=True):
        I_E, I_F, I_G, jacobian, normals = self.compute_FFF(out=out, wrt=wrt, return_grad=True)
        L, M, N = self.compute_SFF(jacobian=jacobian, out=out, wrt=wrt)



        if prevent_nans:
            big = 1e10 * torch.ones_like(L, dtype=L.dtype)
            L = L * (torch.abs(L) < big)
            M = M * (torch.abs(M) < big)
            N = N * (torch.abs(N) < big)

        A = I_E * I_G - I_F.pow(2)
        B = 2 * M * I_F - (I_E * N + I_G * L)
        C = L * N - M.pow(2)

        H = -B / (2.0 * A)
        K = C / A
        H *= -1

        if not compute_principal_directions:
            return H, K

        #### principal curvatures and principal curvature directions ####
        k1 = (-B - torch.sqrt(B**2 - 4*A*C))/(2.0*A)#always smaller principal curvature
        k2 = (-B + torch.sqrt(B**2 - 4*A*C))/(2.0*A)#always bigger principal curvature


        #print('H', H)
        #print('(k1+k2)/2', (k1+k2)/2)

        #print('jac shape', jacobian.shape)
        
        dx_du = jacobian[:, :, :, 0]
        dx_dv = jacobian[:, :, :, 1]
        cross_prod = torch.cross(dx_du, dx_dv, dim=-1)

        #print('jacobian', jacobian.shape)

        #print('e1, e2', dx_du.shape, dx_dv.shape)
        
        x1 = (k1*I_E - L)/(M - k1*I_F)  ###need to adjust for when denominator is zero.
        
        if prevent_nans:
            big = 10000000*torch.ones_like(x1)
            x1 = torch.where(torch.abs(x1)<big, x1, 0.0 )
        

        
        dir1 = ( 1*dx_du + x1.unsqueeze(-1)*dx_dv)

        dir1 = F.normalize(dir1, p=2, dim=-1).squeeze()

        
        x2 = (k2*I_E - L)/(M - k2*I_F)
        
        if prevent_nans:
            big = 10000000*torch.ones_like(x1)
            x2 = torch.where(torch.abs(x2)<big, x2, 0.0 )
            

        dir2 = ( 1*dx_du + x2.unsqueeze(-1)*dx_dv )

        dir2 = F.normalize(dir2, p=2, dim=-1).squeeze()


        print('dir1', (dir1**2).sum(-1)[0,0])
        
        dir3 = cross_prod



        print('dir1shape', dir1.shape)
        print('dir2shape', dir2.shape)


        #print(dir1)
        #print(dir2)

        #print('normals', normals.shape)
        
        
        #cross = torch.cross(e1,e2, dim=-1) #should be able to get rid of this, normals are already computed before FFF
        #normals = F.normalize(cross, p=2, dim=1)
        
        ## fix signs
        #frame = (torch.stack([dir1, dir2, normals.squeeze()])).transpose(0,1)
        #signs = torch.linalg.det(frame)
        #dir1 = (dir1.T*signs.T).T
        
        return H,K, torch.stack([dir1,dir2]), torch.stack([k1, k2]), normals



        
    def laplace_beltrami_divgrad(self, out=None, wrt=None, f=None, f_defined_on_sphere=False):
        """Computes the Laplace-Beltrami operator on a function defined on the surface."""
        normals, _, jacobian3D, _, _ = self.compute_normals(out=out, wrt=wrt, return_grad=True)
        inv_jacobian3D = torch.linalg.inv(jacobian3D)

        if not f_defined_on_sphere:
            df = self.gradient(out=f.unsqueeze(-1), wrt=out).squeeze()
        else:
            df = (self.gradient(out=f.unsqueeze(-1), wrt=wrt).squeeze().unsqueeze(1) @ inv_jacobian3D).squeeze()

        F = df - torch.sum(df * normals, axis=1).unsqueeze(-1) * normals
        dF = self.gradient(out=F, wrt=wrt) @ inv_jacobian3D
        divF = dF[:, 0, 0] + dF[:, 1, 1] + dF[:, 2, 2]

        normals_term = (normals.unsqueeze(1) @ dF @ normals.unsqueeze(2)).squeeze()
        LB_f = divF - normals_term

        return LB_f

    def laplace_beltrami_MC(self, normals, meancurv, f, grad_f=None, hessian_f=None):
        """Computes Laplace-Beltrami using mean curvature."""
        divgrad = hessian_f[:, 0, 0] + hessian_f[:, 1, 1] + hessian_f[:, 2, 2]
        meancurv_term = -2 * meancurv * ((grad_f * normals).sum(-1))
        hessian_term = -1 * (normals.unsqueeze(1) @ hessian_f @ normals.unsqueeze(2)).squeeze()

        LB_f = divgrad + meancurv_term + hessian_term

        return LB_f
