"""
Djuren et al. baseline.

Our implementation of the quadratic polynomial setting from

    Tobias Djuren et al. (2025), "Interpolating splines over triangulated
    surfaces by blending vertex-centric local geometries".

It is used as a baseline for comparison with BCS and is independent of the
BCS framework in bcs/. Each vertex of the coarse mesh gets a quadratic
polynomial in its one-ring coordinates, without a constant term (so it passes
through the vertex), fitted by least squares to the vertex's one-ring
neighbours. These vertex functions are then blended over each triangle.

The quadratic setting of the paper corresponds to
    Djuren_surface(surface_config={'coarse_patches_id': ..., 'degree': 2}, mesh_res=6)
    .compute_blend_weights(flag='djuren')
    .forward(function_flag='polynomial', onering_coords_flag='djuren', degree=2)
The other options (e.g. 'bary' or 'inv-exp' blending, 'equal' one-ring
coordinates, constant vertex functions) are our own variations for
experimentation and are not part of Djuren et al.'s method.

See djuren_test.ipynb in this folder for usage.
"""

import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
from bcs.utils import *
from bcs.mesh_processing import *
import os
import trimesh
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

import torch.nn.functional as F
import copy

from bcs.differential import * 
diffmod = DifferentialModule()

two_pi = 2*torch.pi
pi = torch.pi

print('Using open3d version at',o3d.__path__)


class Djuren_surface():
    def __init__(self, surface_config = None, device='cpu', mesh_res = 4):

            
        self.load_regular_base_patch(mesh_res)
        print('Initialising Djuren surface on', device)

        self.device = device

        self.coarse_patches_id = surface_config['coarse_patches_id']
        
        filepath = 'data/surfaces/'+self.coarse_patches_id+'.obj'
        coarse_mesh_o3d = o3d.io.read_triangle_mesh(filepath)
        self.coarse_mesh_tm = trimesh.load(filepath)
        
        self.he_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(coarse_mesh_o3d)
        
        #halfedge stuff for the coarse representation
        self.V = torch.tensor(np.asarray(self.he_mesh.vertices), dtype=torch.float32, device=self.device)
        self.n = self.V.shape[0]
        self.F = np.asarray(self.he_mesh.triangles)
        self.he = self.he_mesh.half_edges

        self.num_facepatches = self.F.shape[0]

        #vertex normals of coarse mesh
        triangle_mesh = o3d.geometry.TriangleMesh()
        triangle_mesh.vertices = self.he_mesh.vertices
        triangle_mesh.triangles = self.he_mesh.triangles
        triangle_mesh.compute_vertex_normals()
        self.vertex_normals = np.array(triangle_mesh.vertex_normals)

        # V: (n, 3) torch tensor of vertices
        # F: (num_faces, 3) numpy array of vertex indices
        
        V = self.V                      # (n, 3), torch
        F = torch.as_tensor(self.F, device=self.device)  # (F, 3)

        # a halfedge is assigned to each vertex (onering)
        print('Assigning a halfedge to each vertex.')
        initial_V_he = get_V_he(self.he_mesh)
        
        self.V_he = optimise_V_he(self.he_mesh, initial_V_he) #Important if we want to do boundaries! Allows us to find onerings of boundaries.


        self.base_triangle_verts = torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.5, torch.sqrt(torch.tensor(3.0, dtype=torch.float32, device=self.device)) / 2],
                ],
                dtype=torch.float32,
                device=self.device,
            )

        self.onerings = self.compute_onering_data(self.he_mesh, self.V_he)
        

    
    def load_regular_base_patch(self, mesh_res):
        self.base_facepatch = o3d.io.read_triangle_mesh('data/high_precision_subdiv_triangles/triangle_'+str(mesh_res)+'.obj')
                             
        self.x = torch.tensor( self.base_facepatch.vertices)[:,:2]
        return

    def compute_onering_data(self, he_mesh, V_he, vtx_normals=None, sharp_halfedges=[]):
        V = np.asarray(he_mesh.vertices)
        he = he_mesh.half_edges
        
        onerings = []
    
        for i in range(V.shape[0]):
    
            v_pos_central = V[i,:]
            
            onering = {'V_indices':[], 'V_pos':[], 'triangles':[], 'valence':0, 'cumulative_angles':[], 
                       'angles':[], 'central_pos': v_pos_central, 'sharp_halfedges':[], 'is_boundary': False, 'edge-lengths-tensor':[], 'flat-djuren-angles-tensor':[] }
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

            onering['V_indices'] = onering['V_indices'][::-1]
            onering['triangles'] = onering['triangles'][::-1]
    
            onering['valence'] = valence
    
            center_idx = i   
            v_center = self.V[center_idx]
            
            cumulative_angle = 0.0
            
            
            for i in range(valence):
                v1_idx = onering['V_indices'][i]
                v2_idx = onering['V_indices'][(i+1) % valence]
            
                v_1 = self.V[v1_idx] - v_center   # edge vector
                v_2 = self.V[v2_idx] - v_center   # next edge

                onering['edge-lengths-tensor'].append(torch.norm(v_1))
            
                # Normalize (important for stability)
                v_1 = v_1 / (torch.norm(v_1) + 1e-8)
                v_2 = v_2 / (torch.norm(v_2) + 1e-8)

                
            
                # Dot product → angle
                dot = torch.clamp(torch.dot(v_1, v_2), -1.0, 1.0)
                cur_angle = torch.acos(dot)
            
                cumulative_angle += cur_angle.item()
                onering['cumulative_angles'].append(cumulative_angle)
                onering['angles'].append(cur_angle)

            equal_segment_angle = 2*np.pi / valence
            onering['flat-equal-angles-tensor'] = torch.tensor([equal_segment_angle*j for j in range(valence)])

            onering['flat-djuren-angles-tensor'] = torch.tensor(onering['cumulative_angles']) * (2.0*np.pi) / onering['cumulative_angles'][-1]
            onering['edge-lengths-tensor'] = torch.tensor(onering['edge-lengths-tensor'])

            onering['flat-djuren-angles-tensor'] = onering['flat-djuren-angles-tensor'].roll(shifts=1)


            '''
            import matplotlib.pyplot as plt
            
            # Example for one onering
            djuren_theta = onering['flat-djuren-angles-tensor'].numpy()    # polar angles
            equal_theta = onering['flat-equal-angles-tensor'].numpy()    # polar angles
            djuren_r = onering['edge-lengths-tensor'].numpy()       # magnitudes
            
            # Make a polar plot
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111, polar=True)
            
            # Plot the points as a connected line
            ax.plot(djuren_theta, djuren_r, marker='o', linestyle='-')
            ax.plot(equal_theta, np.ones_like(djuren_r), marker='o', linestyle='-', color='red')
            
            # Optionally: close the loop to show full one-ring
            ax.plot([djuren_theta[-1], djuren_theta[0]], [djuren_r[-1], djuren_r[0]], linestyle='-', color='blue')
            ax.plot([equal_theta[-1], equal_theta[0]], [1.0, 1.0], linestyle='-', color='blue')
            
            ax.set_title("One-Ring: djuren 1ring coords vs equal onering coords")
            plt.show()
            '''
            
            onerings.append(onering)

        return onerings
   

    def compute_blend_weights(self, flag='djuren', djuren_exponent=2, overlap_param=0.7):
    
    
        x = self.x #equilateral

        
        bary = torch.stack([self.bary_weight(x, i) for i in range(3)]).transpose(1, 0)
    
        dtype = bary.dtype
        device = bary.device
    
        blend_weights = torch.zeros(self.F.shape[0], x.shape[0], 3, dtype=dtype, device=device)

        
        self.Xb = torch.zeros(self.F.shape[0], x.shape[0], 3, dtype=dtype, device=device)


        if flag=='djuren':
            for face_index in range(self.F.shape[0]):
                verts = self.F[face_index, :]
                #print(bary.shape, verts.shape)
        
                V_face = torch.tensor(self.V[verts, :], dtype=dtype, device=device)   # (3,3)
                Xb = bary @ V_face          # (num_samples, 3)
                self.Xb[face_index, :, :] = Xb
    
                
        
                for i in range(3):
                
                    onering = self.onerings[verts[i]]
        
                    j = onering['triangles'].index(face_index)
    
                    eps = 1e-9
                    tol = 1e-9
                    
                    A = (bary[:,(i+1)%3] * bary[:,(i+2)%3])**djuren_exponent
                    B = (bary[:,(i+0)%3] * bary[:,(i+2)%3])**djuren_exponent
                    C = (bary[:,(i+0)%3] * bary[:,(i+1)%3])**djuren_exponent
                    
                    den = A + B + C + eps
                    
                    W_E_0 = A / den
                    W_E_1 = B / den
                    W_E_2 = C / den
                    
                    # exact edge masks
                    opp_v_i   = bary[:, i] < tol
                    opp_v_ip1 = bary[:, (i+1)%3] < tol
                    opp_v_ip2 = bary[:, (i+2)%3] < tol
                    
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
                    
                    v_i = torch.tensor(self.V[verts[i], :], dtype=dtype, device=device)
                    v_ip1 = torch.tensor(self.V[verts[(i+1)%3], :], dtype=dtype, device=device)
                    v_ip2 = torch.tensor(self.V[verts[(i+2)%3], :], dtype=dtype, device=device)
    
            
                    e_01 = ((Xb - v_ip1)**2).sum(-1)  /  ( ((Xb - v_i)**2).sum(-1) + ((Xb - v_ip1)**2).sum(-1) )  
                    e_02 = ((Xb - v_ip2)**2).sum(-1)  /  ( ((Xb - v_i)**2).sum(-1) + ((Xb - v_ip2)**2).sum(-1) )  


                    
                    blend_weights[face_index, :, i] = e_01 * W_E_2 + e_02 * W_E_1
        elif flag=='bary':
            for f_i in range(self.F.shape[0]):
                blend_weights[f_i,:,:] = bary

                
        elif flag=='inv-exp':
            for face_index in range(self.F.shape[0]):
                #verts = self.F[face_index, :]
        
                #V_face = torch.tensor(self.V[verts, :], dtype=dtype, device=device)   # (3,3)

                cur_verts = torch.tensor( self.base_triangle_verts  , dtype=dtype)
                Tb = bary @   cur_verts      # (num_samples, 3)
    
                for i in range(3):
                
                    #onering = self.onerings[verts[i]]
        
                    #j = onering['triangles'].index(face_index)
                    
                    v_i = torch.tensor(cur_verts[(i)%3], dtype=dtype, device=device)
                    v_ip1 = torch.tensor(cur_verts[(i+1)%3], dtype=dtype, device=device)
                    v_ip2 = torch.tensor(cur_verts[(i+2)%3], dtype=dtype, device=device)
    
                    
                    #blend_weights[face_index, :, i] = torch.exp(    - 100*((Xb - v_i)**2).sum(-1)      ) / 3.0

                    

                    r_a = torch.sqrt(((Tb - v_i)**2).sum(-1))
                    r_b = torch.sqrt(((Tb - v_ip1)**2).sum(-1))
                    r_c = torch.sqrt(((Tb - v_ip2)**2).sum(-1))

                    
                    blend_weights[face_index, :, i] = B2_inv_exp(r_a, v=overlap_param) / ( B2_inv_exp(r_a, v=overlap_param) + B2_inv_exp(r_b, v=overlap_param) + B2_inv_exp(r_c, v=overlap_param)     )
            
            #blend_weights = blend_weights / blend_weights.sum(-1).unsqueeze(-1)

            
        else:
            raise ValueError(f"flag must be one of ['proper', 'bary'], got {flag!r}")

        self.blend_weights = blend_weights
        return blend_weights # faces x samples x 3
    
        
    def bary_weight(self, x, i):
        v_a = x - self.base_triangle_verts[i].to(x.device)
        v_b = x - self.base_triangle_verts[(i + 1) % 3].to(x.device)
        v_c = x - self.base_triangle_verts[(i + 2) % 3].to(x.device)
    
        # Compute 2D cross product as scalar
        cross = v_b[:,0] * v_c[:,1] - v_b[:,1] * v_c[:,0]
    
        bary_weight = 0.5 * abs(cross.to(x.device)) / (0.5 * 0.5 * np.sqrt(3))
    
        return bary_weight


    def forward(self, function_flag = 'constant', onering_coords_flag='equal', degree=2):

   
        f,p,s,c = self.F.shape[0], 3, self.x.shape[0], 3

        onering_coords = torch.zeros(f,p,s,2)
        angles = torch.zeros(f,p,s)
        angles_eq = torch.zeros(f,p,s)

        for f_i in range(self.F.shape[0]):

            verts = self.V[self.F[f_i],:]
            for i in range(3):

                vtx_idx = self.F[f_i][i]
                
                onering = self.onerings[vtx_idx]
                
                j = onering['triangles'].index(f_i)
                
                cur_onering_coords, r, angle = self.onering_coords_func(self.x, i, j, vtx_idx, verts, flag=onering_coords_flag)
                onering_coords[f_i, i, :,:] =  cur_onering_coords
                angles[f_i, i, :] = angle
                
        self.onering_coords = onering_coords
        self.angles = angles
        
        self.vertex_funcs = torch.zeros(f,p,s,c)

        if function_flag == 'constant':
            print('Using constant vtx functions')
            for f_i in range(self.F.shape[0]):
                for k in range(3):
                    self.vertex_funcs[f_i, k, :,:] = self.V[self.F[f_i][k]]

        elif function_flag == 'polynomial':


            def poly_basis_2d(u, v, degree, include_constant=False):
                terms = []
                
                for total_deg in range(degree + 1):
                    for i in range(total_deg, -1, -1):   # u power
                        j = total_deg - i                # v power
                        
                        if not include_constant and i == 0 and j == 0:
                            continue
                            
                        terms.append((u ** i) * (v ** j))
                
                return torch.stack(terms, dim=1)
                

    
            print('Using polynomial vtx functions')

            poly_weights = []
            for vtx_idx in range(self.V.shape[0]):

                onering = self.onerings[vtx_idx]

                valence = onering['valence']
                
                
                
                #coords = torch.stack( [ self.onering_coords_func(torch.tensor([1.0, 0.0]).unsqueeze(0),  
                #                                                 (list(self.F[onering['triangles'][(j)%valence]]).index(vtx_idx)+1)%3, 
                #                                                 (j)%valence, vtx_idx,   self.V[self.F[onering['triangles'][(j)%valence]]], flag=onering_coords_flag   )[0]
                #                                                       for j in range(valence) ] ).squeeze() #oc coords of onering vertices
                if onering_coords_flag=='djuren':
                    coords = torch.stack(  [ onering['edge-lengths-tensor'] * torch.cos(onering['flat-djuren-angles-tensor']), onering['edge-lengths-tensor'] * torch.sin(onering['flat-djuren-angles-tensor'])] ).transpose(1,0)
                elif onering_coords_flag=='equal':
                    
                    coords = torch.stack(  [  torch.cos(onering['flat-equal-angles-tensor']), torch.sin(onering['flat-equal-angles-tensor'])] ).transpose(1,0)
                    coords = torch.roll(coords, shifts=1, dims=0)
                    
                else:
                    raise ValueError('onering coords type not given or not known')

                

                
                coords = torch.roll(coords, shifts=0, dims=0)


                u, v = coords[:, 0], coords[:, 1]

                
                P = poly_basis_2d(u,v, degree)
                
                Y = torch.stack(  [ self.V[onering['V_indices'][j]] - self.V[vtx_idx,:]   for j in range(valence)  ] ) # vertex coords
                #A = P.transpose(1,0)@P
                #b = P.transpose(1,0)@Y
                A=P
                b=Y
                lstsq_solve = torch.linalg.lstsq(A, b)
                W = lstsq_solve.solution #solve Ax=b
                #print('residuals', ((A @ W - b)**2).sum())
                

                poly_weights.append(W)

            for f_i in range(self.F.shape[0]):
                for k in range(3):
                    vtx_idx = self.F[f_i][k]
                    onering = self.onerings[vtx_idx]

                    j = onering['triangles'].index(f_i)
                    valence = onering['valence']

                    coords = self.onering_coords_func(self.x, k, j, vtx_idx, self.V[self.F[onering['triangles'][(j)%valence]]], flag=onering_coords_flag)[0]
                    u, v = coords[:, 0], coords[:, 1]
                    poly_basis = poly_basis_2d(u,v, degree)
                    
                    self.vertex_funcs[f_i, k, :,:] = poly_basis @ poly_weights[vtx_idx] + self.V[vtx_idx,:]

        output = torch.einsum('fsp,fpsc->fsc', torch.tensor(self.blend_weights, dtype=torch.float32), torch.tensor(self.vertex_funcs, dtype=torch.float32))
        self.output=output
            
        return output

    '''
    def build_output_surface(self, colour_flag = 'angles'):
        
    
        def make_mesh(vertices, faces, colors=None):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            if colors is not None:
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            mesh.compute_vertex_normals()
            return mesh
    
        base_faces = np.asarray(self.base_facepatch.triangles)
        num_samples = self.x.shape[0]
        vertex_offset = 0
    
        vertices_list = []

        colors_list = []

        faces_list = []

        if colour_flag=='angles':
            cmap = plt.cm.hsv
        else:
            cmap = plt.cm.flag
    
        for f_i in range(self.F.shape[0]):
            W = self.blend_weights[f_i]          # (S, 3)
            V_patch = self.output[f_i, :, :]     # (S, 3)
            
    
            vertices_list.append(V_patch)
            
    
            if colour_flag=='angles':
                C_patch = np.einsum(
                    'sp,psc->sc',
                    np.asarray(W),
                    cmap(self.angles[f_i, :, :] / (2 * np.pi))[:, :, :-1]
                )

            elif colour_flag=='blend':
                C_patch = np.zeros_like(W)
                for k in range(3):
                    if self.F[f_i][k]==0:
                        C_patch[:,:] = cmap(W[:,k])[:,:3]
                    
            else:
                C_patch = np.zeros_like(W)
            colors_list.append(torch.tensor(C_patch, dtype=torch.float32))
    
            # connectivity
            faces_list.append(base_faces + vertex_offset)
            vertex_offset += num_samples
    
        # ------------------------------------------------------------
        # Concatenate global arrays
        # ------------------------------------------------------------
        V_all = torch.cat(vertices_list, dim=0).cpu().numpy()
        C_all = torch.cat(colors_list, dim=0).cpu().numpy()
        F_all = np.vstack(faces_list)

        # ============================================================
        # WELD VERTICES (Sewing the disjoint patches into a manifold)
        # ============================================================
        # Set tolerance by defining decimal precision (e.g., 5 means 1e-5 tolerance)
        tolerance_decimals = 3 
        V_rounded = np.round(V_all, tolerance_decimals)
        
        # Find the unique vertices. 
        # return_index keeps the original unrounded float data.
        # return_inverse gives us the map to rewire our faces.
        _, unique_indices, inverse_indices = np.unique(
            V_rounded, axis=0, return_index=True, return_inverse=True
        )
        
        # Rebuild the vertex and color arrays without the duplicates
        V_welded = V_all[unique_indices]
        C_welded = C_all[unique_indices]
        
        # Rewire the face indices to point to the new unified vertices
        F_welded = inverse_indices[F_all]
        # ============================================================

        # ------------------------------------------------------------
        # Build meshes
        # ------------------------------------------------------------
        # Now passing in our welded (manifold) geometry!
        mesh_o3d_coloured = make_mesh(V_welded, F_welded, C_welded)         
        mesh_o3d_plain = make_mesh(V_welded, F_welded, None) 

        unblended_patches = [make_mesh(self.vertex_funcs[f_i, k, :,:], base_faces, None) for f_i in range(self.F.shape[0]) for k in range(3)]

        return mesh_o3d_coloured, mesh_o3d_plain, unblended_patches
    '''



    def build_output_surface(self, colour_flag='angles'):
        
        def make_mesh(vertices, faces, colors=None):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            if colors is not None:
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            mesh.compute_vertex_normals()
            return mesh

        base_faces = np.asarray(self.base_facepatch.triangles)
        
        # ------------------------------------------------------------
        # 1. Setup Integer Barycentrics for Topological Indexing
        # ------------------------------------------------------------
        # Get float barycentric coordinates for the base patch (Shape: S x 3)
        bary = torch.stack([self.bary_weight(self.x, i) for i in range(3)]).transpose(1, 0).cpu().numpy()
        S = bary.shape[0]
        
        # Calculate subdivision level (R) from the number of samples (S)
        # Formula derived from S = (R+1)(R+2)/2
        R = int(np.round((-3 + np.sqrt(9 + 8 * (S - 1))) / 2))
        
        # Convert float barycentrics to exact integer grid coordinates
        int_bary = np.round(bary * R).astype(int)
        
        hash_to_global_id = {}
        global_V = []
        global_C = []
        global_faces = []

        if colour_flag == 'angles':
            cmap = plt.cm.hsv
        else:
            cmap = plt.cm.flag

        # ------------------------------------------------------------
        # 2. Iterate and Hash Geometry
        # ------------------------------------------------------------
        for f_i in range(self.F.shape[0]):
            W = self.blend_weights[f_i].cpu().numpy()       
            V_patch = self.output[f_i, :, :].detach().cpu().numpy()
            
            # Prepare Colors
            if colour_flag == 'angles':
                C_patch = np.einsum(
                    'sp,psc->sc',
                    W,
                    cmap(self.angles[f_i, :, :].cpu().numpy() / (2 * np.pi))[:, :, :-1]
                )
            elif colour_flag == 'blend':
                C_patch = np.zeros_like(W)
                for k in range(3):
                    if self.F[f_i][k] == 0:
                        C_patch[:, :] = cmap(W[:, k])[:, :3]
            else:
                C_patch = np.zeros_like(W)

            coarse_v = self.F[f_i]  # Global IDs for [v0, v1, v2]
            local_to_global = {}
            
            for s in range(S):
                u, v, w = int_bary[s]
                v0, v1, v2 = coarse_v[0], coarse_v[1], coarse_v[2]
                
                # --- TOPOLOGICAL HASHING ---
                # 1. Is it a corner vertex?
                if u == R:   sig = f"V_{v0}"
                elif v == R: sig = f"V_{v1}"
                elif w == R: sig = f"V_{v2}"
                
                # 2. Is it on a shared edge?
                # We identify the edge using min/max global vertex IDs to ensure adjacent faces match perfectly.
                # 'dist' locks the barycentric weight to the minimum vertex ID to ensure orientation independence.
                elif u == 0:
                    min_v, max_v = min(v1, v2), max(v1, v2)
                    dist = v if min_v == v1 else w
                    sig = f"E_{min_v}_{max_v}_{dist}"
                elif v == 0:
                    min_v, max_v = min(v0, v2), max(v0, v2)
                    dist = u if min_v == v0 else w
                    sig = f"E_{min_v}_{max_v}_{dist}"
                elif w == 0:
                    min_v, max_v = min(v0, v1), max(v0, v1)
                    dist = u if min_v == v0 else v
                    sig = f"E_{min_v}_{max_v}_{dist}"
                    
                # 3. It's an interior face point
                else:
                    sig = f"F_{f_i}_{u}_{v}_{w}"
                
                # --- MAP TO GLOBAL ARRAYS ---
                if sig not in hash_to_global_id:
                    hash_to_global_id[sig] = len(global_V)
                    global_V.append(V_patch[s])
                    global_C.append(C_patch[s])
                
                local_to_global[s] = hash_to_global_id[sig]
                
            # --- REWIRE BASE FACES ---
            for bf in base_faces:
                global_faces.append([
                    local_to_global[bf[0]], 
                    local_to_global[bf[1]], 
                    local_to_global[bf[2]]
                ])
                
        # ------------------------------------------------------------
        # 3. Build Final Manifold Meshes
        # ------------------------------------------------------------
        V_welded = np.array(global_V)
        C_welded = np.array(global_C)
        F_welded = np.array(global_faces)
        
        mesh_o3d_coloured = make_mesh(V_welded, F_welded, C_welded)         
        mesh_o3d_plain = make_mesh(V_welded, F_welded, None) 

        unblended_patches = [
            make_mesh(self.vertex_funcs[f_i, k, :, :].detach().cpu().numpy(), base_faces, None) 
            for f_i in range(self.F.shape[0]) for k in range(3)
        ]

        return mesh_o3d_coloured, mesh_o3d_plain, unblended_patches



        

    
    def onering_coords_func(self, x, i, j, vtx_idx, verts=None, flag='equal'):
        """
        Map points x from a triangle patch into one-ring coordinates.
    
        Parameters
        ----------
        x : (N, 2) tensor
            Query points in base triangle coordinates.
        i : int
            Local triangle vertex index (0,1,2) corresponding to anchor vertex.
        j : int
            Triangle/wedge index in the one-ring.
        vtx_idx : int
            Central mesh vertex index whose one-ring is being parameterized.
        verts : (3, 2) tensor
            Triangle vertices in the current patch.
        flag : str
            'equal'  -> use equal-angle version
            'djuren' -> use geometry-aware version
    
        Returns
        -------
        onering_x : (N, 2) tensor
            Coordinates in the unfolded one-ring plane.
        r : (N,) tensor
            Radial distance from anchor vertex.
        theta_out : (N,) tensor
            Angular coordinate used in the returned mapping.
        """
    
        def wrap_angle(theta):
            theta = theta + (theta < 0) * two_pi
            theta = theta - (theta >= two_pi) * two_pi
            return theta
    
        cumulative_angles = self.onerings[vtx_idx]['cumulative_angles']
        angles = self.onerings[vtx_idx]['angles']
        valence = self.onerings[vtx_idx]['valence']
    
        # ------------------------------------------------------------
        # Shared: barycentric interpolation into current triangle
        # ------------------------------------------------------------



        if flag=='djuren':
            bary = torch.stack([self.bary_weight(x, k) for k in range(3)], dim=1)
            Xb = bary.to(dtype=torch.float32, device=x.device) @ verts.to(x.device)
        
            a = verts[(i + 0) % 3].to(x.device)
            b = verts[(i + 1) % 3].to(x.device)
            c = verts[(i + 2) % 3].to(x.device)
        
            # displacement from anchor vertex a
            v = Xb - a
            r = torch.sqrt((v ** 2).sum(-1))
        
            # ============================================================
            # PROPER VERSION
            # ============================================================
            #cur_triangle_angle = 

            
            u = b - a
            w = c - a

            cos_full_angle = (w * u).sum(-1) / (torch.norm(w, dim=-1) * torch.norm(u))
            full_angle = torch.acos(torch.clamp(cos_full_angle, -1.0, 1.0))
            
            #print(cumulative_angles_a)
            

            cos_alpha = (v * w).sum(-1) / (torch.norm(v, dim=-1) * torch.norm(w))
            alpha = torch.acos(torch.clamp(cos_alpha, -1.0, 1.0))

            theta_local = alpha #full_angle / 2.0
            
            
            

            theta_out = (

                wrap_angle (
                (theta_local + sum(angles[:j]))
                * (two_pi / sum(angles))
                )
            )
            
            '''
            if vtx_idx==0:
                print('j:',j)
                print('full angle', full_angle)
                print('all angles', angles)
                print('angles cumsum', np.cumsum(angles))
                print('pre-scale', (wrap_angle(theta_local + sum(angles[:j]))))
                print('theta out', theta_out)
            '''
            

            ####################
            '''
            theta_local = theta.clamp(0.0, max_angle)

            if j == 0:
                start_angle = 0.0
            else:
                start_angle = cumulative_angles_a[j - 1]
                
            
            theta_out = (start_angle + theta_local) * (two_pi / cumulative_angles_a[-1])
            '''

            ########################
        
            onering_x = torch.stack(
                [r * torch.cos(theta_out), r * torch.sin(theta_out)],
                dim=-1
            ).to(dtype=torch.float32, device=self.device)

            onering_x = torch.nan_to_num(onering_x, nan=0.0, posinf=0.0, neginf=0.0)
            # force exactly zero when r = 0
            onering_x[r <= 1e-8] = 0.0

        
            
        elif flag=='equal':
    
            # ============================================================
            # EQUAL-ANGLE VERSION
            # ============================================================
            v_eq = x - self.base_triangle_verts[i].to(x.device)
            r = torch.sqrt((v_eq ** 2).sum(-1))
        
            theta_eq = torch.atan2(v_eq[:, 1], v_eq[:, 0])
            theta_eq = wrap_angle(theta_eq - i * (two_pi / 3))
            theta_eq = two_pi / 6 - theta_eq
            theta_eq = theta_eq.clamp(0.0, two_pi / 6)
        
            global_theta_eq = (theta_eq + (j - 1) * two_pi / 6) * (6 / valence)
            theta_out = wrap_angle(global_theta_eq)
        
            onering_x = torch.stack(
                [r * torch.cos(theta_out), r * torch.sin(theta_out)],
                dim=-1
            ).to(dtype=torch.float32, device=self.device)
    
    
        # ------------------------------------------------------------
        # Select output
        # ------------------------------------------------------------
   
        return onering_x, r, theta_out
        
      


    


        