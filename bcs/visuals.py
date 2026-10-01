import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import open3d as o3d
import torch
from .differential import *

from . import colourmappings
import importlib
importlib.reload(colourmappings)
from .colourmappings import *

from .utils import *
from .mesh_processing import *


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from matplotlib import colormaps

from . import implicit_reps
import importlib
importlib.reload(implicit_reps)
from .implicit_reps import *
        
from mpl_toolkits.mplot3d import Axes3D

import subprocess
import warnings









def write_custom_colour_ply_file(tm=None, colouringdict=None, filepath=None):
	print('writing to ', filepath)
	from plyfile import PlyData, PlyElement
	tm.export(filepath)
	p = PlyData.read(filepath)
	v = p.elements[0]
	f = p.elements[1]
	
	# Create the new vertex data with appropriate dtype
	extra_data_specs = [ (colouringname, 'f8', (3,)) for colouringname in colouringdict.keys() ]
	a = np.empty(len(v.data), v.data.dtype.descr + extra_data_specs)
	for name in v.data.dtype.fields:
	    a[name] = v[name]
	
	for colouringname,colouring in colouringdict.items():    
		a[colouringname] = colouring[:,:3]
	
	# Recreate the PlyElement instance
	v = PlyElement.describe(a, 'vertex')
	
	# Recreate the PlyData instance
	p = PlyData([v, f], text=True)
	p.write(filepath)




def show_f_list(f_list):
    #Input: a list of pointcloud dictionaries, possibly with colours.
    #E.g. show_f_list([{'points':np.random.rand(100,1000,3), 'colors':np.random.rand(100,1000,3)}])
    
    pcds = []
    for f in f_list:
        num_triangles = f['points'].shape[0]
        
        # Flatten points and colors to create a single point cloud
        points = f['points'].reshape(-1, 3)  # Shape: e.g. [(80 * 4330), 3]
        
        
        # Create Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = Vector3dVector(points)   # Assign points
        if not (f['colors'] is None):
            colors = f['colors'].reshape(-1,3)
            pcd.colors = Vector3dVector(colors)   # Assign per-point colors
            
        pcds.append(pcd)
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries(pcds, window_name="Colored 3D Point Cloud")





def save_mesh_screenshots(meshes, output_prefix="screenshot", width=3200, height=2400,
                          background_color=[1,1,1], keep_window_open=True ):
        """
        Save screenshots of the given meshes from X, Y, and Z axis views.
        
        Parameters
        ----------
        meshes : list of o3d.geometry.TriangleMesh
            The meshes to display.
        output_prefix : str
            Prefix for screenshot filenames; '_x.png', '_y.png', '_z.png' will be appended.
        width : int
            Window width in pixels.
        height : int
            Window height in pixels.
        background_color : list of float
            RGB background color, each in [0,1].
        keep_window_open : bool
            If True, keeps the Open3D window open at the end.
        use_custom_light : bool
            If True, uses a custom light direction instead of the default camera-based lighting.
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Open3D Screenshot", width=width, height=height)
        
        

        for mesh in meshes:
            mesh.compute_vertex_normals()
            mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.clip(np.asarray(mesh.vertex_colors), 0, 1) )
            
        for mesh in meshes:
            vis.add_geometry(mesh)
        
        opt = vis.get_render_option()
        opt.background_color = np.asarray(background_color)
        opt.mesh_show_wireframe = False
        opt.mesh_show_back_face = False
        opt.point_size = 0.0
        opt.background_color = np.asarray([0.8, 0.8, 0.8])
        
        
        opt.light_on = True
        
        ctr = vis.get_view_control()
        
        # Compute center of all meshes for lookat
        all_points = np.vstack([np.asarray(mesh.vertices) for mesh in meshes])
        center = all_points.mean(axis=0)
        
        # Define camera views: front, side, top
        camera_views = {
            'x': {'front': [1, 0, 0], 'up': [0, 0, 1]},   # looking along +X
            'y': {'front': [0, 1, 0], 'up': [0, 0, 1]},   # looking along +Y
            'z': {'front': [0, 0, 1], 'up': [0, 1, 0]},   # looking along +Z
        }
        
        
        
        for axis, params in camera_views.items():
            ctr.set_lookat(center)
            ctr.set_front(params['front'])
            ctr.set_up(params['up'])
            ctr.set_zoom(0.7)
        
            
            vis.poll_events()
            vis.update_renderer()
        
            # Capture image
            image = vis.capture_screen_float_buffer(do_render=True)
            filename = f"{output_prefix}_{axis}.png"
            o3d.io.write_image(filename, o3d.geometry.Image((255*np.asarray(image)).astype(np.uint8)))
            print(f"✅ Screenshot saved: {filename}")
        
        if keep_window_open:
            vis.run()
        
        vis.destroy_window()





class BCS_visualiser():
    def __init__(self, bcs, mesh_res=6, output_filepath='rendering/rendering_results/',  show_on_coarse=False, blend_type=None, just_onering = False, mobius_example=False, shape_name=None, sdf_id=None, sdf_model=None, transition_width=0.0, error_cmap=None, error_scale_const=None):

        self.mobius_example=mobius_example #flag that is set True if it's the mobius strip example
        self.mesh_res=mesh_res
        self.show_on_coarse=show_on_coarse
        self.blend_type=blend_type
        self.bcs=bcs
        self.output_filepath = output_filepath
        self.shape_name = shape_name

        # Needed only for the 'error' setting (SDF error colouring). Pass
        # sdf_id=training_config['sdf_id'], sdf_model=DEEPSDF_MODEL,
        # transition_width=transition_width to enable it.
        self.sdf_id = sdf_id
        self.sdf_model = sdf_model
        self.transition_width = transition_width

        self.load_regular_base_patches()


        if just_onering==True:

            # If you want indices from a onering
            self.select_patch_indices = torch.tensor(
                self.bcs.onerings[0]['triangles'], dtype=torch.long, device=bcs.device
            )
        else:
            # If you want all triangle indices
            self.select_patch_indices = torch.arange(
                self.bcs.F.shape[0], dtype=torch.long, device=bcs.device
            )

        self.diffmod = DifferentialModule()


        self.meancurv_cmap = plt.get_cmap('seismic')
        self.meancurv_mapping = mapping12

        self.gausscurv_cmap = plt.get_cmap('seismic')
        self.gausscurv_mapping = mapping12

        self.facepatch_cmap = plt.get_cmap('tab20')
        self.angle_cmap = plt.get_cmap('hsv')
        #self.angle_cmap = plt.get_cmap('binary')

        # Colour range for the 'error' setting: colours run from 0 to
        # 1/error_scale_const (raw SDF units). Pass error_cmap (a colormap
        # name or object) / error_scale_const to BCS_visualiser() to
        # experiment per-run without editing this file.
        self.error_cmap = plt.get_cmap(error_cmap) if isinstance(error_cmap, str) else (error_cmap or plt.get_cmap('Reds'))
        self.error_scale_const = error_scale_const if error_scale_const is not None else 50

    
    # ============================================================
    # Mesh / rendering helper utilities (factorised)
    # ============================================================
    
    def _merge_meshes(self, meshes):
        """Merge a list of Open3D TriangleMesh objects into one."""
        merged = o3d.geometry.TriangleMesh()
        for m in meshes:
            merged += m
        return merged
    
    
    def _color_facepatches(self, facepatches, colors):
        """
        Assign per-vertex RGB colours to each facepatch.
    
        colors[i] must be (V_i, 3) for patch i.
        """
        for patch, col in zip(facepatches, colors):
            patch.vertex_colors = o3d.utility.Vector3dVector(col)
    
    
    def _export_colored_ply(self, facepatches, colors, filepath, name):
        """
        Colour patches, merge them, and export a coloured PLY via trimesh.
    
        colors: (num_patches, V_i, 3) array
        """
    
    
        # Merge meshes (Open3D)
        merged = self._merge_meshes(facepatches)
    
        # Trimesh for Hakowan export
        tm = trimesh.Trimesh(
            vertices=np.asarray(merged.vertices),
            faces=np.asarray(merged.triangles),
            process=False,
        )
    
        # Flatten colours to match the merged vertices
        if len(colors.shape)==3:
            flat_colors = np.vstack(colors)
        else:
            flat_colors = colors
    
        write_custom_colour_ply_file(
            tm=tm,
            colouringdict={name: flat_colors},
            filepath=str(filepath),
        )
        
    def _create_mesh(self, vertices: np.ndarray, triangles: np.ndarray) -> o3d.geometry.TriangleMesh:
        """Create an Open3D TriangleMesh with vertex normals computed."""
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = Vector3dVector(vertices)
        mesh.triangles = Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        return mesh

        
    def _create_unblended_meshes(self, unblended_points: torch.Tensor, facepatch_faces: list) -> list:
        """Create Open3D meshes for unblended facepatches."""
        unblended_meshes = []
        num_facepatches, num_copies = unblended_points.shape[:2]
        for i in range(num_facepatches):
            for k in range(num_copies):
                mesh = self._create_mesh(unblended_points[i, k, :, :].detach().cpu().numpy(), facepatch_faces[i])
                unblended_meshes.append(mesh)
        return unblended_meshes


    def _correct_vertex_normals(self, all_normals: torch.Tensor, wrt: torch.Tensor, facepatch_indices: list = None):
        """
        Apply special correction to normals at exactly-on-vertex points.
        
        Args:
            all_normals: Tensor of shape (num_facepatches, n_points_per_patch, 3)
            wrt: Tensor of sampled points (P, N, 2)
            facepatch_indices: Optional list of patches to correct (defaults to all)
        
        Returns:
            Tensor with corrected normals.
        """
        if facepatch_indices is None:
            facepatch_indices = list(range(self.bcs.F.shape[0]))
    
        dtype = all_normals.dtype
        for i in facepatch_indices:
            for t in range(3):
                poly_mat = self.bcs.get_poly_matrix(self.bcs.F[i][t])
    
                # First derivatives
                grad_x = poly_mat[1, :]
                grad_y = poly_mat[2, :]
    
                # Compute normal
                cross_prod = torch.cross(grad_x, grad_y)
                norm = cross_prod.norm(p=2, dim=-1, keepdim=True)
                new_normal = cross_prod / norm
    
                # Mask: points very close to the vertex
                mask = ((wrt - self.bcs.base_triangle_verts[t]).abs() < 0.01).all(dim=-1).squeeze()
    
                # Rotate and assign
                rotated_normal = (new_normal @ self.bcs.rotations[self.bcs.F[i][t]]).to(dtype)
                all_normals[i, mask[i, :], :] = rotated_normal
    
        return all_normals


    def _compute_sdf_error(self, points, batch_size=8192):
        """
        points: (P, N, 3) tensor of surface points. Returns abs(SDF) with
        the same (P, N) shape, evaluated in batches directly on the
        unmerged per-patch points.

        This is the expensive setting: unlike normals/curvature (cheap
        closed-form math + autodiff on the already-fitted polynomial
        surface), 'error' runs a full DeepSDF network forward pass at
        every single output-mesh point -- easily >1M points at high
        mesh_res. Batches stay on-device and only get moved to CPU once
        at the end, so consecutive batches on MPS/CUDA aren't serialised
        by a sync on every iteration.
        """
        shape = points.shape[:2]
        flat = points.reshape(-1, 3)
        num_pts = flat.shape[0]
        batches = []

        with torch.no_grad():
            for start in range(0, num_pts, batch_size):
                end = min(start + batch_size, num_pts)
                batch = flat[start:end].unsqueeze(0)  # (1, B, 3)
                vals = sdf(
                    batch,
                    self.sdf_id,
                    squared=False,
                    model=self.sdf_model,
                    transition_width=self.transition_width,
                )
                batches.append(vals.squeeze(0).reshape(-1).abs())

        abs_sdf = torch.cat(batches).detach().cpu()
        return abs_sdf.reshape(shape)

    def _save_error_colourbar(self, filename):
        """Linear colourbar for the SDF error colouring."""
        vmin, vmax = 0.0, 1.0 / self.error_scale_const
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap=self.error_cmap)
        sm.set_array([])

        fig, ax = plt.subplots(figsize=(2, 5))
        fig.colorbar(sm, cax=ax)
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()


    def show_cbar(self, mapping, cmap, filename="output/colorbar.png", horizontal=False):
    
        values = mapping(np.linspace(0, 1, 256), invert=True)
        sm = ScalarMappable(cmap=cmap)
        sm.set_array(values)

        if horizontal==False:
            fig, ax = plt.subplots(figsize=(1, 5))
            fig.colorbar(sm, cax=ax)
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            # Swap the dimensions for a horizontal layout
            fig, ax = plt.subplots(figsize=(5, 1)) 
            # Tell matplotlib to lay it out horizontally
            fig.colorbar(sm, cax=ax, orientation='horizontal')
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()



            
    def _correct_vertex_curvatures(
        self,
        H: torch.Tensor,
        K: torch.Tensor,
        wrt: torch.Tensor,
        facepatch_indices: list = None
    ):
        """
        Apply special correction to mean (H) and Gaussian (K) curvature
        at points exactly on vertices of the base triangles.
    
        Args:
            H: Tensor of mean curvature (num_facepatches, n_points_per_patch)
            K: Tensor of Gaussian curvature (num_facepatches, n_points_per_patch)
            wrt: Tensor of sampled points (P, N, 2)
            facepatch_indices: Optional list of patches to correct (defaults to all)
    
        Returns:
            Corrected H and K tensors.
        """
        if facepatch_indices is None:
            facepatch_indices = list(range(self.bcs.F.shape[0]))
    
        dtype = H.dtype
        device = H.device
    
        for i in facepatch_indices:
            for t in range(3):
                poly_mat = self.bcs.get_poly_matrix(self.bcs.F[i][t])
    
                # Pad poly_mat to ensure shape is enough for second derivatives
                zeros = torch.zeros(10, poly_mat.shape[1], device=device, dtype=dtype)
                poly_mat = torch.cat([poly_mat, zeros], dim=0)
    
                # First derivatives
                r_u = poly_mat[1, :]
                r_v = poly_mat[2, :]
    
                # Normal
                Nvec = torch.cross(r_u, r_v)
                norm = Nvec.norm(p=2)
                n = Nvec / norm
    
                # Second derivatives
                r_uu = 2 * poly_mat[3, :]
                r_uv = poly_mat[4, :]
                r_vv = 2 * poly_mat[5, :]
    
                # First fundamental form
                E = (r_u * r_u).sum()
                F = (r_u * r_v).sum()
                G = (r_v * r_v).sum()
    
                # Second fundamental form
                L = (r_uu * n).sum()
                M = (r_uv * n).sum()
                Ncoef = (r_vv * n).sum()
    
                # Curvatures
                K_val = (L * Ncoef - M**2) / (E * G - F**2 + 1e-12)
                H_val = (E * Ncoef - 2 * F * M + G * L) / (2 * (E * G - F**2 + 1e-12))
    
                # Mask: points very close to the vertex
                mask = ((wrt - self.bcs.base_triangle_verts[t]).abs() < 0.01).all(dim=-1).squeeze()
    
                # Assign corrected values
                K[i, mask[i, :]] = K_val
                H[i, mask[i, :]] = H_val
    
        return H, K

 

    def load_regular_base_patches(self):
        self.base_facepatches = [ o3d.io.read_triangle_mesh('data/high_precision_subdiv_triangles/triangle_'+str(self.mesh_res)+'.obj')
                             for i in range(self.bcs.num_facepatches) ]

        self.x = self.base_facepatches[0]
        return



    def compute_quantities(self, settings=['default'], degree=None):
        ''' This is the method that does the maths before we display e.g. normals '''

    
        facepatch_faces = [np.asarray(patch.triangles) for patch in self.base_facepatches]
        num_facepatches = len(self.bcs.F)
    
        # build per-patch domain samples and pad to same length (P, N, 2)
        domain_samples_per_facepatch = [torch.tensor(np.asarray(facepatch.vertices)[:, :2]).float().to(self.bcs.device) for facepatch in self.base_facepatches]
        max_pts = max(t.shape[0] for t in domain_samples_per_facepatch)




        '''
        wrt_list = []
        for fp in self.base_facepatches:
            t = torch.tensor(np.asarray(fp.vertices)[:, :2]).float()
            t.requires_grad_(True) 
            wrt_list.append(t)
    
        # 2. Stack them for the single precompute call
        max_pts = max(t.shape[0] for t in wrt_list)
        # Note: Using torch.stack here creates a graph node that links back to the individual leaves in wrt_list
        wrt = torch.stack([torch.nn.functional.pad(t, (0, 0, 0, max_pts - t.shape[0])) for t in wrt_list])
        '''

        wrt = torch.stack([torch.nn.functional.pad(t, (0, 0, 0, max_pts - t.shape[0])) for t in domain_samples_per_facepatch])
        wrt.requires_grad = True





        precomputed_data = self.bcs.precompute_data_from_samples(wrt, detached=False, mobius_example=self.mobius_example, batched_wrt=True) #not computing the gradient because it's not needed (detached=False_

        if degree ==  None:
            degree = self.bcs.degree

        print('degree:', degree)
        output_points, unblended_points = self.bcs.forward( precomputed_data, degree=degree, return_unblended=True )
        
        # output_points shape expected: (P, N, 3)
        # convert to numpy for later meshing
        output_np = output_points.detach().cpu().numpy()
        self.output_np = output_np
    
        n_points_per_patch = wrt.shape[1]
    
        # extra data from precomputed_data dictionary
        self.blends = precomputed_data['blend_weights'].detach().cpu()
        self.angles = precomputed_data['angles'].detach().cpu()
        self.radii = precomputed_data['radii'].detach().cpu()
        
        self.bary = barycentric_coordinates(wrt[:, :, 0], wrt[:, :, 1]).squeeze().detach().cpu().numpy()
        #self.per_vertex_unblended = extra_data['per_vertex_unblended']
    
        verts = self.bcs.V[self.bcs.F].transpose(2, 1)
        coarse_embedding = np.einsum('bij,bjn->bni', verts.cpu(), self.bary)
    
        # containers
        # Use device consistent with output_points
        device = output_points.device
        dtype = output_points.dtype
    
        all_normals = torch.zeros((num_facepatches, n_points_per_patch, 3), dtype=dtype, device=device)
        all_meancurv = torch.zeros((num_facepatches, n_points_per_patch), dtype=dtype, device=device)
        all_gausscurv = torch.zeros((num_facepatches, n_points_per_patch), dtype=dtype, device=device)
        all_directions = torch.zeros((num_facepatches, n_points_per_patch, 3, 2), dtype=dtype, device=device)
        all_princ_curvatures = torch.zeros((num_facepatches, n_points_per_patch, 2), dtype=dtype, device=device)
    
        facepatch_indices = None
        if facepatch_indices is None:
            facepatch_indices = list(range(num_facepatches))
        
        # Normals (batch)
        if ('normals' in settings) or ('abs-normals' in settings):
            print("computing normals...")
            normals_batch = self.diffmod.compute_normals(out=output_points, wrt=wrt)  # expect (P,N,3)
            normals_batch = normals_batch.reshape(num_facepatches, n_points_per_patch, 3).to(dtype)
            all_normals = normals_batch.clone()

            ################# special treatment of exactly-on-vertex points #######################

            all_normals = self._correct_vertex_normals(all_normals, wrt, facepatch_indices)
            
            ############################################################################################
            self.normals = all_normals
    
        # Curvatures (batch)
        if 'curv' in settings:
            print("Attempting batched curvature computation...")
            if 'directions' in settings:
                H, K, directions, princ_curvatures, normals = self.diffmod.compute_curvature(out=output_points, wrt=wrt, compute_principal_directions=True)
                 
            else:
                H, K = self.diffmod.compute_curvature(out=output_points, wrt=wrt, compute_principal_directions=False)
                directions=None
                princ_curvatures=None
    
        if 'curv' in settings:
            H, K = self._correct_vertex_curvatures(H, K, wrt, facepatch_indices)

            self.directions = directions
            self.princ_curvatures = princ_curvatures
            self.meancurv = H.squeeze()
            self.gausscurv = K.squeeze()

        # SDF error (distance of the fitted surface from the true implicit
        # surface).
        if 'error' in settings:
            if self.sdf_id is None:
                warnings.warn(
                    "compute_quantities: 'error' requested but BCS_visualiser "
                    "was not given sdf_id/sdf_model; skipping error computation."
                )
            else:
                print("computing sdf error...")
                self.error = self._compute_sdf_error(output_points)

        #######################################################################################

        # ----------------------------------
        # Build Open3D meshes for facepatches
        # ----------------------------------
        self.facepatches = [self._create_mesh(output_np[i], facepatch_faces[i]) for i in range(num_facepatches)]
        self.coarse_facepatches = [self._create_mesh(coarse_embedding[i], facepatch_faces[i]) for i in range(num_facepatches)]
        self.unblended_facepatches = self._create_unblended_meshes(unblended_points, facepatch_faces)

    '''
    def compute_quantities(self, settings=['default'], degree=None, batch_size=32):
        if degree is None:
            degree = self.bcs.degree
        
        num_facepatches = len(self.bcs.F)
        device = self.bcs.device
        
        # 1. Prepare domain samples (Keep as list of arrays for now to save VRAM)
        domain_samples = [np.asarray(p.vertices)[:, :2] for p in self.base_facepatches]
        max_pts = max(t.shape[0] for t in domain_samples)

        # 2. Pre-allocate NumPy containers on CPU
        self.normals = np.zeros((self.bcs.F.shape[0], max_pts, 3))
        self.meancurv = np.zeros((self.bcs.F.shape[0], max_pts))
        self.gausscurv = np.zeros((self.bcs.F.shape[0], max_pts))
        self.output_np = np.zeros((self.bcs.F.shape[0], max_pts, 3))

    
        # 3. Batch Processing
        for i in range(0, num_facepatches, batch_size):
            indices = list(range(i, min(i + batch_size, num_facepatches)))
            
            # --- THE CRITICAL PART ---
            # Create a fresh batch tensor from the subset of domain samples
            batch_wrt_raw = [torch.tensor(domain_samples[idx]).float() for idx in indices]
            batch_wrt = torch.stack([
                torch.nn.functional.pad(t, (0, 0, 0, max_pts - t.shape[0])) 
                for t in batch_wrt_raw
            ]).to(device)
            
            # Explicitly enable grad on THIS specific batch slice
            batch_wrt.requires_grad_(True)

            print('batch wrt', batch_wrt.shape)
            
            # Now precompute data ONLY for this batch to save massive VRAM
            # This prevents the 'precomputed_data_full' from holding the whole mesh graph
            batch_precomputed = self.bcs.precompute_data_from_samples(batch_wrt, detached=False, selected_face_indices = range(i*batch_size, (i+1)*batch_size))
            
            # 4. Forward Pass
            pts, _ = self.bcs.forward(batch_precomputed, degree=degree, return_unblended=True)


            
            
            # --- Visualisation Block Start ---
            # 1. Flatten the batch: (B, N, 3) -> (B*N, 3)
            print('pts shape', pts.shape)
            flat_pts = pts.detach().cpu().reshape(-1, 3).numpy()
            
            # 2. Create the Open3D PointCloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(flat_pts)
            
            # 3. Optional: Give them a color (light blue) to make them easier to see
            pcd.paint_uniform_color([0.4, 0.7, 1.0])
            
            # 4. Optional: Estimate normals for better shading in the viewer
            pcd.estimate_normals()
            
            print(f"Visualising batch of {flat_pts.shape[0]} points...")
            o3d.visualization.draw_geometries([pcd], 
                                              window_name="Batch Point Visualization",
                                              width=800, height=600)
            # --- Visualisation Block End ---







            
            
            # 5. Compute Curvature/Normals with the active batch graph
            if 'normals' in settings:
                # diffmod will now see the direct relationship between pts and batch_wrt
                batch_norms = self.diffmod.compute_normals(out=pts, wrt=batch_wrt)
                # Correcting normals (make sure this function supports batched input)
                batch_norms = self._correct_vertex_normals(batch_norms, batch_wrt, indices)
                self.normals[indices] = batch_norms.detach().cpu().numpy()
     
            if 'curv' in settings:
                H, K = self.diffmod.compute_curvature(out=pts, wrt=batch_wrt)
                H, K = self._correct_vertex_curvatures(H, K, batch_wrt, indices)
                self.meancurv[indices] = H.detach().cpu().numpy()
                self.gausscurv[indices] = K.detach().cpu().numpy()

            print(indices, pts.shape)
            self.output_np[indices] = pts.detach().cpu().numpy()
    
            # 6. Explicit Cleanup
            del pts, batch_precomputed, batch_wrt
            if device.type == 'mps': torch.mps.empty_cache()
    '''


    def show_bcs(self, settings=['default'], vertex_id=0, patch_id=0, output_dir = None, show_on_coarse=False, save=True,
                 title_prefix=None):
        # title_prefix (e.g. 'BEFORE optimisation') is prepended to every viewer window title
        def _show(geometries, window_name='Open3D'):
            if title_prefix:
                window_name = f'{title_prefix}: {window_name}'
            o3d.visualization.draw_geometries(geometries, window_name=window_name)

        self.show_on_coarse = show_on_coarse

        if output_dir is None:
            shape = self.shape_name or 'unknown_shape'
            if self.shape_name is None:
                warnings.warn(
                    "BCS_visualiser: no shape_name given; writing output under "
                    "'unknown_shape/'. Pass shape_name=... to BCS_visualiser() for a "
                    "sensible per-shape folder."
                )
            output_dir = Path(self.output_filepath) / shape

        output_dir = Path(output_dir)
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)

            proxy_obj_path = Path('data/surfaces') / f"{self.bcs.coarse_patches_id}.obj"
            if proxy_obj_path.exists():
                export_proxy_mesh(proxy_obj_path, output_dir, name="proxy")
            else:
                warnings.warn(f"show_bcs: proxy mesh {proxy_obj_path} not found; skipping proxy export.")

        # No-op helpers so every write/export call below is a one-liner guard-free
        _write_mesh = o3d.io.write_triangle_mesh if save else (lambda *a, **kw: None)
        _write_ply  = self._export_colored_ply   if save else (lambda *a, **kw: None)

        facepatch_sets = [self.facepatches]
        actual_vertex_positions = self.bcs.actual_vertex_positions()
        radius = 0.01
        
        actual_vtx_blobs = [
            o3d.geometry.TriangleMesh.create_sphere(radius)
            .translate(pos.detach().cpu().numpy())
            .paint_uniform_color([1, 0, 0])
            for pos in actual_vertex_positions
        ]

        vtx_blobs = [
            o3d.geometry.TriangleMesh.create_sphere(radius)
            .translate(pos.detach().cpu().numpy())
            .paint_uniform_color([0, 0, 1])
            for pos in self.bcs.V
        ]

        vtx_blobs=[]
        actual_vtx_blobs=[]


        
        if self.show_on_coarse == True:
            facepatch_sets.append(self.coarse_facepatches)

        for facepatches in facepatch_sets:

            
            ############## DEFAULT VIEW (patches) ####################
            if 'default' in settings:
                for i,facepatch in enumerate(facepatches):
                    facepatch.paint_uniform_color(self.facepatch_cmap(i%20)[:-1])
        
                name = 'Output Mesh Resolution ' +str(self.mesh_res)
                _show( facepatches + vtx_blobs + actual_vtx_blobs, window_name = name )
                combined_mesh = self._merge_meshes(facepatches)
                _write_mesh( str( output_dir / "patch-coloured.obj" ), combined_mesh)
                
            # ------------------------------------------------------------
            # Uniform colour
            # ------------------------------------------------------------
            if 'one-colour' in settings:
                for facepatch in facepatches:
                    facepatch.paint_uniform_color([0.5, 0.5, 1.0])
            
                name = f"Output Mesh Resolution {self.mesh_res}"
                _show( facepatches + vtx_blobs + actual_vtx_blobs, window_name = name )
                combined_mesh = self._merge_meshes(facepatches)
                _write_mesh( str(output_dir / "uniform-coloured.obj"), combined_mesh)
                _write_mesh( str(output_dir / "uniform-coloured.ply"), combined_mesh)

            if 'unblended' in settings:
                for facepatch in facepatches:
                    facepatch.paint_uniform_color([0.5, 0.5, 1.0])

                _show(self.unblended_facepatches+vtx_blobs+actual_vtx_blobs, window_name='Unblended')
                combined_unblended_mesh = self._merge_meshes(self.unblended_facepatches)
                _write_mesh( str(output_dir / "uniform-coloured-unblended.obj"), combined_unblended_mesh)
                _write_mesh( str(output_dir / "unblended.ply"), combined_unblended_mesh)
        
    
            # ------------------------------------------------------------
            # Normals
            # ------------------------------------------------------------
            if 'normals' in settings:
            
                # Per-vertex RGB in [0,1]
                colors = (
                    0.5 * self.normals.detach().cpu().numpy() + 0.5
                ).clip(0.0, 1.0)
            
                # Colour Open3D patches
                self._color_facepatches(facepatches, colors)
            
                # Visualise
                _show( facepatches + vtx_blobs + actual_vtx_blobs, window_name="Normals" )
                combined_mesh = self._merge_meshes(facepatches)
            
                # OBJ export
                _write_mesh( str(output_dir / "normals-coloured.obj"), combined_mesh )
            
                # Hakowan-compatible PLY export
                _write_ply( facepatches, colors, output_dir / "normals.ply", name="normals_colours" )


            if 'abs-normals' in settings:
            
                # Per-vertex RGB in [0,1]
                colors = (
                   np.abs( self.normals.detach().cpu().numpy() )
                ).clip(0.0, 1.0)
            
                # Colour Open3D patches
                self._color_facepatches(facepatches, colors)
            
                # Visualise
                _show( facepatches + vtx_blobs + actual_vtx_blobs, window_name="Abs Normals" )
                #Export obj
                _write_mesh( str(output_dir / "abs-normals-coloured.obj"), self._merge_meshes(facepatches) )
            
                # Hakowan-compatible PLY export
                _write_ply( facepatches, colors, output_dir / "abs-normals.ply", name="abs-normals_colours" )


            ################ CURVATURE VIEWS ###################
            if 'curv' in settings:
                # --- Process Mean Curvature ---
                all_mean_colors = []
                for i, facepatch in enumerate(facepatches):
                    # Compute colors for this patch
                    colors = self.meancurv_cmap(self.meancurv_mapping(self.meancurv[i,:].detach().cpu().numpy()))[:, :-1]
                    facepatch.vertex_colors = Vector3dVector(colors)
                    all_mean_colors.append(colors)

                
                
                # Flatten list of arrays into one big array for export
                combined_mean_colors = np.concatenate(all_mean_colors, axis=0)
                # Update the colours of the mesh that gets exported
                combined_mesh.vertex_colors = o3d.utility.Vector3dVector(combined_mean_colors)
                
                _show(facepatches + vtx_blobs + actual_vtx_blobs, window_name='Mean Curvature')
                
                # OBJ export (Note: Ensure combined_mesh vertex colors are updated if using this)
                _write_mesh(str(output_dir / "meancurv-coloured.obj"), combined_mesh)
                
                # Export using the FULL color array
                _write_ply(facepatches, combined_mean_colors, output_dir / "meancurv.ply", name="meancurv_colours")
            
                # --- Process Gauss Curvature ---
                all_gauss_colors = []
                for i, facepatch in enumerate(facepatches):
                    colors = self.gausscurv_cmap(self.gausscurv_mapping(self.gausscurv[i,:].detach().cpu().numpy()))[:, :-1]
                    facepatch.vertex_colors = Vector3dVector(colors)
                    all_gauss_colors.append(colors)
                    
                combined_gauss_colors = np.concatenate(all_gauss_colors, axis=0)
                # Update the colours of the mesh that gets exported
                combined_mesh.vertex_colors = o3d.utility.Vector3dVector(combined_gauss_colors)
            
                _show(facepatches + vtx_blobs + actual_vtx_blobs, window_name='Gauss Curvature')
            
                _write_mesh(str(output_dir / "gausscurv-coloured.obj"), combined_mesh)
                
                # Export using the FULL color array
                _write_ply(facepatches, combined_gauss_colors, output_dir / "gausscurv.ply", name="gausscurv_colours")
    
                    
    
                    
                if save:
                    self.show_cbar(self.gausscurv_mapping, self.gausscurv_cmap, filename=str(output_dir / "gausscurv_cbar.png"), horizontal=True)
                    self.show_cbar(self.meancurv_mapping, self.meancurv_cmap, filename=str(output_dir / "meancurv_cbar.png"), horizontal=True)

            ################ SDF ERROR ###################
            if 'error' in settings:
                if not hasattr(self, 'error'):
                    warnings.warn(
                        "show_bcs: 'error' requested but compute_quantities() wasn't "
                        "called with 'error' in settings (or sdf_id/sdf_model wasn't "
                        "given to BCS_visualiser()); skipping error export."
                    )
                else:
                    scale = lambda x: x * self.error_scale_const

                    all_error_colors = []
                    for i, facepatch in enumerate(facepatches):
                        colors = self.error_cmap(scale(self.error[i, :].detach().cpu().numpy()))[:, :-1]
                        facepatch.vertex_colors = Vector3dVector(colors)
                        all_error_colors.append(colors)

                    combined_error_colors = np.concatenate(all_error_colors, axis=0)
                    combined_error_mesh = self._merge_meshes(facepatches)
                    combined_error_mesh.vertex_colors = o3d.utility.Vector3dVector(combined_error_colors)

                    _show(facepatches + vtx_blobs + actual_vtx_blobs, window_name='SDF Error')

                    _write_mesh(str(output_dir / "error-coloured.obj"), combined_error_mesh)
                    _write_ply(facepatches, combined_error_colors, output_dir / "error.ply", name="error_colours")

                    if save:
                        self._save_error_colourbar(str(output_dir / "error_cbar.png"))

            if 'blend' in settings:

                colors = np.zeros((self.bcs.F.shape[0], self.blends.shape[1], 3))
                verts = self.bcs.F[patch_id,:]

                for f in range(self.bcs.F.shape[0]):
                    for i, v in enumerate(self.bcs.F[f]):
                        if verts[0] == v :
                            colors[f, :, 0] += self.blends[f, :, i].detach().cpu().numpy()

                        if verts[1] == v :
                            colors[f, :, 1] += self.blends[f, :, i].detach().cpu().numpy()
                        if verts[2] == v :
                            colors[f, :, 2] += self.blends[f, :, i].detach().cpu().numpy()
                        
                self._color_facepatches(facepatches, colors)
                _show( facepatches, window_name='Blending Visualisation')
                _write_ply(facepatches, colors, output_dir / "blending_visual.ply", name="blend" )
                    
    
            if 'bary' in settings:
                colors = np.zeros((self.bcs.F.shape[0], self.blends.shape[1], 3))
                for i, facepatch in enumerate(facepatches):
                    cur_colors = self.bary[i,:,:].transpose(1,0)
                    colors[i,:,:] = cur_colors

                self._color_facepatches(facepatches, colors)
                _show( facepatches, window_name='Barycentric')
                _write_ply(facepatches, colors, output_dir / "barycentric.ply", name="bary" )


            if 'angle' in settings:

                angle_colors = self.angle_cmap(( 1*self.angles/(2*np.pi))  )[:,:,:,:-1]
                
                blended_angle_colors = (self.blends.unsqueeze(-1) * angle_colors ).sum(-2)

                circle_angle_colors = ( ( self.radii.unsqueeze(-1)<0.5) * angle_colors ).sum(-2)

                self._color_facepatches( facepatches, blended_angle_colors  )
                _show( facepatches, window_name='Angle')

                combined_mesh = self._merge_meshes(facepatches)
                combined_coarse_mesh = self._merge_meshes(self.coarse_facepatches)
                angle_colours = np.asarray( combined_mesh.vertex_colors )

                _write_ply( facepatches, blended_angle_colors, output_dir / "angle-colors.ply", name="angle-colours" )
                _write_ply( self.coarse_facepatches, blended_angle_colors, output_dir / "coarse-angle-colors.ply", name="angle-colours" )


            if 'param' in settings:


                ### Make a striped colour pattern ###
                
                def compute_procedural_param_colors(face_coords_list):
                    param_colors_list = []
                    stripe_cmap = colormaps["ocean"]
            
                    for coords in face_coords_list:
                        phase = ( 0.5*np.tanh(np.sin(coords[:,2]*100)) + 0.5 ) * 0.8
                        colors = stripe_cmap(phase)[:,:3]
                        param_colors_list.append(colors)
                    return param_colors_list


                # --- Use coarse facepatch vertices to define coloring ---
                coarse_coords_list = [np.asarray(coarse.vertices) for coarse in self.coarse_facepatches]
                param_colors_list = compute_procedural_param_colors(coarse_coords_list)

                all_coarse_colors = param_colors_list
                all_unblended_colors = []
                
                # --- Apply coarse coloring to all facepatches ---
                for i, facepatch in enumerate(facepatches):
                    # Map coarse coloring to fine patch vertices via barycentric embedding
                    coarse_colors = param_colors_list[i]  # shape: (N_coarse, 3)
                    facepatch.vertex_colors = Vector3dVector(coarse_colors)

                    self.coarse_facepatches[i].vertex_colors = Vector3dVector(coarse_colors)
                    self.unblended_facepatches[3*i].vertex_colors = Vector3dVector(coarse_colors)
                    self.unblended_facepatches[3*i + 1].vertex_colors = Vector3dVector(coarse_colors)
                    self.unblended_facepatches[3*i + 2].vertex_colors = Vector3dVector(coarse_colors)

                    all_unblended_colors.append(np.stack([coarse_colors, coarse_colors, coarse_colors]) )
            
                all_coarse_colors = np.stack(all_coarse_colors)
                all_unblended_colors = np.stack(all_unblended_colors).reshape( all_coarse_colors.shape[0]*3, all_coarse_colors.shape[1], all_coarse_colors.shape[2])
                
                #Display                    
                _show(self.unblended_facepatches+vtx_blobs+actual_vtx_blobs, window_name='Unblended Param')
                _show(self.coarse_facepatches+vtx_blobs+actual_vtx_blobs, window_name='Coarse Param')
                _show(facepatches+vtx_blobs+actual_vtx_blobs, window_name='Blended Param')

                #Export ply files
                _write_ply(facepatches, all_coarse_colors, output_dir / "blended_param.ply", name="param" )
                _write_ply(self.coarse_facepatches, all_coarse_colors, output_dir / "coarse_param.ply", name="param" )
                _write_ply(self.unblended_facepatches, all_unblended_colors, output_dir / "unblended_param.ply", name="param" )



            if 'directions' in settings:
                print('Directions shape:', self.directions.shape)
            
                # Paint all patches uniformly white
                for facepatch in facepatches:
                    facepatch.paint_uniform_color((1.0, 1.0, 1.0))
            
                # Create LineSets for both min and max curvature directions
                mincurv_arrows = []
                maxcurv_arrows = []
            
                # Iterate through each patch and its corresponding direction set
                for facepatch_idx, patch in enumerate(facepatches):
                    verts = np.asarray(facepatch.vertices)
                    n = verts.shape[0]
            
                    for direction_idx, color in enumerate([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]):  # Red for min, blue for max
                        directions = self.directions[direction_idx, facepatch_idx, :, :].squeeze().detach().cpu().numpy()

                        #print((directions*directions).sum(-1))
                        directions = np.nan_to_num(directions, nan=0.0)  # Clean up NaNs
            
                        # Scale for visibility
                        scale_factor = 0.01
                        endpoints = np.concatenate((verts, verts + scale_factor * directions))
            
                        # Create LineSet
                        arrow_set = o3d.geometry.LineSet()
                        arrow_set.points = Vector3dVector(endpoints)
                        arrow_set.lines = Vector2iVector(np.array([[i, i + n] for i in range(n)]))
                        arrow_set.paint_uniform_color(color)

                        if direction_idx==0:
                            mincurv_arrows.append(arrow_set)
                        else:
                            maxcurv_arrows.append(arrow_set)
            
                # Visualize everything at once
                _show( mincurv_arrows + maxcurv_arrows, window_name='Curvature Directions')
                
                _show(facepatches + mincurv_arrows, window_name='Minimum Curvature Directions')
                _show(facepatches + maxcurv_arrows, window_name='Maximum Curvature Directions')
                _show(facepatches + mincurv_arrows + maxcurv_arrows, window_name='Curvature Directions')
                    


    

class Elastic_energy_visualiser():
    def __init__(self, bcs1, bcs2, mesh_res=6, output_filepath='output/', test_flag=None, use_original_mesh=False, show_on_coarse=False, blend_type='default', just_onering=False,
                alpha=1.0, beta=1.0, energy_visual_scale=1.0 ):

        self.mesh_res=mesh_res
        self.use_original_mesh=use_original_mesh
        self.show_on_coarse=show_on_coarse
        self.blend_type=blend_type
        
        self.bcs1=bcs1
        self.bcs2=bcs2
        self.output_filepath = output_filepath
        self.alpha=alpha
        self.beta=beta
        

        self.load_regular_base_facepatches()

        if just_onering==True:

            # If you want indices from a onering
            self.select_patch_indices = torch.tensor(
                self.bcs2.onerings[0]['triangles'], dtype=torch.float32, device=self.bcs2.device
            )
        else:
            # If you want all triangle indices
            self.select_patch_indices = torch.arange(
                self.bcs2.F.shape[0], dtype=torch.float32, device=self.bcs2.device
            )
            
        self.test_flag=test_flag

        self.diffmod = DifferentialModule()


        self.elastic_energy_cmap = plt.get_cmap('Oranges')
        self.elastic_energy_mapping = lambda x: linear(x, factor=energy_visual_scale) 

        self.area_dist_cmap = plt.get_cmap('BrBG')
        self.area_dist_mapping = lambda x: 0.4*np.log(x) + 0.5
        #self.area_dist_mapping = lambda x: linear(x, factor=1e2) 

        


        
        

    def load_regular_base_facepatches(self):
        self.base_facepatches = [ o3d.io.read_triangle_mesh('data/subdiv_triangles/triangle_'+str(self.mesh_res)+'.obj')
                             for i in range(self.bcs1.num_facepatches) ]
        return

        


    def compute_quantities(self, config_filepath=None):
        # Get faces for each patch
        patch_faces = [np.asarray(patch.triangles) for patch in self.base_facepatches]
        num_facepatches = len(self.bcs1.F)
    
    
        # build per-patch domain samples and pad to same length (P, N, 2)
        domain_samples_per_facepatch = [torch.tensor(np.asarray(facepatch.vertices)[:, :2]).float().to(self.bcs1.device) for facepatch in self.base_facepatches]
        max_pts = max(t.shape[0] for t in domain_samples_per_facepatch)
        wrt = torch.stack([torch.nn.functional.pad(t, (0, 0, 0, max_pts - t.shape[0])) for t in domain_samples_per_facepatch])

        wrt.requires_grad = True
        precomputed_data1 = self.bcs1.precompute_data_from_samples(wrt, detached=False)
        precomputed_data2 = self.bcs2.precompute_data_from_samples(wrt, detached=False)

    
        # Forward pass through both networks
        output_points1, _ = self.bcs1.forward(
            precomputed_data1, degree=self.bcs1.degree, return_unblended=True
        )
        output_points2, _ = self.bcs2.forward(
            precomputed_data2, degree=self.bcs2.degree, return_unblended=True
        )
    
        
        
    
    
        W, total_energy, FFF = self.diffmod.elastic_energy_density(out1=output_points1, out2=output_points2, wrt=wrt, alpha=self.alpha, beta=self.beta)
    
        # Store per-patch elastic energies
        self.elastic_energy = W

        print('W', W)
        
        self.area_dist = torch.sqrt ( FFF[:,:,0,0] * FFF[:,:,1,1] - FFF[:,:,1,0]**2 )

        
        # Rebuild meshes for visualization and compute area ratios
        deformed_patches = []
        rest_patches = []
        triangle_area_ratios = []
        
        vertex_area_ratios = []

        output_2_np = output_points2.detach().cpu().numpy()
        output_1_np = output_points1.detach().cpu().numpy()
        
        for i in range(num_facepatches):
            # --- Rest patch ---
            rest_V = output_1_np[i, :, :]
            rest_patch = o3d.geometry.TriangleMesh()
            rest_patch.vertices = Vector3dVector(rest_V)
            rest_patch.triangles = Vector3iVector(patch_faces[i])
            rest_patch.compute_vertex_normals()
            rest_patches.append(rest_patch)
        
            # --- Deformed patch ---
            def_V = output_2_np[i, :, :]
            deformed_patch = o3d.geometry.TriangleMesh()
            deformed_patch.vertices = Vector3dVector(def_V)
            deformed_patch.triangles = Vector3iVector(patch_faces[i])
            deformed_patch.compute_vertex_normals()
            deformed_patches.append(deformed_patch)
        
            # --- Vectorised area computation ---
            tris = np.asarray(patch_faces[i])  # (F, 3)
        
            # Gather triangle vertices
            v0r, v1r, v2r = rest_V[tris[:, 0]], rest_V[tris[:, 1]], rest_V[tris[:, 2]]
            v0d, v1d, v2d = def_V[tris[:, 0]], def_V[tris[:, 1]], def_V[tris[:, 2]]
        
            # Cross products and norms → triangle areas
            rest_areas = 0.5 * np.linalg.norm(np.cross(v1r - v0r, v2r - v0r), axis=1)  # (F,)
            def_areas  = 0.5 * np.linalg.norm(np.cross(v1d - v0d, v2d - v0d), axis=1)  # (F,)
        
            # --- Distribute triangle areas to vertices (1/3 each) ---
            n_vertices = rest_V.shape[0]
            rest_vertex_areas = np.zeros(n_vertices)
            def_vertex_areas  = np.zeros(n_vertices)
        
            np.add.at(rest_vertex_areas, tris.ravel(), np.repeat(rest_areas / 3.0, 3))
            np.add.at(def_vertex_areas,  tris.ravel(), np.repeat(def_areas / 3.0, 3))
        
            # --- Compute per-vertex ratios ---
            ratios = np.divide(def_vertex_areas, rest_vertex_areas,
                               out=np.zeros_like(def_vertex_areas),
                               where=rest_vertex_areas > 1e-12)
            vertex_area_ratios.append(ratios)
        
        self.deformed_patches = deformed_patches
        self.rest_patches = rest_patches
        self.vertex_area_ratios = np.stack(vertex_area_ratios)   # list of (Vi,) arrays
        

    def show_cbar(self, mapping, cmap, filename="output/colorbar.png", horizontal=False):
    
        values = mapping(np.linspace(0, 1, 256), invert=True)
        sm = ScalarMappable(cmap=cmap)
        sm.set_array(values)

        if horizontal==False:
            fig, ax = plt.subplots(figsize=(1, 5))
            fig.colorbar(sm, cax=ax)
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            # Swap the dimensions for a horizontal layout
            fig, ax = plt.subplots(figsize=(5, 1)) 
            # Tell matplotlib to lay it out horizontally
            fig.colorbar(sm, cax=ax, orientation='horizontal')
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
            
            
    
  

    def show_bcs(self, settings=['default'], vertex_id=0, patch_id=0):

        

        patch_sets = [self.deformed_patches]
        if self.show_on_coarse == True:
            patch_sets.append(self.coarse_patches)


        actual_vertex_positions = self.bcs2.actual_vertex_positions()
        radius = 0.01
        
        actual_vtx_blobs = [
            o3d.geometry.TriangleMesh.create_sphere(radius)
            .translate(pos.detach().cpu().numpy())
            .paint_uniform_color([1, 0, 0])
            for pos in actual_vertex_positions
        ]

        vtx_blobs = [
            o3d.geometry.TriangleMesh.create_sphere(radius)
            .translate(pos.detach().cpu().numpy())
            .paint_uniform_color([0, 0, 1])
            for pos in self.bcs2.V
        ]

        for patches in patch_sets:
            
            
            for i, patch in enumerate(patches):
                
                colors = self.area_dist_cmap( self.area_dist_mapping ( self.area_dist[i,:].detach().cpu().numpy() ) )[:,:-1]

                patch.vertex_colors = Vector3dVector(colors)
            
            save_mesh_screenshots(patches+vtx_blobs+actual_vtx_blobs, output_prefix="output/cts_area_dist")


            
            for i, patch in enumerate(patches):
                
                colors = self.elastic_energy_cmap( self.elastic_energy_mapping ( self.elastic_energy[i,:].detach().cpu().numpy() ) )[:,:-1]

                patch.vertex_colors = o3d.utility.Vector3dVector(colors)
                
            #o3d.visualization.draw_geometries(patches, window_name='elastic energy')
            save_mesh_screenshots(patches+vtx_blobs+actual_vtx_blobs, output_prefix='output/elastic energy')
        


        data = np.log10( self.elastic_energy.flatten().detach().cpu().numpy() )
        #data = data[(data >= -20)]
        plt.figure()
        plt.hist(data, bins=100)
        plt.title("Elastic Energy Distribution (log10) ")
        plt.show()


        






'''
# deprecated: for fixed correspondences

def error_on_pcd(bcs, num_samples=100000, include_edges=False):
    visual_sample_dict = bcs.compute_samples(num_samples=num_samples, analytic=True, analytic_shape='torus', include_edges=include_edges)
    
    visual_t = visual_sample_dict['uv']
    visual_output = bcs.forward(visual_t).detach().numpy()
    
    visual_target = visual_sample_dict['target'].numpy()
        
    error = np.sqrt(((visual_output - visual_target)**2).sum(-1))
    
    pcd_visualiser = BCS_visualiser(bcs, mesh_res=2, use_original_mesh=False)
    
    colors = pcd_visualiser.error_cmap(pcd_visualiser.error_mapping( error ))[:,:,:3]
    
    
    pcd_visualiser.show_cbar(pcd_visualiser.error_mapping, pcd_visualiser.error_cmap, filename='output/error_cbar.png')
    
    show_f_list([{'points':visual_output, 'colors':colors}])
'''














'''


############# Showing B function (used for radial and 2d_pou) and visualising the overlap #############


x = torch.arange(0.0, 1.01, 0.01)

# Triangle vertices
A = np.array([0, 0])
B = np.array([1, 0])
C = np.array([0.5, np.sqrt(3)/2])
vertices = [A, B, C]
r = (1+overlap_param)/2

# Create side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# First plot: B(x) and B(1 - x)
ax1.plot(x, bcs.B(x), label='B(x)')
ax1.plot(x, bcs.B(1 - x), label='B(1 - x)')
ax1.set_title("Function Plot")
ax1.legend()
ax1.grid(True)

# Second plot: triangle and circles
triangle = plt.Polygon(vertices, fill=None, edgecolor='black')
ax2.add_patch(triangle)
for v in vertices:
    circle_nonzero = plt.Circle(v, r, fill=False, color='blue', linestyle='--')
    ax2.add_patch(circle_nonzero)
    circle_unit = plt.Circle(v, r-overlap_param, fill=False, color='green', linestyle='--')
    ax2.add_patch(circle_unit)

ax2.set_aspect('equal')
ax2.set_xlim(-r, 1 + r)
ax2.set_ylim(-r, 1)
ax2.set_title("Triangle with Circles")
ax2.grid(True)

plt.tight_layout()
plt.show()

'''













