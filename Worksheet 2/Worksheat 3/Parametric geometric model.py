# Parametric geometric model
# -*- coding: utf-8 -*-
import json
import math
import sys

import calfem.core as cfc
import calfem.geometry as cfg  # for geomtric modeling
import calfem.mesh as cfm       # for meshing
import calfem.vis_mpl as cfv    # for visualization
import calfem.utils as cfu      # for utility functions

import numpy as np
import tabulate as tb

class ModelParams:
    """Class defining parametric model properties"""
    def __init__(self):

        # Version tracking

        self.version = 1.0

        # Geometric parameters (example for groundwater problem)

        self.w = 100.0 # width of domain
        self.h = 10.0 # height of domain
        self.d = 5.0 # depth of barrier
        self.t = 0.5 # thickness of domain

        # Material properties

        self.kx = 20.0 # permeability in x-direction
        self.ky = 20.0 # permeability in y-direction

        # Mesh control

        self.el_size_factor = 0.5 # Controls element size in mesh generation

        # Boundary conditions and loads will now reference markers instead of
        # node numbers or degrees of freedom

        self.bc_markers = {
            "left_bc" : 10, #Markers for left boundary condition
            "right_bc" : 20 # Markers for right boundary condition
        }
        self.bc_values = {
            "left_bc" : 60.0, # Left boundary condition value
            "right_bc" : 0.0 # Right boundary condition value
        }

        self.load_markers = {
            "top_load" : 30, # Markers for top load
            "bottom_load" : 40 # Markers for bottom load
        }
        self.load_values = {
            "top_load" : 0.0, # Top load value
            "bottom_load" : -400.0 # Bottom load value
        }

    def geometry(self):
        """Create and return a geometry instance based on defined parameters"""

        # Create a geometry instance

        g = cfg.Geometry()

        # Use shorter variable names for readability 

        w = self.w
        h = self.h
        d = self.d
        t = self.t

        # Define points for the geometry
        # Point indices start at 0

        g.point([0, 0]) # Point 0: Bottom left corner
        g.point([w, 0]) # Point 1: Bottom right corner
        g.point([w, h]) # Point 2: Top right corner
        g.point([0, h]) # Point 3: Top left corner

        # Add points for the barrier 

        g.point([w/2 - t/2, h]) # Point 4: Top left corner of barrier
        g.point([w/2 + t/2, h]) # Point 5: Top right corner of barrier
        g.point([w/2 - t/2, h - d]) # Point 6: Bottom left corner of barrier    
        g.point([w/2 + t/2, h - d]) # Point 7: Bottom right corner of barrier

        # Define splines (lines) connecting the points
        # Use markers for boundaries with conditions

        g.spline([0, 1]) # Bottom boundary
        g.spline([1, 2]) # Right boundary, marker for fixed value
        g.spline([2, 5], marker=self.bc_markers["right_bc"])
        g.spline([5, 4]) # Top of barrier
        g.spline([4, 3], marker=self.bc_markers["left_bc"]) # Left boundary, marker for fixed value
        g.spline([3, 0])
        g.spline([4, 6]) # Left side of barrier
        g.spline([5, 7]) # Right side of barrier
        g.spline([6, 7]) # Bottom of barrier

        # Define the surfaces (domain) using the spline indices
        # Surface is defined by a list of spline indices that form a closed loop

        g.surface([0, 1, 2, 3, 4, 5, 6, 7, 8])

        # Return the complete geometry
        return g
    
class ModelSolver:
    """Class for solving the finite element model"""
    def __init__(self, model_params, model_results):
        self.model_params = model_params
        self.model_results = model_results

    def execute(self):
        """Perform mesh generation and finite element computations"""

        # Creates shorter references to input variables

        ep = self.model_params.ep
        kx = self.model_params.kx
        ky = self.model_params.ky

        # Get geometry from model_params

        geometry = self.model_params.geometry()

        # Store geometry in results for visualization

        self.model_results.geometry = geometry

        # Set up mesh generator

        el_type = 3 # 3 = 4 - node quadrilateral elements
        dofs_per_node = 1 # 1 for scalar problem (flow)

        # Create mesh generator

        mesh = cfm.GmshMeshGenerator(geometry)

        # Configure mesh generator

        mesh.el_type = el_type
        mesh.dofs_per_node = dofs_per_node
        mesh.el_size_factor = self.model_params.el_size_factor
        mesh.return_boundary_elements = True

        # Generate mesh

        coords, edof, dofs, bdofs, element_markers, boundary_elements = mesh.create()

        # Store mesh data in results

        self.model_results.coords = coords
        self.model_results.edof = edof
        self.model_results.dofs = dofs
        self.model_results.bdofs = bdofs
        self.model_results.element_markers = element_markers
        self.model_results.boundary_elements = boundary_elements
        self.model_results.el_type = el_type
        self.model_results.dofs_per_node = dofs_per_node

        ## Use this in simular way as in worksheet 2 but use generated mesh data for FE calculations

class ModelSolver:
    # ... existing code

    def execute(self):
        # Initialize the global stiffness matrix and load vector

        n_dofs = np.max(dofs)
        K = np.zeros((n_dofs, n_dofs))
        f = np.zeros((n_dofs, 1))

        ## Assemble to global matrix in simular way to worksheet 2

        # Apply boundary conditions based on markers

        bc_prescr = []
        bc_values = []

        # For each boundary condition marker in model_params

        for marker_name, marker_id in self.model_params.bc_markers.items():
            if marker_name in self.model_params.bc_values:
                value = self.model_params.bc_values[marker_name]
                cfu.apply_bc_from_markers(
                    bdofs,
                    boundary_elements,
                    marker_id,
                    bc_prescr
                    bc_values,
                    value
                )
        # Convert to numpy arrays
        bc_prescr = np.array(bc_prescr)
        bc_values = np.array(bc_values)

        # Apply loads based on markers
        for marker_name, marker_id in self.model_params.load_markers.items():
            if marker_name in self.model_params.load_values:
                value = self.model_params.load_values[marker_name]
                cfu.apply_load_from_markers(
                    bdofs,
                    boundary_elements,
                    marker_id,
                    f,
                    value
                )
        # Solve equation system
        a, r = cfc.solveq(K, f, bc_prescr, bc_values)
        # Store displacement and reaction forces
        self.model_results.a = a
        self.model_results.r = r

        ## Calculate element displacements, flow etc in similar way to worksheet 2

        # Calculate maximum flow for parameter studies
        element_values = np.sqrt(np.sum(self.model_results.es**2, axis=1))
        self.model_results.max_value = np.max(element_values)

class ModelVisualization:
    """Class for visualizing model geometry, mesh and results"""
    def __inti__(self, model_params, model_results):
        """Constructor"""
        self.model_params = model_params
        self.model_results = model_results

        # Store references to visulalization windows
        self.geom_fig = None
        self.mesh_fig = None
        self.nodal_val_fig = None
        self.element_val_fig = None
        self.deformed_fig = None

        def show_geometry(self):
            """Display model geometry"""
            # Get the geometry from the results
            geometry = self.model_results.geometry
            # Create a new figure 
            cfv.figure()
            cfv.clf()
            # Draw geometry
            cfv.draw_geometry(geometry, draw_points=True, label_points=True, draw_line_numbers=True,
                              title="Model Geometry")
            
            def show_mesh(self):
                """Display finite element mesh"""
                # Create a new figure
                cfv.figure()
                cfv.clf()
                # Draw mesh
                cfv.draw_mesh(
                    coords=self.model_results.coords,
                    edof=self.model_results.edof,
                    dofs_per_node=self.model_results.dofs_per_node,
                    el_type=self.model_results.el_type,
                    filled=True,
                    title="Finite Element Mesh"
                )
        def show_nodal_values(self):
            """Display nodal values (pressure)"""

            ### Implement this method to dispaly nodal values

        def show_element_values(self):
            """Display element values (flow)"""

            ### Implement this method to display element values

        def show_deformed_mesh(self, scale_factor=1.0):
            ## Är detta bara för spänningsproblemet?
        
        def wait(self):
            """Wait for user to close all visualization windows"""
            cfv.show_and_wait()
            


        
        



