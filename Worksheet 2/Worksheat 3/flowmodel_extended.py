# -*- coding: utf-8 -*-
import json
import math
import sys

import calfem.core as cfc
import calfem.geometry as cfg  # for geomtric modeling
import calfem.mesh as cfm       # for meshing
import calfem.vis_mpl as cfv    # for visualization
import calfem.utils as cfu      # for utility functions
import matplotlib.pylab as plt ## osäker på detta

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
        self.D = np.array([[self.kx, 0], [0, self.ky]]) # Permeability matrix

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
    
    def save(self, filename):
        """Save input to file."""
        model_params = {}
        model_params["version"] = self.version
        model_params["t"] = self.t
        model_params["ep"] = self.ep
        model_params["w"] = self.w
        model_params["h"] = self.h
        model_params["d"] = self.d
        model_params["kx"] = self.kx
        model_params["ky"] = self.ky
        model_params["D"] = self.D.tolist() # Convert numpy array to list for JSON compatibility
        model_params["el_size_factor"] = self.el_size_factor
        model_params["bc_markers"] = self.bc_markers
        model_params["bc_values"] = self.bc_values
        model_params["load_markers"] = self.load_markers
        model_params["load_values"] = self.load_values

        #model_params["coord"] = self.coord.tolist() # Convert NumPy array to list for JSON compatibility

        ofile = open(filename, "w")
        json.dump(model_params, ofile, sort_keys = True, indent = 4)
        ofile.close()

    def load(self, filename):
        """Read input from file."""

        ifile = open(filename, "r")
        model_params = json.load(ifile)
        ifile.close()

        self.version = model_params["version"]
        self.t = model_params["t"]
        self.ep = model_params["ep"]
        #self.coord = np.array(model_params["coord"])
        self.w = model_params["w"]
        self.h = model_params["h"]
        self.d = model_params["d"]
        self.kx = model_params["kx"]
        self.ky = model_params["ky"]
        self.D = np.array(model_params["D"]) # Convert list back to numpy array
        self.el_size_factor = model_params["el_size_factor"]
        self.bc_markers = model_params["bc_markers"]
        self.bc_values = model_params["bc_values"]
        self.load_markers = model_params["load_markers"]
        self.load_values = model_params["load_values"]

class ModelResult:
    """Class defining the model parameters"""
    def __init__(self):

        self.version = 1.0

        self.t = 1.0
        self.k = 50.0
        self.ep = [self.t] 

        # --- Element properties
        self.D = np.array([
            [50.0, 0.0],
            [0.0, 50.0],
        ])

        # --- Create input for the example use cases

        self.coord = np.array ([
            [0.0, 0.0],
            [0.0, 600.0],
            [600.0, 0.0],
            [600.0, 600.0],
            [1200.0, 0.0],
            [1200.0, 600.0],
        ])

        # --- Element coordinates in x

        self.ex = np.array ([
            [0.0, 600.0, 0.0],
            [0.0, 600.0, 600.0],
            [600.0, 1200.0, 600.0],
            [600.0, 1200.0, 1200.0],
        ])

        # --- Element coordinates in y
        self.ey = np.array ([
            [0.0, 600.0, 600.0],
            [0.0, 0.0, 600.0],
            [0.0, 600.0, 600.0],
            [0.0, 0.0, 600.0], 
        ])

        # --- Element topology
        self.edof = np.array ([
            [1, 4, 2],
            [1, 3, 4],
            [3, 6, 4],
            [3, 5, 6],
        ])
        # --- Loads
        self.loads = ([
            [6, -400.0],
        ])

         # --- Boundary conditions
        self.bcs = ([
            [2, 60],
            [4, 60],
        ])

        # --- Degrees of freedom
        self.dof = np.array ([1, 2, 3, 4, 5, 6])

        # --- Element numbers
        self.elem = np.array([1, 2, 3, 4])

    """Class for storing results from calculations"""
    def __inti__ (self):
        self.a = None
        self.r = None
        self.ed = None
        self.es = None
        self.et = None

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
         # Create a new figure
        cfv.figure()
        cfv.clf()

        # Draw Nodal Pressure
        cfv.draw_nodal_values(
            self.model_result.a,
            coords=self.model_result.coords,
            edof=self.model_result.edof,
            title="Nodal Pressure"
        )

    def show_element_values(self):
         """Class for solving the finite element model"""
    def __init__(self, model_params, model_results):
        self.model_params = model_params
        self.model_results = model_results




### Gammal kod

        """Display element flows"""
        # Create a new figure
        cfv.figure()
        cfv.clf()
        # Draw Element Flow
        cfv.draw_element_values(
            self.model_results.flow,
            coords=self.model_results.coords,
            edof=self.model_results.edof,
            dofs_per_node=self.model_results.dofs_per_node,
            el_type=self.model_results.el_type,
            title="Element Flows",
        )
    
    def wait(self):
        """Wait for user to close all visualization windows"""
        cfv.show_and_wait()

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
        D = self.model_params.D

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

         # Initialize the global stiffness matrix and load vector

        n_dofs = np.max(dofs)
        K = np.zeros((n_dofs, n_dofs))
        f = np.zeros((n_dofs, 1))

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
                    bc_prescr,
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
        # --- Calculate element flows and gradients

        for load in loads:
            dof = load[0]
            mag = load[1]
            f[dof - 1] = mag
        
        bc_prescr = []
        bc_value=[]

        for bc in bcs:
            dof = bc[0]
            value = bc[1]
            bc_prescr.append(dof)
            bc_value.append(value)

        bc_prescr = np.array(bc_prescr)
        bc_value = np.array(bc_value)

        a, r = cfc.solveq(K, f, bc_prescr, bc_value)

        ed = cfc.extractEldisp(edof, a)
        n_el = edof.shape[0] # 4
        es = np.zeros((n_el, 2))
        et = np.zeros((n_el, 2))

        a_and_r = np.hstack((a, r))

        temp_table = tab.tabulate(
            np.asarray(a_and_r),
            headers=["D.o.f.", "Phi [m]", "q [m^2/day]"],
            numalign="right",
            floatfmt=".4f",
            tablefmt="psq1",
            showindex=range(1, len(a_and_r) + 1),
            )

        # Calculate element flows and gradients
        es = np.zeros([n_el, 2])
        et = np.zeros([n_el, 2])

        for elx, ely, eld, eles, elet in zip(ex, ey, ed, es, et):
            es_el, et_el = cfc.flw2ts(elx, ely, D, eld)
            eles[:] = es_el[0, :]
            elet[:] = et_el[0, :]

        # --- Store results in model_result
        self.model_results.a = a
        self.model_results.r = r
        self.model_results.ed = ed
        self.model_results.es = es
        self.model_results.et = et
    
     # Calculate maximum flow for parameter studies
        element_values = np.sqrt(np.sum(self.model_results.es**2, axis=1))
        self.model_results.max_value = np.max(element_values)



###################### vad händer här egentligen

    """Class for performing the model computations"""
    def __init__(self, model_params, model_results):
        self.model_params = model_params
        self.model_results = model_results
    
    def execute(self):
        # --- Assign shorter variables names from model properties
        edof = self.model_params.edof
        coord = self.model_params.coord
        loads = self.model_params.loads
        dof = self.model_params.dof
        bcs = self.model_params.bcs
        ep = self.model_params.ep
        D = self.model_params.D
        ex = self.model_params.ex
        ey = self.model_params.ey

    def execute(self):
        # Initialize the global stiffness matrix and load vector

        n_dofs = np.max(dofs)
        K = np.zeros((n_dofs, n_dofs))
        f = np.zeros((n_dofs, 1))

        ## Assemble to global matrix in simular way to worksheet 2
        # --- Create global stiffness matrix and load vector
        K = np.zeros((6,6))
        f = np.zeros((6,1))

        f[5] = -400.0
        # --- Calculate element stiffness matrices and assemble global stiffness matrix
        ke1 = cfc.flw2te(ex[0,:], ey[0,:], ep, D)
        ke2 = cfc.flw2te(ex[1,:], ey[1,:], ep, D)
        ke3 = cfc.flw2te(ex[2,:], ey[2,:], ep, D)
        ke4 = cfc.flw2te(ex[3,:], ey[3,:], ep, D)

        # --- Assemble global stiffness matrix
        cfc.assem(edof[0,:], K, ke1, f)
        cfc.assem(edof[1,:], K, ke2, f)
        cfc.assem(edof[2,:], K, ke3, f)
        cfc.assem(edof[3,:], K, ke4, f)

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
                    bc_prescr,
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
        


class ModelReport:
    """Class for presenting input and output parameters in report form"""
    def __init__(self, model_params, model_results):
        self.model_params = model_params
        self.model_results = model_results
        self.report = ""

    def clear(self):
        self.report = ""

    def add_text(self, text=""):
        self.report += str(text) + "\n"

    def __str__(self):
        self.clear()
        self.add_text()
        self.add_text("----------- Model Input --------------------")
        self.add_text()
        self.add_text("Input parameters")
        self.add_text()
        self.add_text(
            tab.tabulate(np.asarray([np.hstack((self.model_params.t, self.model_params.k))]), 
                         headers=["t", "k"], numalign="right",
            floatfmt=".0f", 
            tablefmt="psql",)
        )
        
        self.add_text()
        self.add_text("Coordinates")
        self.add_text()
        self.add_text(
            tab.tabulate(self.model_params.coord, headers=["x", "y"], numalign="right",
            floatfmt=".0f", 
            tablefmt="psql",)
        )
    
        self.add_text()
        self.add_text("Dofs")
        self.add_text()
        self.add_text(
            tab.tabulate(self.model_params.dof.flatten().reshape(-1,1), headers=["Dofs"], numalign="right",
            floatfmt=".0f", 
            tablefmt="psql",)
        )

        self.add_text()
        self.add_text("Edof")
        self.add_text()
        self.add_text(
            tab.tabulate(self.model_params.edof, headers=["e1", "e2", "e3"], numalign="right",
            floatfmt=".0f", 
            tablefmt="psql",)
        )

        self.add_text()
        self.add_text("Loads")
        self.add_text()
        self.add_text(
            tab.tabulate(self.model_params.loads, headers=["dof", "mag"], numalign="right",
            floatfmt=".0f", 
            tablefmt="psql",)
        )

        self.add_text()
        self.add_text("Boundary conditions")
        self.add_text()
        self.add_text(
            tab.tabulate(self.model_params.bcs, headers=["dof", "value"], numalign="right",
            floatfmt=".0f", 
            tablefmt="psql",)
        )
        
        self.add_text()
        self.add_text("----------- Results --------------------")
        self.add_text()
        self.add_text("Nodal temps and flows")
        self.add_text()
        dof=self.model_params.dof.flatten().reshape(-1,1)
        a=np.array(self.model_results.a).flatten().reshape(-1,1)
        r=np.array(self.model_results.r).flatten().reshape(-1,1)
        self.add_text(
            tab.tabulate(
            np.array(np.hstack((dof, a, r))),
            headers=["D.o.f.", "Phi [m]", "q [m^2/day]"],
            numalign="right",
            tablefmt="psq1",
            floatfmt=(".0f", ".4f", ".4f"),)
            
        )

        self.add_text()
        self.add_text("Element flows")
        self.add_text()
        self.add_text(
            tab.tabulate(np.array(np.hstack((self.model_params.elem.reshape(-1,1), self.model_results.es))),
            headers=["Element", "q_x [m^2/day]", "q_y [m^2/day]"],
            numalign="right",
            tablefmt="psql",
            floatfmt=(".0f", ".4f", ".4f"),),
        )
        
        self.add_text()
        self.add_text("Element gradients")
        self.add_text()
        self.add_text(
            tab.tabulate(np.array(np.hstack((self.model_params.elem.reshape(-1,1), self.model_results.et))),
            headers=["Element", "grad_x [1/m]", "grad_y [1/m]"],
            numalign="right",
            tablefmt="psql",
            floatfmt=(".0f", ".4f", ".4f"),),
        )
        self.add_text()
        self.add_text("Element pressure")
        self.add_text()
        self.add_text(
            tab.tabulate(np.array(np.hstack((self.model_params.elem.reshape(-1,1), self.model_results.ed))),
            headers=["Element", "phi_1 [m]", "phi_2 [m]", "phi_3 [m]"],
            numalign="right",
            tablefmt="psql",
            floatfmt=(".0f", ".4f", ".4f", ".4f"),),
        )
        
        return self.report