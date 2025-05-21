# -*- coding: utf-8 -*-
import json
import math
import sys

import calfem.core as cfc
import calfem.geometry as cfg  # for geomtric modeling
import calfem.mesh as cfm       # for meshing
import calfem.vis_mpl as cfv    # for visualization
import calfem.utils as cfu      # for utility functions
import matplotlib.pylab as plt 

import numpy as np
import tabulate as tab
import pyvtk as vtk

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
        self.ep = [self.t, int(2)] # element properties

        # Material properties

        self.kx = 20.0 # permeability in x-direction
        self.ky = 20.0 # permeability in y-direction
        self.D = np.array([[self.kx, 0], [0, self.ky]]) # Permeability matrix

        # Additional parameters for parametric study
        self.param_d = 1.0
        self.param_t = 1.0
        self.d_start = self.d
        self.d_end = 9.0
        self.t_start = self.t
        self.t_end = 5.0
        self.param_filename = 1
        self.param_steps = 1

        # Mesh control

        self.el_size_factor = 0.5 # Controls element size in mesh generation

        # Boundary conditions and loads will now reference markers instead of
        # node numbers or degrees of freedom

        self.bc_markers = {
            "left_bc" : 10, # Markers for left boundary condition
            "right_bc" : 20 # Markers for right boundary condition
        }
        self.bc_values = {
            "left_bc" : 60.0, # Left boundary condition value
            "right_bc" : 0.0 # Right boundary condition value
        }

        self.load_markers = {
        }
        self.load_values = {
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
        # Add points for the barrier

        g.point([0, 0]) # Point 0: Bottom left corner
        g.point([w, 0]) # Point 1: Bottom right corner
        g.point([w, h]) # Point 2: Top right corner
        g.point([w/2 + t/2, h]) # Point 3: Top right corner of barrier 

        g.point([w/2 + t/2, h-d]) # Point 4: Bottom right corner of barrier
        g.point([w/2 - t/2, h-d]) # Point 5: Bottom left corner of barrier
        g.point([w/2 - t/2, h]) # Point 6: Top left corner of barrier    
        g.point([0, h]) # Point 7: Top left corner

        # Define splines (lines) connecting the points
        # Use markers for boundaries with conditions

        g.spline([0, 1]) 
        g.spline([1, 2]) 
        g.spline([2, 3], marker=self.bc_markers["right_bc"])
        g.spline([3, 4]) 
        g.spline([4, 5])
        g.spline([5, 6])
        g.spline([6, 7], marker=self.bc_markers["left_bc"]) # Left boundary, marker for fixed value
        g.spline([7, 0]) 
        

        # Define the surfaces (domain) using the spline indices
        # Surface is defined by a list of spline indices that form a closed loop

        g.surface([0, 1, 2, 3, 4, 5, 6, 7])

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
    """Class for storing results from calculations"""
    def __inti__ (self):
        
        # Initialize attributes for mesh and geometry
        self.loads = None
        self.bcs = None
        self.coords = None
        self.edof = None
        self.dofs = None
        self.bdofs = None
        self.boundary_elements = None
        self.geometry = None

        # Initialize attributes for results
        self.a = None
        self.r = None
        self.ed = None
        self.es = None
        self.et = None

        self.flow = None
        self.pressure = None
        self.gradient = None

        self.max_nodal_flow = None
        self.max_nodal_pressure = None
        self.max_element_flow = None
        self.max_element_pressure = None
        self.max_element_gradient = None

class ModelVisualization:
    """Class for visualizing model geometry, mesh and results"""
    def __init__(self, model_params, model_results):
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
        cfv.draw_geometry(geometry, draw_points=True, label_points=True, label_curves=True,
                              title="Model Geometry")
        cfv.show_and_wait()
            
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
            cfv.show_and_wait()

    def show_nodal_values(self):
        """Display nodal values (pressure)"""
         # Create a new figure
        cfv.figure()
        cfv.clf()

        # Draw Nodal Pressure
        cfv.draw_nodal_values(
            self.model_results.a,
            coords=self.model_results.coords,
            edof=self.model_results.edof,
            title="Nodal Pressure"
        )
        cfv.show_and_wait()

    def show_element_values(self):
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
        cfv.show_and_wait()
    
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

        for marker_name, marker_id in self.model_params.load_markers.items():
            if marker_name in self.model_params.load_values:
                value = self.model_params.load_values[marker_name]
                
                if marker_id in boundary_elements:
                    for be in boundary_elements[marker_id]:
                        nodes = be["node-number-list"]
                        if len(nodes) != 2:
                            continue
                        
                        node1 = nodes[0] - 1
                        node2 = nodes[1] - 1

                        dofs_node1 = bdofs.get(node1)
                        dofs_node2 = bdofs.get(node2)

                        if dofs_node1 in None or dofs_node2 in None:
                            continue
                        dof1 = dofs_node1[0]
                        dof2 = dofs_node2[0]

                        x1, y1 = coords[node1]
                        x2, y2 = coords[node2]

                        edge_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        f[dof1] += value * edge_length / 2.0
                        f[dof2] += value * edge_length / 2.0
        
        # Global stiffness matrix assembly
        nDofs = np.size(dofs) # number of global degrees of freedom
        ex, ey = cfc.coordxtr(edof, coords, dofs) # extract coordinates of elements
        K = np.zeros([nDofs, nDofs]) # global stiffness matrix

        n_el = edof.shape[0] # number of elements
        ep = np.tile(self.model_params.ep, (n_el, 1)).astype(object)

        for i, (eltopo, elx, ely) in enumerate(zip(edof, ex, ey)):
            thickness = float(ep[i][0])
            integration_rule = int(ep[i][1])
            el_ep = [thickness, integration_rule]
            Ke = cfc.flw2i4e(elx, ely, el_ep, D)
            cfc.assem(eltopo, K, Ke)
        
        # Global load vector assembly
        f = np.zeros([nDofs, 1])

        # Boundary conditions
        bc = np.array([], int)
        bcVal = np.array([], int)

        for name, marker in self.model_params.bc_markers.items():
            value = self.model_params.bc_values.get(name, 0.0)
            bc, bcVal = cfu.applybc(bdofs, bc, bcVal, marker, value)
        
        # Solve the system of equations
        a, r = cfc.solveq(K, f, bc, bcVal)
        ed = cfc.extractEldisp(edof, a)

        # Calculate element flows and gradient
        flow = []
        gradient = []

        for i in range(edof.shape[0]):
            el_ep = [float(ep[i][0]), int(ep[i][1])]
            es, et, eci = cfc.flw2i4s(ex[i, :], ey[i, :], el_ep, D, ed[i, :])
            flow.append(np.sqrt(es[0, 0]**2 + es[0, 1]**2))
            gradient.append(np.sqrt(et[0, 0]**2 + et[0, 1]**2))

        # Maximal flow, pressure and gradient for nodes and elements
        max_nodal_flow = np.max(np.abs(r))
        max_nodal_pressure = np.max(np.abs(a))
        max_element_flow = np.max(np.abs(flow))
        max_element_pressure = np.max(np.abs(ed))
        max_element_gradient = np.max(np.abs(gradient))
        # Store results in model_results
        self.model_results.a = a
        self.model_results.r = r
        self.model_results.ed = ed
        self.model_results.es = es 
        self.model_results.et = et

        self.model_results.flow = flow
        self.model_results.gradient = gradient

        self.model_results.loads = list(zip(bc, bcVal))
        self.model_results.bcs = list(zip(bc, bcVal))
        self.model_results.edof = edof
        self.model_results.dofs = dofs
        self.model_results.coords = coords
        self.model_results.elem = np.arange(edof.shape[0])

        self.model_results.max_nodal_flow = max_nodal_flow
        self.model_results.max_nodal_pressure = max_nodal_pressure
        self.model_results.max_element_flow = max_element_flow
        self.model_results.max_element_pressure = max_element_pressure
        self.model_results.max_element_gradient = max_element_gradient

    def run_parameter_study(self):
        """Run a parameter study by varying the barrier depth"""

        # Parameter to vary
        d_values = np.linspace(1.0, 9.0, 9)
        max_flow_values = []

        ## Add parameter study for thickness t
        #t_values = np.linspace(0.5, 5.0, 10)

        # Run simulation for each value
        for d in d_values:
            print(f"Simulating with barrier depth d = {d:.2f}...")

            # Create model with current parameter
            model_params = ModelParams()
            model_params.d = d # Set current barrier depth

            # Other parameters remain constant
            model_params.w = 100.0
            model_params.h = 10.0
            model_params.t = 0.5
            model_params.kx = 20.0
            model_params.ky = 20.0

            # Create result storage and solver
            model_results = ModelResult()
            model_solver = ModelSolver(model_params, model_results)

            # Run the simulation 
            model_solver.execute()

            # Store the maximum flow for this configuration
            max_flow_values.append(np.max(model_results.es))
            print(f"Max flow value: {np.max(model_results.es):.4f}")
        
        ## Run simulation for each value of t
        #for t in t_values:
            #print(f"Simulating with thickness t = {t:.2f}...")

            # Create model with current parameter
            #model_params = ModelParams()
            #model_params.t = t

            # Other parameters remain constant
            #model_params.w = 100.0
            #model_params.h = 10.0
            #model_params.d = 5.0
            #model_params.kx = 20.0
            #model_params.ky = 20.0

            # Create result storage and solver
            #model_results = ModelResult()
            #model_solver = ModelSolver(model_params, model_results)
            
            # Run the simulation
            #model_solver.execute()
            
            # Store the maximum flow for this configuration
            #max_flow_values.append(np.max(model_results.es))
            #print(f"Max flow value: {np.max(model_results.es):.4f}")

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(d_values, max_flow_values, 'o-', linewidth=2)
        plt.grid(True)
        plt.xlabel('Barrier Depth (d)')
        plt.ylabel('Maximum Flow')
        plt.title('Parameter Study: Effect of barrier depth on maximum flow')
        plt.savefig("parameter_study.png")
        plt.show()

        # Returns results for further analysis if needed
        return d_values, max_flow_values
    
    def execute_param_study(self):
        "Kör parameter studie"

        # Store current values for the d and t parameters

        old_d = self.model_params.d
        old_t = self.model_params.t

        i = 1

        if self.model_params.param_d:
            # Create a simulation range
            d_range = np.linspace(self.model_params.d_start, self.model_params.d_end, self.model_params.param_steps) # kan vara så att s:et ska bort i slutet??

            # Loop over the range
            for d in d_range:
                print("Executing for d = %g..." % d)

                # Modify parameter in self.model_params
                self.model_params.d = d
            
                # Execute calculations
                self.execute()

                # Export to VTK
                vtk_filename = "param_study_%02d.vtk" % i
                self.export_vtk(vtk_filename)
        
        elif self.model_params.param_t:
            # Create a simulation range 
            t_range = np.linspace(self.model_params.t_start, self.model_params.t_end, self.model_params.param_steps)

            # Loop over the range
            for t in t_range:
                print("Executing for t = %g..." % t)

                # Modify parameter in self.model_params
                self.model_params.t = t
            
                # Execute calculations
                self.execute()

                # Export to VTK
                vtk_filename = "param_study_%02d.vtk" % i
                self.export_vtk(vtk_filename)
        
        # Restore preious values of d and t

        self.model_params.d = old_d
        self.model_params.t = old_t
    
    def export_vtk(self, filename):
        "Export results to VTK"
        print("Exporting results to %s." % filename)

        # Extract points and polygons
        points = self.model_results.coords.tolist()
        polygons = (self.model_results.edof-1).tolist()

        # Kontrollera datan
        assert len(points) > 0, "Inga punkter att exportera!"
        assert len(polygons) > 0, "Inga polygoner att exportera!"
        assert len(self.model_results.a) == len(points), "Skalärvärden matchar inte antal punkter!"

        # Create point data from a
        point_data = vtk.PointData(
            vtk.Scalars(self.model_results.a.tolist(), name="Pressure")
        )

        # Create cell data from max_flow and flow
        cell_data = vtk.CellData(
            vtk.Scalars(self.model_results.max_flow, name="max_flow"),
            vtk.Scalars(self.model_results.flow, name="flow")
        )

        # Create structure
        structure = vtk.PolyData(points=points, polygons=polygons)

        # Export to vtk
        vtk_data = vtk.VtkData(structure, point_data, cell_data)
        vtk_data.tofile(filename, "ascii")

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
        self.add_text(
            tab.tabulate([
                ["t", self.model_params.t],
                ["w", self.model_params.w],
                ["h", self.model_params.h],
                ["d", self.model_params.d],
                ["kx", self.model_params.kx],
                ["ky", self.model_params.ky],
                ["Element size", self.model_params.el_size_factor],
                ["Left boundary", self.model_params.bc_values.get("left_bc", "N/A")],
                ["Right boundary", self.model_params.bc_values.get("right_bc", "N/A")]
            ],
            headers=["Parameter", "Value"],
            numalign="right",
            tablefmt="psql",
            floatfmt=".1f",
            )
        )

        self.add_text()
        self.add_text("----------- Model Results ------------------")
        self.add_text()
        self.add_text(
            tab.tabulate(
                [[
                    self.model_results.max_nodal_pressure,
                    self.model_results.max_nodal_flow,
                    self.model_results.max_element_pressure,
                    self.model_results.max_element_flow,
                    self.model_results.max_element_gradient
                ]],
                headers=[
                    "Max Nodal Pressure",
                    "Max Nodal Flow",
                    "Max Element Pressure",
                    "Max Element Flow",
                    "Max Element Gradient"
                ],
                numalign="right",
                tablefmt="psql",
                floatfmt=".4f",
            )
        )

        return self.report