import numpy as np 
import calfem.core as cfc
import calfem.utils as cfu
import tabulate as tab
import json

class ModelParams:
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
            [2, 60.0],
            [4, 60.0],
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
    
    def save(self, filename):
        """Save input to file."""
        model_params = {}
        model_params["version"] = self.version
        model_params["t"] = self.t
        model_params["ep"] = self.ep

        model_params["coord"] = self.coord.tolist() # Convert NumPy array to list for JSON compatibility

        ofile = oplen(filename, "w")
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
        self.coord = np.array(model_params["coord"])

class ModelResult:
    """Class for storing results from calculations"""
    def __inti__ (self):
        self.a = None
        self.r = None
        self.ed = None

class ModelSolver:
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

        # --- Store results in model_result
        self.model_results.a = a
        self.model_results.r = r
        self.model_results.ed = ed

        # Calculate element flows and gradients
        es = np.zeros([n_el, 2])
        et = np.zeros([n_el, 2])

        for elx, ely, eld, eles, elet in zip(ex, ey, ed, es, et):
            es_el, et_el = cfc.flw2ts(elx, ely, D, eld)
            eles[:] = es_el[0, :]
            elet[:] = et_el[0, :]

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
        self.add_text("Input parameters")
        self.add_text()
        self.add_text(
            tab.tabulate(np.asarray([np.hstack((self.model_params.t, self.model_params.k))]), 
                         headers=["t", "k"], numalign="right",
            floatfmt=".0f", 
            tablefmt="psql",)
        )
        self.add_text("Coordinates")
        self.add_text()
        self.add_text(
            tab.tabulate(self.model_params.coord, headers=["x", "y"], numalign="right",
            floatfmt=".0f", 
            tablefmt="psql",)
        )
    
        self.add_text("Dofs")
        self.add_text()
        self.add_text(
            tab.tabulate(self.model_params.dof.flatten().reshape(-1,1), headers=["Dofs"], numalign="right",
            floatfmt=".0f", 
            tablefmt="psql",)
        )

        return self.report


# -*- coding: utf-8 -*-

if __name__ == "__main__":
    print("This will is executed as a script and not imported")

# -*- coding: utf-8 -*-

import flowmodel as fm

if __name__ == "__main__":

    model_params = fm.ModelParams()
    model_results = fm.ModelResult()

    solver = fm.ModelSolver(model_params, model_results)
    solver.execute()

    report = fm.ModelReport(model_params, model_results)
    print(report)
    


    
    