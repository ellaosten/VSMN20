Parameter study
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import flowmodel as fm

def run_parameter_study():
    """Run a parameter study by varying the barrier depth"""

    # Parameter to vary
    d_values = np.linspace(3.0, 7.0, 10)
    max_flow_values = []

    # Run simulation for each value
    for d in d_values:
        print(f"Simulating with barrier depth d = {d:.2f}")

        # Create model with current parameter
        model_params = fm.ModelParams()
        model_params.d = d # Set current barrier depth

        # Other parameters remain constant
        model_params.w = 100.0
        model_params.h = 10.0
        model_params.t = 0.5
        model_params.kx = 20.0
        model_params.ky = 20.0

        # Create result storage and solver
        model_results = fm.ModelResult()
        solver = fm.ModelSolver(model_params, model_results)

        # Run the simulation 
        solver.execute()

        # Store the maximum flow for this configuration
        max_flow_values.append(model_results.max_value)
        print(f"Max flow value: {model_results.max_value:.4f}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(d_values, max_flow_values, 'o-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Barrier Depth (d)')
    plt.ylabel('Max Flow')
    plt.title('Parameter Study: Effect of barrier depth on maximum flow')
    plt.savefig("parameter_study.png")
    plt.show()

    # Returns results for further analysis if needed
    return d_values, max_flow_values

if __name__ == "__main__":
    run_parameter_study()




