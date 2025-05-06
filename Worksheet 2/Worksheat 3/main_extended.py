# -*- coding: utf-8 -*-
import flowmodel_extended as fm # Importing the extended module

if __name__ == "__main__":
    # Initialization
    model_params = fm.ModelParams() # Initiaate class ModelParams
    model_results = fm.ModelResult() # Initiate class ModelResult
    model_solver = fm.ModelSolver(model_params, model_results) # Initiate class ModelSolver
    
    # Save and load
    try:
        model_params.load("model_params.json") # Load file
    except FileNotFoundError:
        print("File not found. Creating a new one.")
        model_params.save("model_params.json")

    # Calculations
    model_solver.execute() # Execute the model solver
    model_solver.run_parameter_study() # Execute the parameter study

    # Results
    report = fm.ModelReport(model_params, model_results) # Initiate class ModelReport
    print(str(report)) # Print the report
    print(report, file=open("Results.txt", "w"))

    # Visualization
    model_visualization = fm.ModelVisualization(model_params, model_results) # Initiate class ModelVisualization
    model_visualization.show_mesh()
    model_visualization.show_geometry()
    model_visualization.show_nodal_values()
    model_visualization.show_element_values()
    model_visualization.wait()
