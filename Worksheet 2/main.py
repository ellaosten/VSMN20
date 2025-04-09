# -*- coding: utf-8 -*-
import flowmodel as fm

if __name__ == "__main__":
    model_params = fm.ModelParams()
    model_results = fm.ModelResult()

    model_params.save("model_param.json")
    
    solver = fm.ModelSolver(model_params, model_results)
    solver.execute()
    
    report = fm.ModelReport(model_params, model_results)
    print(report)