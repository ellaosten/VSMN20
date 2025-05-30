# -*- coding: utf-8 -*-
import os

from qtpy.QtCore import QThread
from qtpy.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from qtpy import uic
from qtpy.uic import loadUi

import Flowmodel as fm

class SolverThread(QThread):
    "Class for handling calculations in the background"
    def __init__(self, solver, param_study = False):
        "Classconstruction"
        QThread.__init__(self)
        self.solver = solver
        self.param_study = param_study

        super().__init__()
        self.param_study = param_study
        self.solver = solver
    
    def __del__(self):
        self.wait()
        
    def run(self):
        "RUn calculations in a separate thread"
        if self.param_study:
            self.solver.execute_param_study()
        else:
            self.solver.execute()

class MainWindow(QMainWindow):
    "MainWindow-class handle the user interface and interaction with the model"
    def __init__(self):
        "Constructor for the main window"
        self.model_params = fm.ModelParams()
        self.model_results = fm.ModelResult()

        super(QMainWindow, self).__init__()
       
        # Load the UI file
        print("---> Current working directory:", os.getcwd())
        uic.loadUi('mainwindow.ui', self) # For local testing
        # Show window
        self.show()
        self.raise_()

        # Menu placement in ui window
        self.menuBar().setNativeMenuBar(False)

        # Element size slider
        self.element_size_label.setText('Element size:')
        self.element_size_slider.setRange(50, 300)

        # Disable visualization buttons initially
        for btn in (self.show_geometry_button,
                    self.show_mesh_button,
                    self.show_nodal_values_button,
                    self.show_element_values_button):
            btn.setEnabled(False)
        
        # Connect menu actions
        self.actionnew_action.triggered.connect(self.handle_new_action)
        self.actionopen_action.triggered.connect(self.handle_open_action)
        self.actionsave_action.triggered.connect(self.handle_save_action)
        self.actionsave_as_action.triggered.connect(self.handle_save_as_action)
        self.actionexit_action.triggered.connect(self.handle_action_exit)
        self.actionexecute_action.triggered.connect(self.handle_action_execute)

        # Connect visualization buttons
        self.show_geometry_button.clicked.connect(self.on_show_geometry)
        self.show_mesh_button.clicked.connect(self.on_show_mesh)
        self.show_nodal_values_button.clicked.connect(self.on_show_nodal_values)
        self.show_element_values_button.clicked.connect(self.on_show_element_values)

        # Connect param study button
        self.param_button.clicked.connect(self.on_execute_param_study)

        # Slider only updates element_size_factor
        self.element_size_slider.valueChanged.connect(self.on_element_size_changed)

        self.update_controls()
    
    def update_controls(self):
        "Default values for widget"
        self.w_edit.setText(str(self.model_params.w))
        self.h_edit.setText(str(self.model_params.h))
        self.d_edit.setText(str(self.model_params.d))
        self.t_edit.setText(str(self.model_params.t))
        self.kx_edit.setText(str(self.model_params.kx))
        self.ky_edit.setText(str(self.model_params.ky))
       
        self.d_end_edit.setText(str(self.model_params.d_end))
        self.t_end_edit.setText(str(self.model_params.t_end))
       
        self.left_bc_edit.setText(str(self.model_params.bc_values['left_bc']))
        self.right_bc_edit.setText(str(self.model_params.bc_values['right_bc']))
        
        self.element_size_slider.setValue(int(self.model_params.el_size_factor))

        if self.paramvarydradio.isChecked():
            self.model_params.param_d = True

        if self.paramvarytradio.isChecked():
            self.model_params.param_t = True

    def update_model(self):
        "Load values from controller and update model"

        # Make controll for valied inputs

        # Define the mapping of UI fields to model parameters
        fields = [
             (self.w_edit, 'w', 'Width of domain (w)'),
             (self.h_edit, 'h', 'Height of domain (h)'),
             (self.d_edit, 'd', 'Depth of barrier (d)'),
             (self.t_edit, 't', 'Thickness of barrier (t)'),
             (self.kx_edit, 'kx', 'Permeability in x-direction (kx)'),
             (self.ky_edit, 'ky', 'Permeability in y-direction (ky)'),
             (self.left_bc_edit, 'left_bc', 'Left surface pressure (mvp)'),
             (self.right_bc_edit, 'right_bc', 'Right surface pressure (mvp)'),
        ]

        invalid_fields = []
        parsed_values = {}

        # Try converting all float fields
        for widget, key, label in fields:
            try:
                parsed_values[key] = float(widget.text())
            except ValueError:
                invalid_fields.append(label)

        # Check int conversion separately
        try:
            param_steps = int(self.param_step.text())
        except ValueError:
            invalid_fields.append('Number of parameter steps')

        if invalid_fields:
            QMessageBox.warning(
                self,
                'Invalid Input',
                'Please enter valid numbers for:\n' + '\n'.join(invalid_fields)
            )
            return False # Abort
        
        # If everything is valid, assign to model:
        self.model_params.w = float(self.w_edit.text())
        self.model_params.h = float(self.h_edit.text())
        self.model_params.d = float(self.d_edit.text())
        self.model_params.t = float(self.t_edit.text())
        self.model_params.kx = float(self.kx_edit.text())
        self.model_params.ky = float(self.ky_edit.text())
        self.model_params.d_end = float(self.d_end_edit.text())
        self.model_params.t_end = float(self.t_end_edit.text())
        self.model_params.bc_values['left_bc'] = float(self.left_bc_edit.text())
        self.model_params.bc_values['right_bc'] = float(self.right_bc_edit.text())
        self.model_params.param_steps = int(self.param_step.text())

        self.model_params.el_size_factor = self.element_size_slider.value() / 100.0

        if self.paramvarydradio.isChecked():
            self.model_params.param_d = True

        if self.paramvarytradio.isChecked():
            self.model_params.param_t = True
    
   
    def handle_new_action(self):
        "Creates a new model and resets the UI"
        self.w_edit.clear()
        self.h_edit.clear()
        self.d_edit.clear()
        self.t_edit.clear()
        self.kx_edit.clear()
        self.ky_edit.clear()
        self.d_end_edit.clear()
        self.t_end_edit.clear()
        self.left_bc_edit.clear()
        self.right_bc_edit.clear()
        self.element_size_slider.setValue(50)  # Reset to default value 
    
    
    def handle_open_action(self):
        "Open a model file and load its parameters into the UI"
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Model File', '', 'Model files (*.json *.jpg *.bmp)')
        if filename!=" ":
            self.filename = filename

    def handle_save_action(self):
        "Save model parameters to the current file or prompt for a new file"
        self.update_model()

        if self.filename=="":
            filename,_ = QFileDialog.getSaveFileName(self, 'Save Model File', '', 'Model files (*.json)')

            if filename!="":
                self.filename = filename
    
    def handle_save_as_action(self):
        "Prompt for a file name and save model parameters to that file"
        # Ensure we have a model_params to save into
        if not self.model_params:
            self.model_params = fm.ModelParams()
        
        # Prompt for filename
        fn, _ = QFileDialog.getSaveFileName(self, 'Save As', '', 'Model files (*.json)')

        # If cancel is pressed or no filename is provided
        if not fn:
            return

        # Save the model parameters to the specified file
        try:
            _ = self.update_model()
            self.model_params.save(fn)
            self.model_params.filename = fn
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save model file:\n{e}')
    
    def handle_action_execute(self):
        print("==> handle_action_execute start")
        "Run solver"

        # Disable user interface during calculations
        self.setEnabled(False)

        # Update model from UI
        self.update_model()

        # Create a solver
        self.solver = fm.ModelSolver(self.model_params, self.model_results)

        # Create a thread with the calculations so that the UI doesn't freeze
        self.solver_thread = SolverThread(self.solver)
        self.solver_thread.finished.connect(self.on_solver_finished)
        self.solver_thread.start()
        
        # Silently abort execution if parameters are missing
        if not self.update_model():
            print("==> update_model() returnerade False")
            return
        print("==> startar beräkning")
    
    def handle_action_exit(self):
        "Exit the application"
        self.close()

    
    def on_solver_finished(self):
        print("==> on_solver_finished running")
        "Handle completion os the solver thread"
        self.setEnabled(True)

        # Calculation finished
        self.calc_done = True

        print("==> Calculation finished")

        # Recreate visualization object
        self.visualization = fm.ModelVisualization(self.model_params, self.model_results)

        for btn in (self.show_geometry_button, self.show_mesh_button,
                    self.show_nodal_values_button, self.show_element_values_button):
            btn.setEnabled(True)

        # Print results
        self.report = fm.ModelReport(self.model_params, self.model_results)
        self.report_edit.clear()
        self.report_edit.setPlainText(str(self.report)) 
        
    def on_show_geometry(self):
        print("==> on_show_geometry running")
        "Display the geometry of the model"

        if not self.calc_done:
            print("Calculations not finished!")
        if self.visualization is None:
            print("No visualization found!")

        if not self.calc_done or self.visualization is None:
            QMessageBox.warning(self, 'No data', 'Please run calculation first')
            return
        
        print("Showing geometry now...")
        self.visualization.show_geometry()

    def on_show_mesh(self):
        "Display the finite element mesh of the model"

        if not self.calc_done or self.visualization is None:
            QMessageBox.warning(self, 'No data', 'Please run calculation first')
            return
        
        print("Showing mesh now...")
        self.visualization.show_mesh()
    
    def on_show_nodal_values(self):
        "Display the nodal values of the model"

        if not self.calc_done or self.visualization is None:
            QMessageBox.warning(self, 'No data', 'Please run calculation first')
            return
        
        print("Showing nodal values now...")
        self.visualization.show_nodal_values()
    
    def on_show_element_values(self):
        "Display the element values of the model"

        if not self.calc_done or self.visualization is None:
            QMessageBox.warning(self, 'No data', 'Please run calculation first')
            return
        
        print("Visualizing element values now...")
        self.visualization.show_element_values()
    
    def on_element_size_changed(self, value):
        "Update the element size factor based on the slider value"
        if self.model_params is None:
            self.model_params = fm.ModelParams()
        self.model_params.el_size_factor = value / 100.0
    
    def on_execute_param_study(self):
        "Run a parameter study either on depth (d) or thickness (t)"
        
        # Update model from UI
        self.update_model()
        
        # Update filename
        self.model_params.params_filename = "param_study"

        # Create a solver for the parameter study
        self.solver = fm.ModelSolver(self.model_params, self.model_results)

        # Create a thread with the calculations so that the UI doesn't freeze
        self.solverThread = SolverThread(self.solver, param_study=True)
        self.solverThread.finished.connect(self.on_solver_finished)
        self.solverThread.start()    
