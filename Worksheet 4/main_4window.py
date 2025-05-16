# -*- coding: utf-8 -*-
import sys
import os

from qtpy.QtCore import QThread
from qtpy.QtWidgets import QApplication, QDialog, QWidget, QMainWindow, QFileDialog
from qtpy import uic
from qtpy.uic import loadUi

import calfem.ui as cfui
import flowmodel_4 as fm
import xml.etree.ElementTree as ET
import numpy as np

class SolverThread(QThread):
    "Klass för att hantera beräkningar i bakgrunden"
    def __init__(self, solver, param_study = False):
        "Klasskonstruktion"
        QThread.__init__(self)
        self.solver = solver
        self.param_study = param_study
    
    #def __del__(self):
       # self.wait()

    def run(self):
        "Kör beräkningarna i en separat tråd"
        self.solver.execute()

class MainWindow(QMainWindow):
    "MainWindow-klass som hanterar vårt huvudfönster"
    def __init__(self):
        "Constructor"
        super(QMainWindow, self).__init__()

        # Visualization object
        self.visualization = None

        # Calculation not finished
        self.calc_done = False

        # Clean UI file and load interface description
        #ui_path = os.path.join(os.path.dirname(__file__), 'mainwindow.ui')
        #loadUi(clean_ui(ui_path), self)

        # Menu placement in ui window
        self.menuBar().setNativeMenuBar(False)

        # Element size slider
        self.element_size_label.setText('Element size:')
        self.element_size_slider.setRange(50, 100)

        # Set input placeholders including boundary fields
        placeholders = {
            'w_text': '100.0 m',
            'h_text': '10.0 m',
            'd_text': '5.0 m',
            't_text': '0.5 m',
            'kx_text': '20.0 m/day',
            'ky_text': '20.0 m/day',
            #'left_bc_text': '60.0 mvp',
            #'right_bc_text': '0.0 mvp',
        }

        # Set placeholder text for all QLineEdit widgets
        for attr, text in placeholders.items():
            if hasattr(self, attr):
                widget = getattr(self, attr)
                widget.clear()
                widget.setPlaceholderText(text)
    
        # Disable visualization buttons initially
        for btn in (self.show_geometry_button,
                    self.show_mesh_button,
                    self.show_nodal_values_button,
                    self.show_element_values_button):
            btn.setEnabled(False)

        # Connect menu actions
        self.new_action.triggered.connect(self.on_new_action)
        self.open_action.triggered.connect(self.on_open_action)
        self.save_action.triggered.connect(self.on_save_action)
        self.save_as_action.triggered.connect(self.on_save_as_action)
        self.exit_action.triggered.connect(self.on_exit_action)
        self.execute_action.triggered.connect(self.on_action_execute)

        # Connect visualization buttons
        self.show_geometry_button.clicked.connect(self.on_show_geometry)
        self.show_mesh_button.clicked.connect(self.on_show_mesh)
        self.show_nodal_values_button.clicked.connect(self.on_show_nodal_values)
        self.show_element_values_button.clicked.connect(self.on_show_element_values)

        # Slider only updates element_size_factor
        self.element_size_slider.valueChanged.connect(self.on_element_size_changed)

        self.model_params = None
        self.model_results = None

        self.show()
        self.raise_()

    def update_model(self):
        "Read UI fields into model_params and update boundary conditions"
        # Ensure we have a ModelParams to write into
        if not self.model_params:
            self.model_params = fm.ModelParams()
    
        # Define the mapping of UI fields to model parameters
        fields = [
            ('w_text', 'w', 'Width of domain (w)'),
            ('h_text', 'h', 'Height of domain (h)'),
            ('d_text', 'd', 'Depth of barrier (d)'),
            ('t_text', 't', 'Thickness of barrier (t)'),
            ('kx_text', 'kx', 'Permeability in x-direction (kx)'),
            ('ky_text', 'ky', 'Permeability in y-direction (ky)'),
            #('left_bc_text', 'left_bc', 'Left surface pressure (mvp)'),
            #('right_bc_text', 'right_bc', 'Right surface pressure (mvp)'),
        ]

        invalid = []

        # Read values from UI fields and set them in model_params
        for widget_name, param_name, label in fields:
            widget = getattr(self, widget_name, None)
            txt = widget.text().strip() if widget else ''
            try:
                value = float(txt)
            except Exception:
                invalid.append(label)
            else:
                setattr(self.model_params, param_name, value)
    
        # Warnings for invalid inputs
        if invalid:
            QMessageBox.warning(
                self,
                'Invalid Input',
                'Please enter valid numbers for:\n' + '\n'.join(invalid)
            )
            return False
    
        # Propagate into bc_values
        if hasattr(self.model_params, 'bc_values'):
            self.model_params.bc_values['left_bc'] = self.model_params.left_bc
            self.model_params.bc_values['right_bc'] = self.model_params.right_bc
    
        # Properties
        mp = self.model_params
        mp.D = np.array([[mp.kx, 0], [0, mp.ky]]) 
        mp.el_size_factor = self.element_size_slider.value() / 100.0
        return True

    def on_new_action(self):
        self.__init__()

    def on_open_action(self):
        "Open a model file and load its parameters into the UI"
        # Open file dialog to select a model file
        fn, _= QFileDialog.getOpenFileName(self, 'Open Model File', '', 'Model files (*.json)')
        if not fn: return
        mp = fm.ModelParams()
        try:
            mp.load(fn)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load model file:\n{e}')
            return
    
        # Set the model parameters in the UI
        for param, attr in [
                        ('w', 'w_text'),
                        ('h', 'h_text'),
                        ('d', 'd_text'),
                        ('t', 't_text'),
                        ('kx', 'kx_text'),
                        ('ky', 'ky_text')
                        #('left_bc', 'left_bc_text'),
                        # ('right_bc', 'right_bc_text')
                        ]:
            if not hasattr(self, attr):
                continue
            if param in mp.bc_values:
                text = str(mp.bc_values[param])
            else:
                text = str(getattr(mp, param, ''))
            getattr(self, attr).setText(text)
    
        self.model_params = mp
        self.model_results = None
        for btn in (self.show_geometry_button,self.show_mesh_button,
                    self.show_nodal_values_button, self.show_element_values_button):
            btn.setEnabled(False)
    
    def on_save_action(self):
        "Save model parameters to the current file or prompt for a new file"
        # Check if model_params in None or if update_model() fails
        if not self.model_params:
            QMessageBox.warning(self, 'Warning', 'Nothing to save or invalid data')
            return
        fn = getattr(self.model_params, 'filename', None)
        if fn:
            try:
                self.model_params.save(fn)
            except Exception:
                self.on_save_as_action()
        else:
            self.on_save_as_action()
    
    def on_save_as_action(self):
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
    
    def on_action_execute(self):
        "Run solver unless already executed; prompt to start new model if so"
        # Prevent re-execution
        if self.model_results is not None:
            QMessageBox.warning(
                self,
                'Execution Already Run',
                'To generate another domain create a new file'
            )
            return
        
        # Silently abort execution if parameters are missing
        if not self.update_model():
            return
        
        # Calculations not finished
        self.calc_done = False
        self.setEnabled(False)

        # Start solver thread
        self.model_results = fm.ModelResults()
        self.solver_thread = SolverThread(
            fm.ModelSolver(self.model_params, self.model_results)
        )
        self.solver_thread.finished.connect(self.on_solver_finished)
        self.solver_thread.start()
    
    def on_solver_finished(self):
        "Handle completion os the solver thread"
        self.setEnabled(True)

        # Calculation finished
        self.calc_done = True

        # Recreate visualization object
        self.visualization = fm.ModelVisualization(self.model_params, self.model_results)

        for btn in (self.show_geometry_button, self.show_mesh_button,
                    self.show_nodal_values_button, self.show_element_values_button):
            btn.setEnabled(True)
        
    def on_show_geometry(self):
        "Displat the geometry of the model"

        if not self.calc_done or self.visualization is None:
            QMessageBox.warning(self, 'No data', 'Please run calculation first')
            return
        self.visualization.show_geometry()

    def on_show_mesh(self):
        "Display the finite element mesh of the model"

        if not self.calc_done or self.visualization is None:
            QMessageBox.warning(self, 'No data', 'Please run calculation first')
            return
        self.visualization.show_mesh()
    
    def on_show_nodal_values(self):
        "Display the nodal values of the model"

        if not self.calc_done or self.visualization is None:
            QMessageBox.warning(self, 'No data', 'Please run calculation first')
            return
        self.visualization.show_nodal_values()
    
    def on_show_element_values(self):
        "Display the element values of the model"

        if not self.calc_done or self.visualization is None:
            QMessageBox.warning(self, 'No data', 'Please run calculation first')
            return
        self.visualization.show_element_values()
    
    def on_element_size_changed(self, value):
        "Update the element size factor based on the slider value"
        if self.model_params is None:
            self.model_params = fm.ModelParams()
        self.model_params.el_size_factor = value / 100.0
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())

        

   

    

