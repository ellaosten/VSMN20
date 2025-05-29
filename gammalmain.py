# -*- coding: utf-8 -*-
import sys
import os

from qtpy.QtCore import QThread
from qtpy.QtWidgets import QApplication, QProgressDialog, QMainWindow, QFileDialog, QMessageBox, QWidget
from qtpy import uic
from qtpy.uic import loadUi
from qtpy.QtWidgets import QMessageBox

#import calfem.ui as cfui
import flowmodel_5 as fm
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import calfem.vis_mpl as cfv

def clean_ui(uifile):
    #"""Fix issues with Orientation:Horizontal/Vertical by creating _cleaned_mainwindow_5.ui"""
    #tree = ET.parse(uifile)
    #root = tree.getroot()
    #for enum in root.findall(".//property[@name='orientation']/enum"):
        #txt = enum.text or ''
        #if 'Orientation::Horizontal' in txt:
            #enum.text = 'Horizontal'
        #elif 'Orientation::Vertical' in txt:
            #enum.text = 'Vertical'
    clean_file = os.path.join(os.path.dirname(uifile), '_cleaned_mainwindow_5.ui')
    #tree.write(clean_file, encoding='utf-8', xml_declaration=True)
    return clean_file

class SolverThread(QThread):
    "Klass för att hantera beräkningar i bakgrunden"
    def __init__(self, solver, param_study = False):
        "Klasskonstruktion"
        super().__init__()
        self.param_study = param_study
        self.solver = solver
        
       # QThread.__init__(self) #gammal kod
        #self.solver = solver
       # self.param_study = param_study
        #self.update_controls()
    
        
    def run(self):
        "Kör beräkningarna i en separat tråd"
        if self.param_study:
            self.solver.execute_param_study()
        else:
            self.solver.execute()

class MainWindow(QMainWindow):
    "MainWindow-klass som hanterar vårt huvudfönster"
    def __init__(self):
        "Constructor for the main window"
        super(QMainWindow, self).__init__()

        # Visualization object
        self.visualization = None

        # Calculation not finished
        self.calc_done = False

        # Clean UI file and load interface description
        ui_path = os.path.join(os.path.dirname(__file__), 'mainwindow_5.ui')
        loadUi(clean_ui(ui_path), self)

        self.model_params = fm.ModelParams()
        self.model_results = fm.ModelResult()

        # Menu placement in ui window
        self.menuBar().setNativeMenuBar(False)

        # Element size slider
        self.element_size_label.setText('Element size:')
        self.element_size_slider.setRange(50, 100)

        # Set input placeholders including boundary fields
        placeholders = {
            'w_edit': '100.0 m',
            'h_edit': '10.0 m',
            'd_edit': '5.0 m',
            't_edit': '0.5 m',
            'kx_edit': '20.0 m/day',
            'ky_edit': '20.0 m/day',
            'left_bc_text': '10.0 mvp',
            'right_bc_text': '20.0 mvp',
            'd_end_edit': '9.0 m',
            't_end_edit': '5.0 m',
        }


        ### TA BORT
        # Set placeholder text for all QLineEdit widgets
        #for attr, text in placeholders.items():
            #if hasattr(self, attr):
                #widget = getattr(self, attr)
                #widget.clear()
                #widget.setPlaceholderText(text)
        
        # Set default values
        defaults = {
            'w_edit': str(self.model_params.w),
            'h_edit': str(self.model_params.h),
            'd_edit': str(self.model_params.d),
            't_edit': str(self.model_params.t),
            'kx_edit': str(self.model_params.kx),
            'ky_edit': str(self.model_params.ky),
            'left_bc_edit': str(self.model_params.bc_values['left_bc']),
            'right_bc_edit': str(self.model_params.bc_values['right_bc']),
            'd_end_edit': str(self.model_params.d_end),
            't_end_edit': str(self.model_params.t_end),
        }

        # Set default values in UI
        for attr, value in defaults.items():
            if hasattr(self, attr):
                getattr(self, attr).setText(value)

            self.element_size_slider.setValue(int(self.model_params.el_size_factor * 100))

        # Set default values for parameter study
        if hasattr(self, 'param_step'):
            self.param_step.setValue(5)

        # Clear checked radio buttons
        if hasattr(self, 'paramvarydradio'):
            self.paramvarydradio.setChecked(False)
        if hasattr(self, 'paramvarytradio'):
            self.paramvarytradio.setChecked(False)
        
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
        #self.on_action_exit.triggered.connect(self.on_action_exit)
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

        self.show()
        self.raise_()
    
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
       
        self.left_bc_text.setText(str(self.model_params.bc_values['left_bc']))
        self.right_bc_text.setText(str(self.model_params.bc_values['right_bc']))
        
        self.element_size_slider.setValue(int(self.model_params.el_size_factor))

        if self.paramvarydradio.isChecked():
            self.model_params.param_d = True

        if self.paramvarytradio.isChecked():
            self.model_params.param_t = True

    def update_model(self):
        "Load values from controller and update model"
        self.model_params.w = float(self.w_edit.text())
        self.model_params.h = float(self.h_edit.text())
        self.model_params.d = float(self.d_edit.text())
        self.model_params.t = float(self.t_edit.text())
        self.model_params.kx = float(self.kx_edit.text())
        self.model_params.ky = float(self.ky_edit.text())
        self.model_params.d_end = float(self.d_end_edit.text())
        self.model_params.t_end = float(self.t_end_edit.text())
        self.model_params.bc_values['left_bc'] = float(self.left_bc_text.text())
        self.model_params.bc_values['right_bc'] = float(self.right_bc_text.text())

        self.model_params.el_size_factor = self.element_size_slider.value() / 100.0

        if self.paramvarydradio.isChecked():
            self.model_params.param_d = True

        if self.paramvarytradio.isChecked():
            self.model_params.param_t = True

        # Define the mapping of UI fields to model parameters
        fields = [
             ('w_edit', 'w', 'Width of domain (w)'),
             ('h_edit', 'h', 'Height of domain (h)'),
             ('d_edit', 'd', 'Depth of barrier (d)'),
             ('t_edit', 't', 'Thickness of barrier (t)'),
             ('kx_edit', 'kx', 'Permeability in x-direction (kx)'),
             ('ky_edit', 'ky', 'Permeability in y-direction (ky)'),
             ('left_bc_text', 'left_bc', 'Left surface pressure (mvp)'),
             ('right_bc_text', 'right_bc', 'Right surface pressure (mvp)'),
        ]

        invalid = []

        # Warnings for invalid inputs
        if invalid:
             QMessageBox.warning(
                 self,
                 'Invalid Input',
                 'Please enter valid numbers for:\n' + '\n'.join(invalid)
             )
             return False


        
        
    
    # def update_model(self):
    #     "Read UI fields into model_params and update boundary conditions"
    #     # Ensure we have a ModelParams to write into
    #     if not self.model_params:
    #         self.model_params = fm.ModelParams()

    #     if self.paramvarydradio.isChecked():
    #         self.model_params.param_d = True

    #     if self.paramvarytradio.isChecked():
    #         self.model_params.param_t = True
    
    #     # Define the mapping of UI fields to model parameters
    #     fields = [
    #         ('w_edit', 'w', 'Width of domain (w)'),
    #         ('h_edit', 'h', 'Height of domain (h)'),
    #         ('d_edit', 'd', 'Depth of barrier (d)'),
    #         ('t_edit', 't', 'Thickness of barrier (t)'),
    #         ('kx_edit', 'kx', 'Permeability in x-direction (kx)'),
    #         ('ky_edit', 'ky', 'Permeability in y-direction (ky)'),
    #         #('left_bc_text', 'left_bc', 'Left surface pressure (mvp)'),
    #         #('right_bc_text', 'right_bc', 'Right surface pressure (mvp)'),
    #     ]

    #     invalid = []

    #     # Read values from UI fields and set them in model_params
    #     for widget_name, param_name, label in fields:
    #         widget = getattr(self, widget_name, None)
    #         txt = widget.text().strip() if widget else ''
    #         try:
    #             value = float(txt)
    #         except Exception:
    #             invalid.append(label)
    #         else:
    #             setattr(self.model_params, param_name, value)
    
    #     # Warnings for invalid inputs
    #     if invalid:
    #         QMessageBox.warning(
    #             self,
    #             'Invalid Input',
    #             'Please enter valid numbers for:\n' + '\n'.join(invalid)
    #         )
    #         return False
        
    #     # Properties
    #     mp = self.model_params
    #     mp.D = np.array([[mp.kx, 0], [0, mp.ky]]) 
    #     mp.el_size_factor = self.element_size_slider.value() / 100.0
    #     return True
    
    def handle_new_action(self):
    #Återställer fält och förbereder nytt tomt projekt
        self.reset_ui()
    
    # ----- RESET_UI -----
    def reset_ui(self):
        "Återställ gränssnittet och förifyll standardvärden"
        for attr in ['w_edit', 'h_edit', 'd_edit', 't_edit', 'kx_edit', 'ky_edit']:
            widget = getattr(self, attr, None)
            if widget:
                widget.clear()

        self.element_size_slider.setValue(50)

        for btn in (self.show_geometry_button,
                    self.show_mesh_button,
                    self.show_nodal_values_button,
                    self.show_element_values_button):
            btn.setEnabled(False)

        self.model_params = None
        self.model_results = None
        self.visualization = None
        self.calc_done = False

        # Fyll i rimliga standardvärden
        standard_values = {
            'w_edit': '100.0',
            'h_edit': '10.0',
            'd_edit': '5.0',
            't_edit': '0.5',
            'kx_edit': '20.0',
            'ky_edit': '20.0',
        }
        for attr, value in standard_values.items():
            widget = getattr(self, attr, None)
            if widget:
                widget.setText(value)
    
    def handle_open_action(self):
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
                        ('w', 'w_edit'),
                        ('h', 'h_edit'),
                        ('d', 'd_edit'),
                        ('t', 't_edit'),
                        ('kx', 'kx_edit'),
                        ('ky', 'ky_edit'),
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
    
    def handle_save_action(self):
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
            print("==> update_model() returnerade False")
            return
        print("==> startar beräkning")
        
        # Calculations not finished
        self.calc_done = False
        self.setEnabled(False)

        # Start solver thread
        self.model_results = fm.ModelResult()
        self.solver_thread = SolverThread(
            fm.ModelSolver(self.model_params, self.model_results)
        )
        self.solver_thread.finished.connect(self.on_solver_finished)
        self.solver_thread.start()
    
    def on_solver_finished(self):
        print("==> on_solver_finished körs")
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
        print("==> on_show_geometry körs")
        "Displat the geometry of the model"

        if not self.calc_done:
            print("Beräkning inte klar!")
        if self.visualization is None:
            print("Ingen visualisering hittades!")

        if not self.calc_done or self.visualization is None:
            QMessageBox.warning(self, 'No data', 'Please run calculation first')
            return
        
        print("Visar geometrin nu...")
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
    
    def on_execute_param_study(self):
        "Run a parameter study either on depth (d) or thickness (t)"
        ###############################################################
        # Update model from UI
        if not self.update_model():
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numbers')
            return
        
        # Update filename
        self.model_params.params_filename = "param_study"
        self.model_results = fm.ModelResult()

        # Skapa en lösare
        self.solver = fm.ModelSolver(self.model_params, self.model_results)

        # Starta en tråd för att köra beräkningen, så att gränssnittet inte fryser
        self.solverThread = SolverThread(self.solver, param_study=True)
        self.solverThread.finished.connect(self.on_solver_finished)
        self.solverThread.start()        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())