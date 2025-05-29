# Steps from UI
        n_steps = self.param_step.value()
        if n_steps < 2:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter a number of steps greater than 1')
            return
        
        vals = np.linspace(start_val, end_val, n_steps)

        # Progress dialog
        progress = QProgressDialog(
            f"Running parameter study for {var_name}",
            "Abort", 0, n_steps-1, self
        )
        progress.setWindowTitle("Please wait")
        progress.setWindowModality(Qt.WindowModal) ## ??
        progress.setMinimumDuration(0)
        progress.show()

        # Prepare for plotting
        self.plainTextEdit.clear()
        flows = []

        # Abort button
        for i, v in enumerate(vals):
            progress.setValue(i)
            QApplication.processEvents()
            if progress.wasCanceled():
                break

            # Copy all other params into a fresh ModelParams
            base = self.model_params
            p = fm.ModelParams()
            p.w = base.w
            p.h = base.h
            p.kx = base.kx
            p.ky = base.ky
            p.el_size_factor = base.el_size_factor
            p.d = v if var_name == 'd' else base.d
            p.t = v if var_name == 't' else base.t
            p.bc_markers = base.bc_markers
            p.bc_values = base.bc_values.copy()
            p.load_markers = base.load_markers
            p.load_values = base.load_values.copy()
        
            # Solve the model
            mr = fm.ModelResult()
            solver = fm.ModelSolver(p, mr)
            solver.execute()

            mf = mr.max_element_flow()
            flows.append(mf)

            # Log into the plainTextEdit
            self.plainTextEdit.append(f"{var_name} = {v:.4g}, max flow = {mf:.2f}")
        
        progress.setValue(n_steps-1)
        progress.close()

        cfv.figure()
        plt.clf()
        plt.plot(vals[:len(flows)], flows)
        plt.xlabel(xlabel)
        plt.ylabel('Max element flow (m3/day)')
        plt.title(f'Parameter study: {xlabel} vs Max element flow')
        plt.grid(True)
        cfv.show()









# Update model from UI
        if not self.update_model():
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numbers')
            return
        
        # Update filename
        self.model_params.params_filename = "param_study"

        # Decide which parameter to vary
        if self.paramvarydradio.isChecked():
            var_name = 'd'
            start_val = self.model_params.d
            try:
                end_val = float(self.d_end_edit.text())
            except ValueError:
                QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid number for depth end value')
                return
            xlabel = 'Barrier depth (d)'
        elif self.paramvarytradio.isChecked():
            var_name = 't'
            start_val = self.model_params.t
            try:
                end_val = float(self.t_end_edit.text())
            except ValueError:
                QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid number for thickness end value')
                return
            xlabel = 'Barrier thickness (t)'
        else:
            QMessageBox.warning(self, 'Parameter study', 'Please check d or t to enable a sweep')
            return
        
        # Steps from UI
        n_steps = self.param_step.value()
        if n_steps < 2:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter a number of steps greater than 1')
            return
        
        vals = np.linspace(start_val, end_val, n_steps)

        # Progress dialog
        progress = QProgressDialog(
            f"Running parameter study for {var_name}",
            "Abort", 0, n_steps-1, self
        )
        progress.setWindowTitle("Please wait")
        progress.setWindowModality(Qt.WindowModal) ## ??
        progress.setMinimumDuration(0)
        progress.show()

        # Prepare for plotting
        self.plainTextEdit.clear()
        flows = []

        # Abort button
        for i, v in enumerate(vals):
            progress.setValue(i)
            QApplication.processEvents()
            if progress.wasCanceled():
                break

            # Copy all other params into a fresh ModelParams
            base = self.model_params
            p = fm.ModelParams()
            p.w = base.w
            p.h = base.h
            p.kx = base.kx
            p.ky = base.ky
            p.el_size_factor = base.el_size_factor
            p.d = v if var_name == 'd' else base.d
            p.t = v if var_name == 't' else base.t
            p.bc_markers = base.bc_markers
            p.bc_values = base.bc_values.copy()
            p.load_markers = base.load_markers
            p.load_values = base.load_values.copy()
        
            # Solve the model
            mr = fm.ModelResult()
            solver = fm.ModelSolver(p, mr)
            solver.execute()

            mf = mr.max_element_flow()
            flows.append(mf)

            # Log into the plainTextEdit
            self.plainTextEdit.append(f"{var_name} = {v:.4g}, max flow = {mf:.2f}")
        
        progress.setValue(n_steps-1)
        progress.close()

        cfv.figure()
        plt.clf()
        plt.plot(vals[:len(flows)], flows)
        plt.xlabel(xlabel)
        plt.ylabel('Max element flow (m3/day)')
        plt.title(f'Parameter study: {xlabel} vs Max element flow')
        plt.grid(True)
        cfv.show()


            # Update model parameter
            #setattr(self.model_params, var_name, v)
            #self.model_params.params_filename = f"param_study_{var_name}_{v:.2f}"
            
            # Run solver
            #self.model_results = fm.ModelResult()
            #self.solver = fm.ModelSolver(self.model_params, self.model_results)
            #self.solver.execute()

            # Store results for plotting
            #flows.append(self.model_results.q)