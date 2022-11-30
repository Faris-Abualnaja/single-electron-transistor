# !/usr/bin/env python3
# This runs on python3
# Master equation simulation of a basic SET
# The gui is made using PyQt6
# By: Dr Faris Abualnaja

# A list of Qt widgets can be found in the following link
# https://doc.qt.io/qtforpython/PySide6/QtWidgets/index.html

# ---- Import required libraries ----

import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QPalette, QColor
from matplotlib import projections
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

# ---- Global variables ----
q0 = 0            # Initial charge
q  = 1.602e-19    # Electron charge (C)
k  = 1.381e-23    # Boltzmann constant (m^2kg/s^2K)

N_min = -20
N_max =  20

T1p = np.arange(0.0, N_max-N_min)
T1n = np.arange(0.0, N_max-N_min)
T2p = np.arange(0.0, N_max-N_min)
T2n = np.arange(0.0, N_max-N_min)

mega = 1000000
atto = 1.0e-18

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("SET Simulator (Master Equation Method)")

        # self.setFixedSize(500, 700)

        # Create the grid layout for app
        layout = QGridLayout()
        # Set spacing between grid margins
        layout.setContentsMargins(10,10,10,10)
        # Set spacing between grid elements
        layout.setSpacing(10)

        # Add labels to grid (html syntax)
        layout.addWidget(QLabel('C<sub>1</sub> (aF)'),   2, 0)
        layout.addWidget(QLabel('C<sub>2</sub> (aF)'),   2, 1)
        layout.addWidget(QLabel('C<sub>g</sub> (aF)'),   2, 2)
        layout.addWidget(QLabel('R<sub>1</sub> (MOhm)'), 4, 0)
        layout.addWidget(QLabel('R<sub>2</sub> (MOhm)'), 4, 1)
        layout.addWidget(QLabel('Temperature (K)'),      4, 2)
        layout.addWidget(QLabel('V<sub>d</sub> start'),  6, 0)
        layout.addWidget(QLabel('V<sub>d</sub> end'),    6, 1)
        layout.addWidget(QLabel('V<sub>d</sub> step'),   6, 2)
        layout.addWidget(QLabel('V<sub>g</sub> start'),  8, 0)
        layout.addWidget(QLabel('V<sub>g</sub> end'),    8, 1)
        layout.addWidget(QLabel('V<sub>g</sub> step'),   8, 2)

        # Create entries
        self.C1  = QLineEdit('   1')
        self.C2  = QLineEdit('   1')
        self.Cg  = QLineEdit(' 0.1')
        self.R1  = QLineEdit('   1')
        self.R2  = QLineEdit('   1')
        self.T   = QLineEdit('   1')
        self.Vdi = QLineEdit('-0.2')
        self.Vdf = QLineEdit(' 0.2')
        self.Vds = QLineEdit('  20')
        self.Vgi = QLineEdit('  -2')
        self.Vgf = QLineEdit('   2')
        self.Vgs = QLineEdit('  20')

        # Add entries to grid
        layout.addWidget(self.C1 , 3, 0)
        layout.addWidget(self.C2 , 3, 1)
        layout.addWidget(self.Cg , 3, 2)
        layout.addWidget(self.R1 , 5, 0)
        layout.addWidget(self.R2 , 5, 1)
        layout.addWidget(self.T  , 5, 2)
        layout.addWidget(self.Vdi, 7, 0)
        layout.addWidget(self.Vdf, 7, 1)
        layout.addWidget(self.Vds, 7, 2)
        layout.addWidget(self.Vgi, 9, 0)
        layout.addWidget(self.Vgf, 9, 1)
        layout.addWidget(self.Vgs, 9, 2)

        # Create buttons
        self.Osc  = QPushButton('Oscillations')
        self.Str  = QPushButton('Staircases')
        self.Dmd  = QPushButton('Diamonds')
        

        # Add buttons to grid
        layout.addWidget(self.Osc, 10, 0)
        layout.addWidget(self.Str, 10, 1)
        layout.addWidget(self.Dmd, 10, 2)

        # Check if Osc button is pressed
        self.Osc.released.connect(self.Coulomb_Oscillations)
        # Check if Str button is pressed
        self.Str.released.connect(self.Coulomb_Staircases)
        # Check if Dmd button is pressed
        self.Dmd.released.connect(self.Coulomb_Diamonds)

        # Create progress bar to see how long it
        # takes to calculate and plot     
        self.progressBar = QProgressBar()
        self.progressBar.setMinimum(0)

        self.statusbar = QStatusBar()
        self.statusbar.showMessage('Ready')

        layout.addWidget(self.statusbar, 11, 0, 1, 1)
        layout.addWidget(self.progressBar, 11, 1, 1, 2) 

        # Create plot and add to grid
        self.fig = Figure(figsize=(4, 4), dpi=80)
        self.canvas = FigureCanvas(self.fig)
        self.a = self.canvas.figure.add_subplot()
        self.cbExists = 0
        layout.addWidget(self.canvas, 0, 0, 1, 3)
        # Create frame for navigation toolbar
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Raised)
        layout.addWidget(frame, 1,0,1,3)
        # Navigation tools from matplotlib
        layout.addWidget(NavigationToolbar(self.canvas, self), 1, 0, 1, 3)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def Coulomb_Oscillations(self):
        # Clear plot
        self.a.remove()
        # Remove colorbar if it exists
        if self.cbExists:
            self.cb.remove()
            self.cbExists = 0
        # Reinstate figure
        self.a = self.canvas.figure.add_subplot()

        try:
            C1 = float(self.C1.text())*atto
            C2 = float(self.C2.text())*atto
            Cg = float(self.Cg.text())*atto
            R1 = float(self.R1.text())*mega
            R2 = float(self.R2.text())*mega
            T  = float(self.T.text())
            Vg_min = float(self.Vgi.text())
            Vg_max = float(self.Vgf.text())
            Vd_min = float(self.Vdi.text())
            Vd_max = float(self.Vdf.text())
            Vg_steps = int(self.Vgs.text())
            Vd_steps = int(self.Vds.text())
        except:
            print('Failed')
        finally:
            Ctotal = C1 + C2 + Cg

            Vg_values =  (Vg_max-Vg_min)/Vg_steps
            Vd_values =  (Vd_max-Vd_min)/Vd_steps

            I = np.zeros(shape=(Vd_steps, Vg_steps))

            # Coloumb oscillations
            variations = Vd_steps # How many drain voltage variations

            self.progressBar.setMaximum(variations-1)
            
            I  = np.zeros(Vg_steps)
            Vg = np.zeros(Vg_steps)
            Vd = np.zeros(variations)

            # Loop through all Vd values
            for i in range(0, variations):
                Vd[i] = Vd_min + i*Vd_values

                # Loop through all Vg points
                for j in range(0, Vg_steps):
                    Vg[j] = Vg_min + j*Vg_values

                    # Loop through all possible charge states N
                    for N in range(0, (N_max-N_min)):
                        # N charge number in dot
                        n = N_min+N
                        # Calculation of ∆F across tunnel junction 1
                        dF1p = q/Ctotal*(0.5*q+(n*q-q0)-(C2+Cg)*Vd[i]+Cg*Vg[j])
                        dF1n = q/Ctotal*(0.5*q-(n*q-q0)+(C2+Cg)*Vd[i]-Cg*Vg[j])
                        # Calculation of ∆F across tunnel junction 2
                        dF2p = q/Ctotal*(0.5*q-(n*q-q0)-C1*Vd[i]-Cg*Vg[j])
                        dF2n = q/Ctotal*(0.5*q+(n*q-q0)+C1*Vd[i]+Cg*Vg[j])
                        # Tunnel-rate 1 (p) calculations depending on ∆F
                        if dF1p < 0.0:
                            T1p[N] = 1/(R1*q*q)*(-dF1p)/(1-np.exp(dF1p/(k*T),
                            dtype=np.longdouble))
                        else:
                            T1p[N] = 1e-9
                        
                        # Tunnel-rate 1 (n) calculations depending on ∆F
                        if dF1n < 0.0:
                            T1n[N] = 1/(R1*q*q)*(-dF1n)/(1-np.exp(dF1n/(k*T),
                            dtype=np.longdouble))
                        else:
                            T1n[N] = 1e-9
                        
                        # Tunnel-rate 2 (p) calculations depending on ∆F
                        if dF2p < 0.0:
                            T2p[N] = 1/(R1*q*q)*(-dF2p)/(1-np.exp(dF2p/(k*T),
                            dtype=np.longdouble))
                        else:
                            T2p[N] = 1e-9
                        
                        # Tunnel-rate 2 (n) calculations depending on ∆F
                        if dF2n < 0.0:
                            T2n[N] = 1/(R1*q*q)*(-dF2n)/(1-np.exp(dF2n/(k*T),
                            dtype=np.longdouble))
                        else:
                            T2n[N] = 1e-9
                        
                        # Ideally, N is from -∞ to +∞ w/ boundary conditions p[min] = p[max] = 0
                        p = np.arange(0.0, N_max-N_min)
                        p[0] = 0.0
                        p[(N_max - N_min)-1] = 0.0
                        # Initial sum value to calculate ρ
                        Sum = 0.0
                        for N in range(1,(N_max - N_min)-2):
                            # Calculation of ρ(N)
                            p[N] = p[N-1]*(T2n[N-1]+T1p[N-1])/(T2p[N]+T1n[N])
                            # Conditions below are used to avoid divergence in Python-3
                            if p[N] > 1.0e323:
                                p[N] = 1.0e323
                            elif p[N] < 1.0e-323:
                                p[N] = 1.0e-323
                            Sum = Sum+p[N]

                        if Sum > 1.0e300:
                            Sum = 1.0e300
                        elif Sum < 1.0e-300:
                            Sum = 1.0e-300

                        # Normalisation
                        for N in range(0, (N_max-N_min)-1):
                            p[N] = p[N]/Sum
                        
                        # Initial condition for current calculation
                        sumI = 0.0

                        for N in range(0, (N_max-N_min)-1):
                            sumI = sumI + p[N]*(T2p[N]-T2n[N])
                        
                        # Current at each Vg point put into array
                        I[j] = (q*sumI)*10**9
                
                self.statusbar.showMessage('Calculating...')
                self.progressBar.setValue(i)

                # Plot.
                self.a.plot(Vg, I)
            
            self.a.set_xlabel('$V_{gs}$ (V)')
            self.a.set_ylabel('$I_{ds}$ (nA)')

        self.canvas.draw()
        self.statusbar.showMessage('Ready')
        self.progressBar.setValue(0)
            
        return
    
    def Coulomb_Staircases(self):
        # Clear plot
        self.a.remove()
        # Remove colorbar if it exists
        if self.cbExists:
            self.cb.remove()
            self.cbExists = 0
        # Reinstate figure
        self.a = self.canvas.figure.add_subplot()

        try:
            C1 = float(self.C1.text())*atto
            C2 = float(self.C2.text())*atto
            Cg = float(self.Cg.text())*atto
            R1 = float(self.R1.text())*mega
            R2 = float(self.R2.text())*mega
            T  = float(self.T.text())
            Vg_min = float(self.Vgi.text())
            Vg_max = float(self.Vgf.text())
            Vd_min = float(self.Vdi.text())
            Vd_max = float(self.Vdf.text())
            Vg_steps = int(self.Vgs.text())
            Vd_steps = int(self.Vds.text())
        except:
            print('Failed')
        finally:
            Ctotal = C1 + C2 + Cg

            Vg_values =  (Vg_max-Vg_min)/Vg_steps
            Vd_values =  (Vd_max-Vd_min)/Vd_steps

            I = np.zeros(shape=(Vd_steps, Vg_steps))

            # Coloumb oscillations
            variations = Vg_steps # How many drain voltage variations

            self.progressBar.setMaximum(variations-1)
            
            I  = np.zeros(Vd_steps)
            Vg = np.zeros(variations)
            Vd = np.zeros(Vd_steps)

        # Loop through all Vd values
        for i in range(0, variations):
            Vg[i] = Vg_min + i*Vg_values

            # Loop through all Vg points
            for j in range(0, Vd_steps):
                Vd[j] = Vd_min + j*Vd_values

                # Loop through all possible charge states N
                for N in range(0, (N_max-N_min)):
                    # N charge number in dot
                    n = N_min+N
                    # Calculation of ∆F across tunnel junction 1
                    dF1p = q/Ctotal*(0.5*q+(n*q-q0)-(C2+Cg)*Vd[j]+Cg*Vg[i])
                    dF1n = q/Ctotal*(0.5*q-(n*q-q0)+(C2+Cg)*Vd[j]-Cg*Vg[i])
                    # Calculation of ∆F across tunnel junction 2
                    dF2p = q/Ctotal*(0.5*q-(n*q-q0)-C1*Vd[j]-Cg*Vg[i])
                    dF2n = q/Ctotal*(0.5*q+(n*q-q0)+C1*Vd[j]+Cg*Vg[i])
                    # Tunnel-rate 1 (p) calculations depending on ∆F
                    if dF1p < 0.0:
                        T1p[N] = 1/(R1*q*q)*(-dF1p)/(1-np.exp(dF1p/(k*T),
                        dtype=np.longdouble))
                    else:
                        T1p[N] = 1e-9
                    
                    # Tunnel-rate 1 (n) calculations depending on ∆F
                    if dF1n < 0.0:
                        T1n[N] = 1/(R1*q*q)*(-dF1n)/(1-np.exp(dF1n/(k*T),
                        dtype=np.longdouble))
                    else:
                        T1n[N] = 1e-9
                    
                    # Tunnel-rate 2 (p) calculations depending on ∆F
                    if dF2p < 0.0:
                        T2p[N] = 1/(R1*q*q)*(-dF2p)/(1-np.exp(dF2p/(k*T),
                        dtype=np.longdouble))
                    else:
                        T2p[N] = 1e-9
                    
                    # Tunnel-rate 2 (n) calculations depending on ∆F
                    if dF2n < 0.0:
                        T2n[N] = 1/(R1*q*q)*(-dF2n)/(1-np.exp(dF2n/(k*T),
                        dtype=np.longdouble))
                    else:
                        T2n[N] = 1e-9
                    
                    # Ideally, N is from -∞ to +∞ w/ boundary conditions p[min] = p[max] = 0
                    p = np.arange(0.0, N_max-N_min)
                    p[0] = 0.0
                    p[(N_max - N_min)-1] = 0.0
                    # Initial sum value to calculate ρ
                    Sum = 0.0
                    for N in range(1,(N_max - N_min)-2):
                        # Calculation of ρ(N)
                        p[N] = p[N-1]*(T2n[N-1]+T1p[N-1])/(T2p[N]+T1n[N])
                        # Conditions below are used to avoid divergence in Python-3
                        if p[N] > 1.0e323:
                            p[N] = 1.0e323
                        elif p[N] < 1.0e-323:
                            p[N] = 1.0e-323
                        Sum = Sum+p[N]

                    if Sum > 1.0e300:
                        Sum = 1.0e300
                    elif Sum < 1.0e-300:
                        Sum = 1.0e-300

                    # Normalisation
                    for N in range(0, (N_max-N_min)-1):
                        p[N] = p[N]/Sum
                    
                    # Initial condition for current calculation
                    sumI = 0.0

                    for N in range(0, (N_max-N_min)-1):
                        sumI = sumI + p[N]*(T2p[N]-T2n[N])
                    
                    # Current at each Vg point put into array
                    I[j] = (q*sumI)*10**9
                    
            self.statusbar.showMessage('Calculating...')
            self.progressBar.setValue(i)
            # Plot.
            self.a.plot(Vd, I)
        
        self.a.set_xlabel('$V_{ds}$ (V)')
        self.a.set_ylabel('$I_{ds}$ (nA)')

        self.canvas.draw()

        self.statusbar.showMessage('Ready')
        self.progressBar.setValue(0)
                    
        return

    def Coulomb_Diamonds(self):
        # Clear plot
        self.a.remove()
        # Resinstate figure, but 3d projection
        self.a = self.canvas.figure.add_subplot(projection='3d')
        
        try:
            C1 = float(self.C1.text())*atto
            C2 = float(self.C2.text())*atto
            Cg = float(self.Cg.text())*atto
            R1 = float(self.R1.text())*mega
            R2 = float(self.R2.text())*mega
            T  = float(self.T.text())
            Vg_min = float(self.Vgi.text())
            Vg_max = float(self.Vgf.text())
            Vd_min = float(self.Vdi.text())
            Vd_max = float(self.Vdf.text())
            Vg_steps = int(self.Vgs.text())
            Vd_steps = int(self.Vds.text())
        except:
            print('Failed')
        finally:
            Ctotal = C1 + C2 + Cg

            Vg_values =  (Vg_max-Vg_min)/Vg_steps
            Vd_values =  (Vd_max-Vd_min)/Vd_steps

            I = np.zeros(shape=(Vd_steps, Vg_steps))

            Vg = np.zeros(Vg_steps)
            Vd = np.zeros(Vd_steps)

            self.progressBar.setMaximum(Vd_steps-1)

            # Loop through all Vd values
            for i in range(0, Vd_steps):
                Vd[i] = Vd_min + i*Vd_values

                # Loop through all Vg points
                for j in range(0, Vg_steps):
                    Vg[j] = Vg_min + j*Vg_values

                    # Loop through all possible charge states N
                    for N in range(0, (N_max-N_min)):
                        # N charge number in dot
                        n = N_min+N
                        # Calculation of ∆F across tunnel junction 1
                        dF1p = q/Ctotal*(0.5*q+(n*q-q0)-(C2+Cg)*Vd[i]+Cg*Vg[j])
                        dF1n = q/Ctotal*(0.5*q-(n*q-q0)+(C2+Cg)*Vd[i]-Cg*Vg[j])
                        # Calculation of ∆F across tunnel junction 2
                        dF2p = q/Ctotal*(0.5*q-(n*q-q0)-C1*Vd[i]-Cg*Vg[j])
                        dF2n = q/Ctotal*(0.5*q+(n*q-q0)+C1*Vd[i]+Cg*Vg[j])
                        # Tunnel-rate 1 (p) calculations depending on ∆F

                        if (dF1p) < 0.0:
                            T1p[N] = 1/(R1*q*q)*(-dF1p)/(1-np.exp(dF1p/(k*T),
                            dtype=np.longdouble))
                        else:
                            T1p[N] = 1e-9
                        
                        # Tunnel-rate 1 (n) calculations depending on ∆F
                        if (dF1n) < 0.0:
                            T1n[N] = 1/(R1*q*q)*(-dF1n)/(1-np.exp(dF1n/(k*T),
                            dtype=np.longdouble))
                        else:
                            T1n[N] = 1e-9
                        
                        # Tunnel-rate 2 (p) calculations depending on ∆F
                        if (dF2p) < 0.0:
                            T2p[N] = 1/(R1*q*q)*(-dF2p)/(1-np.exp(dF2p/(k*T),
                            dtype=np.longdouble))
                        else:
                            T2p[N] = 1e-9
                        
                        # Tunnel-rate 2 (n) calculations depending on ∆F
                        if (dF2n) < 0.0:
                            T2n[N] = 1/(R1*q*q)*(-dF2n)/(1-np.exp(dF2n/(k*T),
                            dtype=np.longdouble))
                        else:
                            T2n[N] = 1e-9
                        
                        # Ideally, N is from -∞ to +∞ w/ boundary conditions p[min] = p[max] = 0
                        p = np.arange(0.0, N_max-N_min)
                        p[0] = 0.0
                        p[(N_max - N_min)-1] = 0.0
                        # Initial sum value to calculate ρ
                        Sum = 0.0
                        for N in range(1,(N_max - N_min)-2):
                            # Calculation of ρ(N)
                            p[N] = p[N-1]*(T2n[N-1]+T1p[N-1])/(T2p[N]+T1n[N])
                            # Conditions below are used to avoid divergence in Python-3
                            if p[N] > 1.0e323:
                                p[N] = 1.0e323
                            elif p[N] < 1.0e-323:
                                p[N] = 1.0e-323
                            Sum = Sum+p[N]

                        if Sum > 1.0e300:
                            Sum = 1.0e300
                        elif Sum < 1.0e-300:
                            Sum = 1.0e-300

                        # Normalisation
                        for N in range(0, (N_max-N_min)-1):
                            p[N] = p[N]/Sum
                        
                        # Initial condition for current calculation
                        sumI = 0.0

                        for N in range(0, (N_max-N_min)-1):
                            sumI = sumI + p[N]*(T2p[N]-T2n[N])
                        
                        # Current at each Vg point put into array
                        I[i,j] = (q*sumI)*10**9
            
                self.statusbar.showMessage('Calculating...')
                self.progressBar.setValue(i)

        Vgm, Vdm = np.meshgrid(Vg, Vd)
  
        # Plot.
        surf = self.a.plot_surface(Vgm, Vdm, I, cmap = matplotlib.cm.jet,
                                    linewidth = 0.5, edgecolors = 'k',
                                    antialiased = True)
        self.a.view_init(elev = 30, azim = -45)
        self.a.set_xlabel('$V_{g}$ (V)' , labelpad = 20)
        self.a.set_ylabel('$V_{d}$ (V)' , labelpad = 20)
        #self.a.set_zlabel('$I_{d}$ (nA)', labelpad = 20)

        # Add a color bar which maps values to colors.
        if self.cbExists:
            self.cb.remove()
            self.cbExists = 0
        
        self.cb = self.fig.colorbar(surf, shrink = 0.5, aspect = 20, pad = 0.1)
        self.cb.set_label('$I_{d}$ (nA)')
        self.cbExists = 1

        self.canvas.draw()
        self.statusbar.showMessage('Ready')
        self.progressBar.setValue(0)

        return

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
