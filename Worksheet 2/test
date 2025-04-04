import numpy as np 
import calfem.core as cfc
import calfem.utils as cfu
import tabulate as tab
import json

ex = np.array ([
    [0.0, 0.0, 600],
    [0.0, 600.0, 600.0],
    [600.0, 600.0, 1200.0],
    [600.0, 1200.0, 1200.0],
])
ey = np.array ([
    [0.0, 600.0, 600.0],
    [0.0, 600.0, 0.0],
    [0.0, 600.0, 600.0],
   [0.0, 600.0, 0.0], 
])
ep = [1.0]
ed = np.array ([
    [0.0, 60.0, 60.0],
    [0.0, 60.0, 0.0],
    [0.0, 60.0, 0.0],
    [0.0, 0.0, 0.0],
])
D = np.array([
    [50.0, 0.0],
    [0.0, 50.0],
])
edof = np.array ([
            [1, 2, 4],
            [1, 4, 3],
            [3, 4, 6],
            [3, 6, 5],
        ])
for elx, ely, eltopo in zip(ex, ey, edof):
    Ke = cfc.flw2te(elx, ely, ep, D)

    print(Ke)

