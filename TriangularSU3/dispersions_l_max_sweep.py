import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from time import perf_counter

from importlib import reload
import SU3_1hole_triangular
import SU3_2hole_triangular
reload(SU3_1hole_triangular)
reload(SU3_2hole_triangular)
from SU3_1hole_triangular import StringBasis as basis_tri_1h
from SU3_2hole_triangular import StringBasis as basis_tri_2h

import SU3_helper_sc_cc_overlaps
reload(SU3_helper_sc_cc_overlaps)  
from SU3_helper_sc_cc_overlaps  import *

honeycomb = True
system = 'SU2Hc' if honeycomb else 'SU3Tri'

#set system parameters
l_max = 12
n_bands = 1

L= 61
size = 2*np.pi/np.sqrt(3) * L/(L-1)

J = 0.3
j_perp = 0.3
t = 1
t2 = 0
unit_cell = 1

#relevant high symmetry points
K = 4*np.pi/(3*np.sqrt(3))*np.array([1, 0])
Kp = 2*np.pi/(3*np.sqrt(3))*np.array([1, np.sqrt(3)])
M = np.pi/3*np.array([np.sqrt(3), 1])
Gamma = np.array([0,0])

k_cc = Gamma
k_cc2 = K

# make k_grid
k1 = make_triangular_grid_bz(L, grid_size = 2*np.pi/np.sqrt(3) * L/(L-1))
k_shape = k1[0].shape[0]
# print(f'k shape:{k_shape}')
k2 = np.array([np.full(k_shape, k_cc2[0]), np.full(k_shape, k_cc2[1])])
k2 = k2 - k1
# print(f'k1 shape:{k1.shape}')
# print(f'k2 shape:{k2.shape}')

print("params: l_sc=", l_max, "t=", t, "J=", J, "honeycomb=", honeycomb, "unit_cell=", unit_cell)

# for l in range(2, l_max):

#     t0 = perf_counter()

#     basis_1 = basis_tri_1h(depth=l, only_connected=True, honeycomb=honeycomb, unit_cell=unit_cell)

#     # print(f"---------- calculating 2D single hole disperion for string length={l} ------------")

#     disp_1,_ = basis_1.dispersion(k_array=k1.T, two_D=False, j=J, t=t, t2=t2)
#     print(f'computed for k1')
#     disp_2,_ = basis_1.dispersion(k_array=k2.T, two_D=False, j=J, t=t, t2=t2)
#     # print(f'disp2 shape: {disp_2.shape}')
#     print(f'computed for k2')
#     print(f'computed 2D dispersion for depth={l} in {t:.3f}s'.format(t=perf_counter()-t0))


#     np.save(f"../results/TRI/{system}_2D_dispersion_trigrid_sc_depth={l}_t={t}_t2={t2}_j={J}_uc={unit_cell}.npy", disp_1)
#     np.save(f"../results/TRI/{system}_2D_dispersion_trigrid_k2_sc_depth={l}_t={t}_t2={t2}_j={J}_uc={unit_cell}.npy", disp_2)

# print("----------- finished calculations -----------------")

for l in range(2, l_max):
    basis_2 = basis_tri_2h(depth=l, only_connected=False, honeycomb=honeycomb, unit_cell=unit_cell)

    t0 = perf_counter()
    disp_bands_sc = []
    disp_bands_cc = []
    for n in range(n_bands):
        # disp_cut_sc,_ = basis_1.dispersion(k1, two_D=False, j=J, t=t, t2=t2, state=n)
        disp_cut_cc,_ = basis_2.dispersion(k1, two_D=False, j=J, t=t, t2=t2, state=n)
        disp_bands_cc.append(disp_cut_cc)
    disp_all_sc = np.array(disp_bands_sc)
    disp_all_cc = np.array(disp_bands_cc)
    print(f'computed 2D dispersion for depth={l} in {t:.3f}s'.format(t=perf_counter()-t0))
    np.save(f"../results/TRI/{system}_1D_dispersion_sc_path_GKMKpG_depth={l}_t={t}_t2={t2}_j={J}_uc={unit_cell}.npy", disp_bands_sc)
    np.save(f"../results/TRI/{system}_1D_dispersion_cc_path_GKMKpG_depth={l}_t={t}_t2={t2}_j={J}_uc={unit_cell}.npy", disp_bands_cc)

print("----------- finished calculations -----------------")

