'''compute overlap between 2 sc-pairs and cc'''
import numpy as np
import matplotlib.pyplot as plt
import os.path
from scipy.sparse import csr_matrix
from time import perf_counter
import copy
import cmath
from importlib import reload
import SU3_2hole_triangular
reload(SU3_2hole_triangular)  
from SU3_2hole_triangular import StringBasis as Lat_cc
from SU3_2hole_triangular import sorted_list

import SU3_1hole_triangular
reload(SU3_1hole_triangular)  
from SU3_1hole_triangular import StringBasis as Lat_sc1
import SU3_1hole_triangular2
reload(SU3_1hole_triangular2)  
from SU3_1hole_triangular2 import StringBasis as Lat_sc2

def index2momentum(i, Lx, Ly=0, size=np.pi):    #returns momentum grid of size L, what is i?
    if Ly==0:
        L = Lx
    else:
        assert i.shape[0] == 2
        L = np.array([Lx, Ly]).reshape((2,) + (len(i.shape)-1)*(1,))
    return (size*(2*i/(L)-1)) 

def sum_ind(i, j, Lx, Ly=0):
    if Ly==0:
        L = Lx
    else:
        assert i.shape[0] == 2
        assert j.shape[0] == 2
        L = np.array([Lx, Ly]).reshape((2,) + (len(i.shape)-1)*(1,))
    return np.mod(np.round(i+j-L/2), L).astype(int)

def find_l0_state(lat: Lat_sc1, n_max=7): #lower n_max for testing
    '''
    Find the first energy level which has considerable quasiparticle weight i.e. overlap with the zero string-length state.
                #cc or sc?                                                                   cc or sc?
    ARGUMENTS:
    lat: Lat_sc: an instance of the sc lattice class, where the Hamiltonian has already been computed.
    n_max : int (optional): the highest energy level considered

    OUTPUT:
    n: int: gives the number of the first energy level correspondig to the l=0 state. (n=0 corresponds to the ground state)
    '''
    _, vs = lat.eigensys(n_max, full=True) # lat is Lat_sc, eigenstates up to n_max=7
    vs = np.abs(vs[:2, :])
    n = 0
    while n <= n_max:
        if vs[0, n] > vs[1, n] and vs[0, n] > 1e-4:
            break
        else:
            n += 1
    return n

def init_lattices(l_max_sc, l_max_cc, l_max_sc_overlaps, j_perp_div_j=1., connected=True, honeycomb=False, basis2=False):
    """
    initialize and construct truncated basis for sc and cc.
    save indices of sc_basis, where string is smaller (or equal) than l_max_sc_overlaps
    Also compute Hamiltonian matrix.
    ------------
    ARGUMENTS:
    l_max_sc (int): maximal string length in sc truncated basis
    l_max_cc (int): maximal string length in cc truncated basis
    l_max_sc_overlaps (int): maximal string length considered in overlaps
    connected (bool): if True, we take only the connected strings into account
    -------------
    RETURNS:
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    """
    lat_sc1 = Lat_sc1(l_max_sc, only_connected=connected, honeycomb=honeycomb, basis2=basis2)
    lat_sc2 = Lat_sc2(l_max_sc, only_connected=connected, honeycomb=honeycomb)
    #lat_sc.generate_basis()
    #lat_sc.matrix_el(1., 1., j_perp_div_j, 1.) #t=1, j_z=1, t2=1, j_perp=j_perp_div_j, why?
    lat_sc1.matrix_el() 
    lat_sc2.matrix_el() 

    #lat_sc.indices = list(np.argwhere(np.count_nonzero([x[0] for x in lat_sc.bin_basis], axis=(1,2)) <= l_max_sc_overlaps).flatten()) #makes list of indices(position in basis) of all states that have strings <= l_max_sc_overlaps
    lat_sc1.indices1 = list(np.argwhere(np.array([len(x['seq']) for x in lat_sc1.bin_basis]) <= l_max_sc_overlaps).flatten()) # linus
    lat_sc2.indices2 = list(np.argwhere(np.array([len(x['seq']) for x in lat_sc2.bin_basis]) <= l_max_sc_overlaps).flatten()) # linus

    lat_cc = Lat_cc(l_max_cc, only_connected=connected, honeycomb=honeycomb)
    lat_cc.matrix_el()

    print('initialized lattices')
    # print(f'dim(H_sc) = {lat_sc.basis.length}')
    # print(f'dim(H_cc) = {lat_cc.basis.length}')

    return lat_sc1, lat_sc2, lat_cc

def compute_eigenvectors(L, j, j_perp, t2, lat_sc: Lat_sc1, lat_cc: Lat_cc):
    '''
    computes all necessary eigenvectors
    ------------
    ARGUMENTS:
    L: int: size of momentum grid
    j: float: spin coupling constant
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    -------------
    RETURNS:
    None
    '''
    v0 = True
    mom_ind = np.indices((L,L))
    mom = index2momentum(mom_ind, L)
    k1 = mom
    k2 = np.ones((2,L,L)) * np.pi - k1
    vs_sc_1 = []
    vs_sc_2 = []
    for x in range(L):
        for y in range(L):
            lat_sc.compute_H(k1[:,x,y], t=1., t2=t2, j=j)
            v = lat_sc.eigenvec(0)
            n0 = find_l0_state(lat_sc)
            v = v * np.exp(-1j * np.angle(v[n0]))
            vs_sc_1.append(v)

            lat_sc.compute_H(k2[:,x,y], t=1., t2=t2, j=j)
            v = lat_sc.eigenvec(0)
            n0 = find_l0_state(lat_sc)
            v = v * np.exp(-1j * np.angle(v[n0]))
            vs_sc_2.append(v)
    vs_sc_1 = np.array(vs_sc_1).reshape((L, L, lat_sc.basis.length))
    vs_sc_2 = np.array(vs_sc_2).reshape((L, L, lat_sc.basis.length))

    lat_sc.vs_sc_1 = vs_sc_1
    lat_sc.vs_sc_2 = vs_sc_2
    print('computed eigenvectors')
    
    lat_cc.compute_H(k1[:,0,0] + k2[:,0,0], 1., j, j_perp, p=-1)
    v_cc = lat_cc.eigenvec(0, v0=v0)
    v_cc = v_cc * np.exp(-1j * np.angle(v_cc[0]))
    lat_cc.v_cc = v_cc

def compute_eigenenergies_all_momenta(L, j, j_perp, t2, lat_sc: Lat_sc1, lat_cc: Lat_cc, p=-1):
    '''
    computes all necessary eigenenergies and save them
    ------------
    ARGUMENTS:
    L: int: size of momentum grid
    j: float: spin coupling constant
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    p: int -1 or 1: parity of quasiparticles i.ei +1 for bosons and -1 for fermions
    -------------
    RETURNS:
    None
    '''
    v0 = True
    mom_ind = np.indices((L,L))
    mom = index2momentum(mom_ind, L)
    k1 = mom
    es_sc = []
    es_cc = []
    for x in range(L):
        for y in range(L):
            lat_sc.compute_H(k1[:,x,y], 1., t2=t2, j=j)
            e = lat_sc.eigenval(0) + 2*j # + 2j only because of different convention for energy of l=0 state for sc vs. cc (sorry for this)
            es_sc.append(e)

            lat_cc.compute_H(k1[:,x,y], 1., j, j_perp, t2=t2, p=p)
            e = lat_cc.eigenval(0)
            es_cc.append(e)

    es_sc = np.array(es_sc).reshape((L, L))
    es_cc = np.array(es_cc).reshape((L, L))

    if p == -1:
        name = 'fer'
    elif p == 1:
        name = 'bos'
    else:
        raise ValueError('parity must be +1 or -1.')
    if j_perp == j:
        np.save(os.path.join(path, 'all_momenta', f'energies_sc_j{j:.2f}_t2{t2:.2f}_d{lat_sc.depth}.npy'), es_sc)
        np.save(os.path.join(path, 'all_momenta', f'energies_cc_j{j:.2f}_t2{t2:.2f}_d{lat_cc.depth}_p{name}.npy'), es_cc)
    else:
        np.save(os.path.join(path, 'all_momenta', f'energies_sc_j{j:.2f}_{j_perp:.2f}_t2{t2:.2f}_d{lat_sc.depth}.npy'), es_sc)
        np.save(os.path.join(path, 'all_momenta', f'energies_cc_j{j:.2f}_{j_perp:.2f}_t2{t2:.2f}_d{lat_cc.depth}_p{name}.npy'), es_cc)

def compute_eigenvectors_all_momenta(Lx, j, j_perp, t2, lat_sc1: Lat_sc1, lat_sc2: Lat_sc2, lat_cc: Lat_cc, Ly=0, p=-1):
    '''
    computes all necessary eigenvectors
    ------------
    ARGUMENTS:
    L: int: size of momentum grid
    j: float: spin coupling constant
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    p: int -1 or 1: parity of quasiparticles i.e. +1 for bosons and -1 for fermions
    -------------
    RETURNS:
    None
    '''
    if Ly == 0:
        Ly = Lx
    v0 = True
    mom_ind = np.indices((Lx, Ly))
    mom = index2momentum(mom_ind, Lx, Ly)
    k1 = mom
    print(mom.shape)
    vs_sc = []
    vs_cc = []
    for x in range(Lx):
        for y in range(Ly):
            lat_sc1.compute_H(k1[:,x,y], 1., t2=t2, j=j)
            if len((lat_sc1.H.toarray())[0]) > 500:
                v = lat_sc.eigenvec(0)
            else:
                _,v = lat_sc1.eigensys(0)
            n0 = find_l0_state(lat_sc1)
            v = v * np.exp(-1j * np.angle(v[n0])) # fix phase (set phase of l=0 state to zero)
            vs_sc.append(v)

            lat_sc2.compute_H(k1[:,x,y], 1., t2=t2, j=j)
            if len((lat_sc2.H.toarray())[0]) > 500:
                v = lat_sc.eigenvec(0)
            else:
                _,v = lat_sc2.eigensys(0)
            n0 = find_l0_state(lat_sc2)
            v = v * np.exp(-1j * np.angle(v[n0])) # fix phase (set phase of l=0 state to zero)
            vs_sc.append(v)

            lat_cc.compute_H(k1[:,x,y], 1., j, j_perp, p=p)
            if len((lat_cc.H.toarray())[0]) > 500:
                v_cc = lat_cc.eigenvec(0)
            else:
                _,v_cc = lat_cc.eigensys(0)
            v_cc = v_cc * np.exp(-1j * np.angle(v_cc[0])) # fix phase (set phase of l=0 state to zero)
            vs_cc.append(v_cc)

    vs_sc1 = np.array(vs_sc1).reshape((Lx, Ly, lat_sc1.basis.length))
    vs_sc2 = np.array(vs_sc2).reshape((Lx, Ly, lat_sc2.basis.length))
    vs_cc = np.array(vs_cc).reshape((Lx, Ly, len(lat_cc.representatives)))

    lat_sc1.vs = vs_sc1
    lat_sc2.vs = vs_sc2
    lat_cc.vs = vs_cc
    print('computed eigenvectors') 

# def compute_eigensys_all_momenta_exc(L, j, t2, n_sc, n_cc, lat_sc1: Lat_sc1, lat_cc: Lat_cc, p=-1):
    '''
    computes all necessary eigenvectors
    ------------
    ARGUMENTS:
    L: int: size of momentum grid
    j: float: spin coupling constant
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    p: int -1 or 1: parity of quasiparticles i.e +1 for bosons and -1 for fermions
    -------------
    RETURNS:
    None
    '''
    v0 = True
    mom_ind = np.indices((L,L))
    mom = index2momentum(mom_ind, L)
    k1 = mom
    vs_sc = []
    Es_sc = []
    vs_cc = []
    Es_cc = []
    Zs_cc = []
    for x in range(L):
        for y in range(L):
            lat_sc.compute_H(k1[:,x,y], 1., t2=t2, j=j)
            Es, vs = lat_sc.eigensys(n_sc-1, full=True)
            n0 = find_l0_state(lat_sc)
            vs = vs * np.exp(-1j * np.angle(vs[n0, 0]))
            vs_sc.append(vs)
            Es_sc.append(Es)

            lat_cc.compute_H(k1[:,x,y], 1., j, j, p=p)
            Es, vs = lat_cc.eigensys(n_cc-1, full=True)
            vs = vs * np.exp(-1j * np.angle(vs[0, 0]))

            # compute spectral weights
            Z_temp = []
            for m4 in range(4):
                v_rot = lat_cc.rot_trial_state(m4, k1[:,x,y], p)
                Z_temp.append(np.abs(np.einsum('i,in->n', np.conj(v_rot), vs)))
            Zs_cc.append(Z_temp)
            vs_cc.append(vs)
            Es_cc.append(Es)

    vs_sc = np.array(vs_sc).reshape((L, L, lat_sc.basis.length, n_sc))
    vs_cc = np.array(vs_cc).reshape((L, L, len(lat_cc.representatives), n_cc))
    Es_sc = np.array(Es_sc).reshape((L, L, n_sc))
    Es_cc = np.array(Es_cc).reshape((L, L, n_cc))
    Zs_cc = np.array(Zs_cc).reshape((L, L, 4, n_cc))

    lat_sc.vs = vs_sc
    lat_cc.vs = vs_cc
    lat_sc.Es = Es_sc
    lat_cc.Es = Es_cc
    lat_cc.Zs = Zs_cc
    print('computed eigenvectors and values')

def add_holes(n1, n2, jx, jy, lat_sc1: Lat_sc1, lat_sc2: Lat_sc2, lat_cc: Lat_cc):
    '''
    Takes 2 string states for the magnetic polaron (sc) with distance hole_pos_2 and create dictionary state in the format of Lat_cc
    ------------
    ARGUMENTS:
    n1 (int): index of string state 1 in sc truncated basis
    n2 (int): index of string state 2 in sc truncated basis
    hole_pos_2 (array of shape (2,)): gives position of the additional hole we poke into the system in the LLP frame of the first hole
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    -------------
    RETURNS:
    state: dictionary with lat, hole_pos, seq: state of the 2 sc's in the format of Lat_cc
    '''
    lat1 = lat_sc1.bin_basis[n1]['lat'].copy()
    subl1 = lat_sc1.find_sublattice(lat_sc1.bin_basis[n1])
    latsc1 = (lat1-subl1)%3 
    latsc1[lat_sc1.depth+1,lat_sc1.depth+1] = 3 #to include hole in sitelist
    sitelist1 = np.array(np.argwhere(latsc1).tolist())   

    lat2 = lat_sc2.bin_basis[n2]['lat'].copy()
    subl2 = lat_sc2.find_sublattice(lat_sc2.bin_basis[n2])
    latsc2 = (lat2-subl2)%3 
    latsc2[lat_sc2.depth+1,lat_sc2.depth+1] = 3 #to include hole in sitelist
    sitelist2 = np.array(np.argwhere(latsc2).tolist())

    offset = np.array([jx, jy])    
    x = lat_sc1.find_hole_sublattice(lat_sc1.bin_basis[n1]['seq'])
    y = lat_sc2.find_hole_sublattice(lat_sc2.bin_basis[n2]['seq'])
    skip_config = False  
    if np.amax(np.abs(sitelist2+offset-lat_sc1.depth-1)) > lat_cc.depth: #skips all sc states that are too long
        skip_config = True     
    if np.amax(np.abs(sitelist1-lat_sc1.depth-1)) > lat_cc.depth: #skips all sc states that are too long
        skip_config = True     
    if np.any([np.any(np.all(sitelist1 == row, axis=1)) for row in (sitelist2 + offset)]):   #skips all strings that would overlap, since the resulting state is not represented correctly at the moment
        skip_config = True    
                    
    if not skip_config:
        # if len(lat_sc.bin_basis[n1]['seq']) == 1:
        #     print(f'state of length=1 fulfilled conditions, n1={n1}')
        # if len(lat_sc.bin_basis[n1]['seq']) == 1 or len(lat_sc.bin_basis2[n2]['seq']) == 1:
        #     print(f'state of length=1 fulfilled conditions, n1={n1}, n2={n2}')
        #     print(f'seq of n1: {lat_sc.bin_basis[n1]['seq']}, seq of n2: {lat_sc.bin_basis2[n2]['seq']}')
        #     print()
        lat0 = copy.deepcopy(lat_cc.Neel_state[x])
        seq = []
        # print(f'sitelist1:{sitelist1}')
        for i in sitelist1:
            lat0[tuple(np.array(i)-lat_sc1.depth+lat_cc.depth)] = lat1[tuple(i)]
        # print(lat0)
        # print((lat0-lat_cc.Neel_state[x])%3)
        for i in sitelist2:
            lat0[tuple(np.array(i)+np.array(offset)-lat_sc1.depth+lat_cc.depth)] = lat2[tuple(i)]
        lat0[tuple(np.array(offset)+lat_cc.depth+1)] = 0     #mark hole site as 0
            
        # print(lat0)
        # print((lat0-lat_cc.Neel_state[x])%3)
            
        hole_pos = [np.ones((2,), dtype=int) * (lat_cc.depth + 1), np.ones((2,), dtype=int) * (lat_cc.depth + 1) + offset]
        for xx in hole_pos:
            lat0[tuple(xx)] = 0
        state = {'lat': lat0, 'hole_pos': hole_pos, 'seq': seq}
        # print(state)
        # if len(lat_sc.bin_basis[n1]['seq']) == 1:
        #     # print(f'state of length=1 fulfilled conditions, n1={n1}')
        #     # ###### plot states for debugging
        #     transformed_basis = transform_lattice_j_perp(lat_cc)
        #     found, m_t = transformed_basis.search(lat_cc.state_2_list_entry(state))
        #     # if 1==1:
        #     # if found:
        #     # print(f'state of length=1 fulfilled conditions, n1={n1}')
        #     lat_cc1 = (lat0-lat_cc.Neel_state[x])%3
        #     lat_cc1[tuple(hole_pos[0])] = 3
        #     lat_cc1[tuple(hole_pos[1])] = 4
        #     lat_sc1 = latsc1.copy()
        #     lat_sc1[lat_sc1.depth+1,lat_sc1.depth+1] = 3
        #     lat_sc2 = latsc2.copy()
        #     lat_sc1[lat_sc1.depth+1,lat_sc1.depth+1] = 3

        #     lats = []
        #     lats.append(lat_sc1)
        #     lats.append(lat_sc2)
        #     lats.append(lat_cc1)

        #     print(f'latsc1:{lat_sc1}')
        #     print(f'latsc2:{lat_sc2}')
        #     print(f'latcc:{lat_cc1}')
        #     print(f'offset: {offset}')
        #     print()

            # import matplotlib.pyplot as plt
            # from matplotlib.colors import ListedColormap
            # colors = ['red', 'blue', 'green', 'black', 'white']
            # custom_cmap = ListedColormap(colors)
            # fig, axs = plt.subplots(1,3, figsize=(8,3))
            # for i in range(3):
            #     # seq = lat_sc.bin_basis[indices[i]]['seq']
            #     # seq = tuple(move.tolist() for move in seq)
            #     axs[i].imshow(lats[i], origin='lower',cmap=custom_cmap)
            # plt.title(f'n1: {n1}, n2: {n2}, offset: {offset}, m_t: {m_t}')
            

        # ###### plot states for debugging
        # found,_ = lat_cc.basis.search(lat_cc.state_2_list_entry(state))
        # if not found:
        #     transformed_basis = transform_lattice_j_perp(lat_cc)
        #     found, m_t = transformed_basis.search(lat_cc.state_2_list_entry(state))
        #     # if 1==1:
        #     if found:
        #         lat_cc1 = lat0-lat_cc.Neel_state[x]
        #         lat_cc1[tuple(hole_pos[0])] = 3
        #         lat_cc1[tuple(hole_pos[1])] = 4
        #         lat_sc1 = latsc1.copy()
        #         lat_sc1[lat_sc.depth+1,lat_sc.depth+1] = 3
        #         lat_sc2 = latsc2.copy()
        #         lat_sc1[lat_sc.depth+1,lat_sc.depth+1] = 3

        #         lats = []
        #         lats.append(lat_sc1)
        #         lats.append(lat_sc2)
        #         lats.append(lat_cc1)

        #         # print(f'latsc1:{lat_sc1}')
        #         # print(f'latsc2:{lat_sc2}')
        #         # print(f'latcc:{lat_cc1}')

        #         import matplotlib.pyplot as plt
        #         from matplotlib.colors import ListedColormap
        #         colors = ['red', 'blue', 'green', 'black', 'white']
        #         custom_cmap = ListedColormap(colors)
        #         fig, axs = plt.subplots(1,3, figsize=(8,3))
        #         for i in range(3):
        #             # seq = lat_sc.bin_basis[indices[i]]['seq']
        #             # seq = tuple(move.tolist() for move in seq)
        #             axs[i].imshow(lats[i], origin='lower',cmap=custom_cmap)
        #         plt.title(f'n1: {n1}, n2: {n2}, offset: {offset}, m_t: {m_t}')
    else:
        state = -1 #not a dictionary, will get skipped              
    return state

def crop_lattice(lat, depth_sc, depth_cc, hole_pos_2):
    '''
    Takes a lattice of spin configurations from two sc's, crops it and gives out a state dictionary
    ------------
    ARGUMENTS:
    lat: array of shape(N, N) of bool: spin configuration of (sc)^2
    depth_sc (int): l_max for lat_sc
    depth_cc (int): l_max for lat_cc
    hole_pos_w: arrow of shape(2,) of int: posotion of second hole
    -------------
    RETURNS:
    state: dict: state as in Lat_cc
    '''
    seq = []
    j_max = np.amax(np.abs(hole_pos_2))
    dx = depth_sc + j_max - depth_cc
    if dx > 0:
        count = np.count_nonzero(lat[0:dx,:]) + np.count_nonzero(lat[-dx:,:]) + np.count_nonzero(lat[:,0:dx]) + np.count_nonzero(lat[:,-dx:])
        if count > 0:
            return -1
        else:
            lat = lat[dx:-dx, dx:-dx]
    elif dx < 0:
        lat = np.pad(lat, (-dx, -dx), 'constant', constant_values=False)
    hole_pos = [np.ones((2,), dtype=int) * (depth_cc + 1), np.ones((2,), dtype=int) * (depth_cc + 1) + hole_pos_2]
    for x in hole_pos:
        lat[tuple(x)] = False
    state = {'lat': lat, 'hole_pos': hole_pos, 'seq': seq}
    return state

def triangular_Neel_size_L(L,sublattice):
        if L%2 != 1:
            raise Exception(f'L = {L} not possible, Neel state lattice must have odd number of sites') 
        size = int((L-3)/2)
        # print(f'size={size}')
        triangular_lattice = np.zeros((L, L), dtype=int)
        for i in range(L):
            for j in range(L):
                if (i+j+sublattice+size+1) % 3 == 0: # +self.depth+1 so center site is same for all lattice sizes
                    triangular_lattice[i,j] = 0
                elif (i+j+sublattice+size+1) % 3 == 1:
                    triangular_lattice[i,j] = 1
                else:
                    triangular_lattice[i,j] = 2
        triangular_lattice[size+1,size+1] = 0
        #print(triangular_lattice)
        return triangular_lattice

def add_holes_j_perp(n1, n2, jx, jy, lat_sc1: Lat_sc1, lat_sc2: Lat_sc2, lat_cc: Lat_cc):
    '''
    Takes 2 string states for the magnetic polaron (sc) with distance hole_pos_2, apply H_J_perp^(-) (i.e. only flipping frustrated (wrt Neel) spins)
    and create dictionary state in the format of Lat_cc
    ------------
    ARGUMENTS:
    n1 (int): index of string state 1 in sc truncated basis
    n2 (int): index of string state 2 in sc truncated basis
    hole_pos_2 (array of shape (2,)): gives position of the additional hole we poke into the system in the LLP frame of the first hole
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    -------------
    RETURNS:
    ms: list of int: indices of cc states 
    '''
    lat1 = lat_sc1.bin_basis[n1]['lat'].copy()
    subl1 = lat_sc1.find_sublattice(lat_sc1.bin_basis[n1])
    latsc1 = (lat1-subl1)%3 
    latsc1[lat_sc1.depth+1,lat_sc1.depth+1] = 3 #to include hole in sitelist
    sitelist1 = np.array(np.argwhere(latsc1).tolist())   

    # print('seq_hole_1',seq_hole_1) #fix for case seq1=[]
    # print()
    lat2 = lat_sc2.bin_basis[n2]['lat'].copy()
    subl2 = lat_sc2.find_sublattice(lat_sc2.bin_basis[n2])
    latsc2 = (lat2-subl2)%3 
    latsc2[lat_sc2.depth+1,lat_sc2.depth+1] = 3 #to include hole in sitelist
    # print()
    # print(f'n1:{n1},n2:{n2}')
    # print('seq2:',lat_sc.bin_basis2[n2]['seq'])
    # print('latsc2:',latsc2)
    sitelist2 = np.array(np.argwhere(latsc2).tolist())
    # print(f'sitelist2:{sitelist2}')
    offset = np.array([jx, jy])    
    x = lat_sc1.find_hole_sublattice(lat_sc1.bin_basis[n1]['seq'])
    # print(f'x={x}')
    y = lat_sc2.find_hole_sublattice(lat_sc2.bin_basis[n2]['seq'])

    skip_config = False
    if np.amax(np.abs(sitelist2+offset-lat_sc1.depth-1)) > lat_cc.depth + 2: #skips all sc2 states that are longer even after flipping 2 spins
        skip_config = True    
    if np.amax(np.abs(sitelist1-lat_sc1.depth-1)) > lat_cc.depth + 2: #skips all sc1 states that are too long
        skip_config = True  
    if np.any([np.any(np.all(sitelist1 == row, axis=1)) for row in (sitelist2 + offset)]):   
        skip_config = True    #skips all strings that would overlap, since the resulting state is not represented correctly at the moment

    ms = []          
    if not skip_config:
        sublattice = triangular_Neel_size_L(lat_cc.L_size + 4,x)
        # print(f'subl:{sublattice}')
        lat0 = sublattice.copy()
        seq = []
        for i in sitelist1:
            lat0[tuple(np.array(i)-lat_sc1.depth+lat_cc.depth+2)] = lat1[tuple(i)]
            # print('sitelist1:')
            # print(f'site:{tuple(np.array(i)-lat_sc1.depth+lat_cc.depth+2)}, lat[site]:{lat1[tuple(i)]}')
        lat0[lat_cc.depth+3, lat_cc.depth+3] = 0     #mark hole site as 0
        # print('lat0_pre=',lat0)
        for i in sitelist2:
            lat0[tuple(np.array(i)+np.array(offset)-lat_sc1.depth+lat_cc.depth+2)] = lat2[tuple(i)]
            # print('sitelist2:')
            # print(f'site:{np.array(i)+np.array(offset)-lat_sc1.depth+lat_cc.depth+2}, lat[site]:{lat2[tuple(i)]}')
        lat0[tuple(np.array(offset)+lat_cc.depth+3)] = 0     #mark hole site as 0
        # print('lat: ',lat0)
        latcc = (lat0-sublattice)%3
        hole_pos = [np.ones((2,), dtype=int) * (lat_cc.depth + 3), offset + lat_cc.depth + 3]
        # print(f'hole_pos:{hole_pos}')
        for xx in hole_pos:
            lat0[tuple(xx)] = 0      #set holes to zero
            latcc[tuple(xx)] = 0 
        # print(f'offset:{offset}')
        # print(f'latsc1:{latsc1}')
        # print(f'latsc2:{latsc2}')  
        # print(f'lat1:{lat1}')
        # print(f'lat2:{lat2}')  
        # print('lat: ',lat0) 
        # print('latcc: ',latcc)
        siteslist = np.argwhere(latcc).tolist()
        
        hole_pos[0] += -2
        hole_pos[1] += -2
        # print(f'cc_L_size:{lat_cc.L_size}')
        # print(f'offset:{offset}, sc_state1: {n1}, sc_state2: {n2}')
        # print(f'lat_sc1:{latsc1}')
        # print(f'lat_sc2:{latsc2}')
        # print(f'lat_cc:{latcc}')
        # print(f'lat0:{lat0-np.pad(lat_cc.Neel_state[x], (2,2), 'constant', constant_values=False)}')
        # print('sitelist:',sitelist)
        # print('hole_pos:',hole_pos)
        # print()

        # print(f'sitelist1:{sitelist1},sitelist2:{sitelist2 + offset}')
        # print('sitelist',sitelist)
        # print('siteslist',siteslist)
        # print()
        #for site in siteslist: 
        for site in siteslist: 
            # print(site)
            nx = [[1,0],[0,1],[1,1]]  
            for nn in nx:
                # print('siteslist: ',siteslist)
                # print('test site: ',site[0]+nn[0],site[1]+nn[1])
                # print(f'site:{site}, nn:{nn}, sc state: {n2} ')
                # print()
                if [site[0]+nn[0],site[1]+nn[1]] in siteslist:
                    # print(f'found sites: {site[0]+nn[0],site[1]+nn[1]}')
                    lat1 = lat0.copy()
                    lat1[site[0],site[1]],lat1[site[0]+nn[0],site[1]+nn[1]]=lat1[site[0]+nn[0],site[1]+nn[1]],lat1[site[0],site[1]] #exchange both sites
                    #check if any spins are flipped outside L_size 
                    # print(f'lat_cc after:{lat1-np.pad(lat_cc.Neel_state[x], (2,2), 'constant', constant_values=False)}')
                    lat2 = lat1.copy()
                    lat2 = (lat2-sublattice)%3
                    lat2[2:-2, 2:-2] = 0 
                    if not np.all(lat2 == 0):
                        continue
                    lat1 = lat1[2:-2, 2:-2]  #crop lattice back to cc size
                    # print(f'lat_cc after after :{lat1-lat_cc.Neel_state[x]}')
                    # print(f'hole_pos after after:{hole_pos}')
                    # print()
                    state1 = {'lat': lat1, 'seq':seq, 'hole_pos': hole_pos}
                    # print(state1)
                    list_entry = lat_cc.state_2_list_entry(state1)
                    found, m = lat_cc.basis.search(list_entry)
                    if found:
                        ms.append(m)
                        # print(f'n1: {n1}, n2: {n2}, offset: {offset}, m: {m}')
                        # print()
                        # print(lat_cc.bin_basis[m]['lat'])
                        # print(f'found cc state connecting flipped sc states')
                        # print(f'ms={m}, hole_pos:{hole_pos}, sc_state1:{n1}, sc_state2:{n2}, site:{site},nn:{nn}')
                        # print(f'lat_sc1:{latsc1}')
                        # print(f'lat_sc2:{latsc2}')
                        # latcc = (lat_cc.bin_basis[m]['lat']-lat_cc.Neel_state[x])%3
                        # print(f'lat_cc:{latcc}')

                        #plot states for debugging
                        # latcc1 = (lat_cc.bin_basis[m]['lat']-lat_cc.Neel_state[x])%3
                        # latcc1[tuple(hole_pos[0])] = 3
                        # latcc1[tuple(hole_pos[1])] = 4
                        # latsc1_ = latsc1.copy()
                        # latsc1_[lat_sc1.depth+1,lat_sc1.depth+1] = 4
                        # latsc2_ = latsc2.copy()
                        # latsc1_[lat_sc1.depth+1,lat_sc1.depth+1] = 4

                        # lats = []
                        # lats.append(latsc1_)
                        # lats.append(latsc2_)
                        # lats.append(latcc1)
                        # # print(f'latsc1:{latsc1}')
                        # # print(f'latsc2:{latsc2}')
                        # # print(f'latcc:{latcc1}')

                        # indices = []
                        # indices.append(n1)
                        # indices.append(n2)
                        # indices.append(m)
                        # import matplotlib.pyplot as plt
                        # from matplotlib.colors import ListedColormap
                        # colors = ['red', 'blue', 'green', 'black', 'white']
                        # custom_cmap = ListedColormap(colors)
                        # fig, axs = plt.subplots(1,3, figsize=(8,3))
                        # for i in range(3):
                        #     # seq = lat_sc.bin_basis[indices[i]]['seq']
                        #     # seq = tuple(move.tolist() for move in seq)
                        #     axs[i].imshow(lats[i], origin='lower',cmap=custom_cmap)
                        # plt.title(f'n1: {n1}, n2: {n2}, offset: {offset}, m: {m}')
                        
                        
        #print('ms=',ms)
    return ms

def transform_lattice_j_perp(lat_cc: Lat_cc):
    '''
    Builds a basis with all possible states of the form H_J_perp|n_cc>
    ------------
    ARGUMENTS:
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    -------------
    RETURNS:
    transformed_basis: sorted_list: sorted_list of all transformed basis states
                        entries are lists with [:L_size] = lat
                        [L_size, L_size+2] = hole_pos
                        [L_size+2, :] originial indices, i.e. index in lat_cc to know from which string state the n-th transformed state was obtained
    '''
    transformed_basis = sorted_list(length_arr = lat_cc.L_size + 2)
    N = 0
    for i in range(lat_cc.basis.length):
        state = lat_cc.bin_basis[i]
        lat = state['lat']
        hole_pos = state['hole_pos']
        seq = state['seq']
        if len(seq) < 2:
            continue
        sublattice = lat_cc.find_sublattice(state)
        lat0 = (lat-sublattice)%3
        lat0[tuple(hole_pos[0])] = 0
        lat0[tuple(hole_pos[1])] = 0
        # print('state: ',state)
        # print('lat0: ',lat0)

        siteslist = (np.argwhere(lat0)).tolist()
        for site in siteslist: 
            nx = [[1,0],[0,1],[1,1]]  
            for nn in nx:
                # print('siteslist: ',siteslist)
                # print('test site: ',site[0]+nn[0],site[1]+nn[1])
                if [site[0]+nn[0],site[1]+nn[1]] in siteslist:
                    lat1 = lat.copy()
                    lat1[site[0],site[1]],lat1[site[0]+nn[0],site[1]+nn[1]]=lat1[site[0]+nn[0],site[1]+nn[1]],lat1[site[0],site[1]] #exchange both sites
                    state1 = {'lat': lat1, 'seq':seq, 'hole_pos': hole_pos}
                    list_entry = lat_cc.state_2_list_entry(state1)
                    # print('test2')
                    # print('list_entry: ',list_entry)
                    new = transformed_basis.add(list_entry)
                    # if not new:
                    #     print(f'redundent cc state in trafo_basis, seq:{len(seq)}')
                    _, n = transformed_basis.search(list_entry)
                    transformed_basis.list[n].append(i)
                    N += 1

                    # #### plot states for debugging
                    # lat_pre = lat0.copy()
                    # lat_pre[tuple(hole_pos[0])] = 3
                    # lat_pre[tuple(hole_pos[1])] = 4
                    # lat_after = (lat1-sublattice)%3
                    # lat_after[tuple(hole_pos[0])] = 3
                    # lat_after[tuple(hole_pos[1])] = 4
                    # lats = []
                    # lats.append(lat_pre)
                    # lats.append(lat_after)
                    # import matplotlib.pyplot as plt
                    # from matplotlib.colors import ListedColormap
                    # colors = ['red', 'blue', 'green', 'black', 'white']
                    # custom_cmap = ListedColormap(colors)
                    # fig, axs = plt.subplots(1,2, figsize=(12,3))
                    # for j in range(2):
                    #     axs[j].imshow(lats[j], origin='lower',cmap=custom_cmap)
                    #     plt.title(f'n:{i}, seq:{seq}')
    print(f'elements in transformed basis N={N}')
    return transformed_basis

def overlap(k1, k2, j, t2, lat_sc: Lat_sc, lat_cc: Lat_cc):
    '''
    Computes the absolute value of the overlap |M(k, p)| of the cc wavefunction at momentum k1+k2 and the (sc)^2 wave-functions with momenta k1 and k2.
    ------------
    ARGUMENTS:
    k1: array of shape (2,): momentum of sc 1
    k2: array of shape (2,): momentum of sc 2
    j: float: spin coupling constant
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    -------------
    RETURNS:
    |M|: (float): absolute value of the overlap
    '''
    v0 = True
    lat_sc.compute_H(k1, t=1., j=j, t2=t2)
    v_sc_1 = lat_sc.eigenvec(0)
    n0 = find_l0_state(lat_sc)
    v_sc_1 = v_sc_1 * np.exp(-1j * np.angle(v_sc_1[n0]))
    lat_sc.compute_H(k2, t=1., j=j, t2=t2)
    v_sc_2 = lat_sc.eigenvec(0)
    n0 = find_l0_state(lat_sc)
    v_sc_2 = v_sc_2 * np.exp(-1j * np.angle(v_sc_2[n0]))
    lat_cc.compute_H(k1 + k2, 1., j, j, p=-1)
    v_cc = lat_cc.eigenvec(0, v0=v0)
    v_cc = v_cc * np.exp(-1j * np.angle(v_cc[0]))

    # #  test where difference between different runs comes from
    # print('v_sc_1')
    # print(np.abs(v_sc_1[:5]))
    # print('v_sc_2')
    # print(np.abs(v_sc_2[:5]))
    # print('v_cc')
    # print(np.abs(v_cc[:5]))

    # assert ishermitian(lat_sc.H.toarray())
    # assert ishermitian(lat_cc.H.toarray())

    # print('Hamiltonian sc')
    # print((lat_sc.H.data[:5]))
    # # end test

    l_max = lat_cc.depth

    M = 0
    for n1 in lat_sc.indices:
        alpha_sc1 = v_sc_1[n1]
        for n2 in lat_sc.indices:
            alpha_sc2 = v_sc_2[n2]
            # determine area in which the second hole can be placed to form a cc pair
            temp = np.argwhere(lat_sc.bin_basis[n1][0]) - lat_sc.depth - 1
            temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
            lower_bound_1 = np.amin(temp, axis=0)
            upper_bound_1 = np.amax(temp, axis=0)

            temp = np.argwhere(lat_sc.bin_basis[n2][0]) - lat_sc.depth - 1
            temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
            lower_bound_2 = np.amin(temp, axis=0)
            upper_bound_2 = np.amax(temp, axis=0)

            j_min = np.maximum(lower_bound_1 - upper_bound_2 - 1, lower_bound_2 - l_max)
            j_max = np.minimum(upper_bound_1 - lower_bound_2 + 1, l_max - upper_bound_2)

            for jx in range(j_min[0], j_max[0] + 1):
                for jy in range(j_min[1], j_max[1] + 1):
                    if jx == 0 and jy == 0:
                        continue
                    state = add_holes(n1, n2, np.array([jx, jy]), lat_sc, lat_cc)
                    if type(state)== dict:
                        found, m = lat_cc.basis.search(lat_cc.state_2_list_entry(state))
                        if found:
                            repr, _, m = lat_cc.is_representative[m]
                            dM = 1/np.sqrt(2) * np.conj(v_cc[m]) * alpha_sc1 * alpha_sc2 * np.exp(1j * np.dot(k2, np.array([jx, jy])))
                            # factor 1/sqrt(2) comes from projection onto ferminoic states
                            if not repr:
                                dM *= -1 * np.exp(-1j * np.dot((k1 + k2), np.array([jx, jy])))
                            M += dM
    # return np.abs(M)
    return M

def overlap_t(k1, k2, j, t2, lat_sc: Lat_sc, lat_cc: Lat_cc):
    '''
    Computes the absolute value of the overlap |M_t'(k, p)| = <psi_cc(k1+k2)|H_t'|psi_sc(k1), psi_sc(k2)>
    of the cc wavefunction at momentum k1+k2 and H_t' applied to (sc)^2 wavefunction with momenta k1 and k2
    ------------
    ARGUMENTS:
    k1: array of shape (2,): momentum of sc 1
    k2: array of shape (2,): momentum of sc 2
    j: float: spin coupling constant
    t2: float: NNN hopping
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    -------------
    RETURNS:
    |M|: (float): absolute value of the overlap
    '''
    v0 = True
    lat_sc.compute_H(k1, t=1., j=j, t2=t2)
    v_sc_1 = lat_sc.eigenvec(0)
    n0 = find_l0_state(lat_sc)
    v_sc_1 = v_sc_1 * np.exp(-1j * np.angle(v_sc_1[n0]))
    lat_sc.compute_H(k2, t=1., j=j, t2=t2)
    v_sc_2 = lat_sc.eigenvec(0)
    n0 = find_l0_state(lat_sc)
    v_sc_2 = v_sc_2 * np.exp(-1j * np.angle(v_sc_2[n0]))
    lat_cc.compute_H(k1 + k2, 1., j, j, p=-1)
    v_cc = lat_cc.eigenvec(0, v0=v0)
    v_cc = v_cc * np.exp(-1j * np.angle(v_cc[0]))

    l_max = lat_cc.depth

    M = 0
    for n1 in lat_sc.indices:
        alpha_sc1 = v_sc_1[n1]
        for n2 in lat_sc.indices:
            alpha_sc2 = v_sc_2[n2]
            # determine area in which the second hole can be placed to form a cc pair
            temp = np.argwhere(lat_sc.bin_basis[n1][0]) - lat_sc.depth - 1
            temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
            lower_bound_1 = np.amin(temp, axis=0)
            upper_bound_1 = np.amax(temp, axis=0)

            temp = np.argwhere(lat_sc.bin_basis[n2][0]) - lat_sc.depth - 1
            temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
            lower_bound_2 = np.amin(temp, axis=0)
            upper_bound_2 = np.amax(temp, axis=0)

            j_min = np.maximum(lower_bound_1 - upper_bound_2 - 2, lower_bound_2 - l_max)
            j_max = np.minimum(upper_bound_1 - lower_bound_2 + 2, l_max - upper_bound_2)

            for jx in range(j_min[0], j_max[0] + 1):
                for jy in range(j_min[1], j_max[1] + 1):
                    if jx == 0 and jy == 0:
                        continue
                    state = add_holes(n1, n2, np.array([jx, jy]), lat_sc, lat_cc)
                    if type(state)==dict:
                        found, _ = lat_cc.basis.search(lat_cc.state_2_list_entry(state))
                    else:
                        found = False
                    if not found:
                        for delta in [np.array([1,1]), np.array([1,-1]), np.array([-1,-1]), np.array([-1,1])]:
                            ### Hopping of hole 1 (w.r.t. we chose the reference frame)
                            hole_pos = [state['hole_pos'][0] + delta, state['hole_pos'][1]]
                            test_lat = state['lat'].copy()
                            pos0 = tuple(np.array([1, 1]) * (lat_cc.depth+1))
                            pos1 = tuple(np.array([1, 1]) * (lat_cc.depth+1) + delta)
                            pos2 = tuple(np.array([1, 1]) * (lat_cc.depth+1) + np.array([jx, jy]))
                            test_lat[pos0] = test_lat[pos1]
                            test_lat[pos1] = False
                            test_lat[pos2] = False

                            test_lat, hole_pos = lat_cc.translation(test_lat, hole_pos, np.concatenate(([0], delta)))

                            test_state = {'lat': test_lat, 'hole_pos': hole_pos, 'seq': state['seq']}
                            found, m = lat_cc.basis.search(lat_cc.state_2_list_entry(test_state))
                            if found:
                                repr, _, m = lat_cc.is_representative[m]
                                dM = 1/np.sqrt(2) * t2 * np.conj(v_cc[m]) * alpha_sc1 * alpha_sc2 * np.exp(1j * np.dot(k2, np.array([jx, jy]) - delta)) * np.exp(-1j * np.dot(k1, delta))
                                # dM = 1/np.sqrt(2) * t2 * np.conj(v_cc[m]) * alpha_sc1 * alpha_sc2 * np.exp(1j * np.dot(k1, np.array([jx, jy])))
                                # factor 1/sqrt(2) comes from projection onto fermionic states
                                if not repr:
                                    dM *= -1 * np.exp(-1j * np.dot((k1 + k2), np.array([jx, jy]) - delta))
                                M += dM

                            ### Hopping of hole 2
                            hole_pos = [state['hole_pos'][0], state['hole_pos'][1] + delta]
                            test_lat = state['lat'].copy()
                            pos1 = tuple(np.array([1, 1]) * (lat_cc.depth+1) + np.array([jx, jy]))
                            pos2 = tuple(np.array([1, 1]) * (lat_cc.depth+1) + np.array([jx, jy]) + delta)
                            test_lat[pos1] = test_lat[pos2]
                            test_lat[pos2] = False
                            test_state = {'lat': test_lat, 'hole_pos': hole_pos, 'seq': state['seq']}
                            found, m = lat_cc.basis.search(lat_cc.state_2_list_entry(test_state))
                            if found:
                                repr, _, m = lat_cc.is_representative[m]
                                dM = 1/np.sqrt(2) * t2 * np.conj(v_cc[m]) * alpha_sc1 * alpha_sc2 * np.exp(1j * np.dot(k2, np.array([jx, jy])))
                                # factor 1/sqrt(2) comes from projection onto fermionic states
                                if not repr:
                                    dM *= -1 * np.exp(-1j * np.dot((k1 + k2), np.array([jx, jy]) + delta))
                                M += dM
    # return np.abs(M)
    return M

def overlap_j(k1, k2, j, t2, lat_sc: Lat_sc, lat_cc: Lat_cc):
    '''
    Computes the absolute value of the overlap |M_t'(k, p)| = <psi_cc(k1+k2)|H_J_perp|psi_sc(k1), psi_sc(k2)>
    of the cc wavefunction at momentum k1+k2 and H_J_perp applied to (sc)^2 wavefunction with momenta k1 and k2
    ------------
    ARGUMENTS:
    k1: array of shape (2,): momentum of sc 1
    k2: array of shape (2,): momentum of sc 2
    j: float: spin coupling constant
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    -------------
    RETURNS:
    |M|: (float): absolute value of the overlap
    '''
    v0 = True
    lat_sc.compute_H(k1, t=1., j=j, t2=t2)
    v_sc_1 = lat_sc.eigenvec(0)
    n0 = find_l0_state(lat_sc)
    v_sc_1 = v_sc_1 * np.exp(-1j * np.angle(v_sc_1[n0]))
    lat_sc.compute_H(k2, t=1., j=j, t2=t2)
    v_sc_2 = lat_sc.eigenvec(0)
    n0 = find_l0_state(lat_sc)
    v_sc_2 = v_sc_2 * np.exp(-1j * np.angle(v_sc_2[n0]))
    lat_cc.compute_H(k1 + k2, 1., j, j, p=-1)
    v_cc = lat_cc.eigenvec(0, v0=v0)
    v_cc = v_cc * np.exp(-1j * np.angle(v_cc[0]))

    l_max = lat_cc.depth

    transformed_basis = transform_lattice_j_perp(lat_cc)

    M = 0
    for n1 in lat_sc.indices:
        alpha_sc1 = v_sc_1[n1]
        for n2 in lat_sc.indices:
            alpha_sc2 = v_sc_2[n2]
            # determine area in which the second hole can be placed to form a cc pair
            temp = np.argwhere(lat_sc.bin_basis[n1][0]) - lat_sc.depth - 1
            temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
            lower_bound_1 = np.amin(temp, axis=0)
            upper_bound_1 = np.amax(temp, axis=0)

            temp = np.argwhere(lat_sc.bin_basis[n2][0]) - lat_sc.depth - 1
            temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
            lower_bound_2 = np.amin(temp, axis=0)
            upper_bound_2 = np.amax(temp, axis=0)

            j_min = np.maximum(lower_bound_1 - upper_bound_2 - 4, lower_bound_2 - l_max)
            j_max = np.minimum(upper_bound_1 - lower_bound_2 + 4, l_max - upper_bound_2)

            for jx in range(j_min[0], j_max[0] + 1):
                for jy in range(j_min[1], j_max[1] + 1):
                    if jx == 0 and jy == 0:
                        continue
                    # apply S ^(+)_i S^(+)_j on l.h.s. of expectation value, i.e.
                    # <cc|SS
                    state = add_holes(n1, n2, np.array([jx, jy]), lat_sc, lat_cc)
                    if type(state)== dict:
                        found, m_t = transformed_basis.search(lat_cc.state_2_list_entry(state))
                        if found:
                            ms = transformed_basis.list[m_t][lat_cc.L_size + 2:]
                            for m in ms:
                                repr, _, m = lat_cc.is_representative[m]
                                dM = 0.5 * 1/np.sqrt(2) * j * np.conj(v_cc[m]) * alpha_sc1 * alpha_sc2 * np.exp(1j * np.dot(k2, np.array([jx, jy])))
                                # factor 1/sqrt(2) comes from projection onto ferminoic states
                                # factor 1/2 comes from H_J_perp
                                if not repr:
                                    dM *= -1 * np.exp(-1j * np.dot((k1 + k2), np.array([jx, jy])))
                                M += dM
                    # apply S ^(-)_i S^(-)_j on r.h.s. of expectation value, i.e.
                    # SS|(sc)^2>
                    ms = add_holes_j_perp(n1, n2, np.array([jx, jy]), lat_sc, lat_cc)
                    for m in ms:
                        repr, _, m = lat_cc.is_representative[m]
                        dM = 0.5 * 1/np.sqrt(2) * j * np.conj(v_cc[m]) * alpha_sc1 * alpha_sc2 * np.exp(1j * np.dot(k2, np.array([jx, jy])))
                        # factor 1/sqrt(2) comes from projection onto ferminoic states
                        # factor 1/2 comes from H_J_perp
                        if not repr:
                            dM *= -1 * np.exp(-1j * np.dot((k1 + k2), np.array([jx, jy])))
                        M += dM
    # return np.abs(M)
    return M

def overlap_grid(j,j_perp, L, t2, lat_sc: Lat_sc, lat_cc: Lat_cc):
    '''
    Computes the absolute value of the overlap |M(k, p)| of the cc wavefunction at momentum k1+k2 and the (sc)^2 wave-functions with momenta k1 and k2.
    ------------
    ARGUMENTS:
    j: float: spin coupling constant
    L: int: size of momentum grid
    t2: float: next-neartest neighbor hopping constant t'
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    -------------
    RETURNS:
    |M|: (float): absolute value of the overlap
    '''
    mom_ind = np.indices((L,L))
    mom = index2momentum(mom_ind, L)
    k1 = mom
    k2 = np.ones((2,L,L)) * np.pi - k1

    # vs_sc_1, vs_sc_2, v_cc = compute_eigenvectors(L, j, j_perp, t2, lat_sc, lat_cc)
    vs_sc_1 = lat_sc.vs_sc_1
    vs_sc_2 = lat_sc.vs_sc_2
    v_cc= lat_cc.v_cc

    l_max = lat_cc.depth

    Ms = np.zeros((L, L))
    for n1 in lat_sc.indices:
        for n2 in lat_sc.indices:
            # determine area in which the second hole can be placed to form a cc pair
            temp = np.argwhere(lat_sc.bin_basis[n1][0]) - lat_sc.depth - 1
            temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
            lower_bound_1 = np.amin(temp, axis=0)
            upper_bound_1 = np.amax(temp, axis=0)

            temp = np.argwhere(lat_sc.bin_basis[n2][0]) - lat_sc.depth - 1
            temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
            lower_bound_2 = np.amin(temp, axis=0)
            upper_bound_2 = np.amax(temp, axis=0)

            j_min = np.maximum(lower_bound_1 - upper_bound_2 - 1, lower_bound_2 - l_max)
            j_max = np.minimum(upper_bound_1 - lower_bound_2 + 1, l_max - upper_bound_2)

            for jx in range(j_min[0], j_max[0] + 1):
                for jy in range(j_min[1], j_max[1] + 1):
                    if jx == 0 and jy == 0:
                        continue
                    state = add_holes(n1, n2, np.array([jx, jy]), lat_sc, lat_cc)
                    if type(state)== dict:
                        found, m = lat_cc.basis.search(lat_cc.state_2_list_entry(state))
                        if found:
                            repr, _, m = lat_cc.is_representative[m]
                            dM = 1/np.sqrt(2) * np.conj(v_cc[m]) * vs_sc_1[:,:,n1] * vs_sc_2[:,:,n2] * np.exp(1j * np.einsum('axy,a->xy', k2, np.array([jx, jy])))
                            # factor 1/sqrt(2) comes from projection onto ferminoic states
                            if not repr:
                                dM *= -1 * np.exp(-1j * np.einsum('axy,a->xy', k1 + k2, np.array([jx, jy])))
                            Ms = Ms + dM
    Ms = np.pad(Ms, (0,1), mode='wrap')
    # return np.abs(Ms)
    return np.real_if_close(Ms, tol=1e-4)

def overlap_grid_j(j, j_perp, L, t2, lat_sc: Lat_sc, lat_cc: Lat_cc):
    '''
    Computes the absolute value of the overlap |M_t'(k, p)| = <psi_cc(k1+k2)|H_j|psi_sc(k1), psi_sc(k2)>
    of the cc wavefunction at momentum 0 and H applied to (sc)^2 wavefunction with momenta k and -k
    for k in a L x L grid 
    ------------
    ARGUMENTS:
    j: float: spin coupling constant
    L : int: linear size of momentum grid
    t2: float: next-neartest neighbor hopping constant t'
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    -------------
    RETURNS:
    |Ms|: array ph shape (L, L) of (float): absolute value of the overlaps
    '''
    mom_ind = np.indices((L,L))
    mom = index2momentum(mom_ind, L)
    k1 = mom
    k2 = np.ones((2,L,L)) * np.pi - k1

    # vs_sc_1, vs_sc_2, v_cc = compute_eigenvectors(L, j, j_perp, t2, lat_sc, lat_cc)
    vs_sc_1 = lat_sc.vs_sc_1
    vs_sc_2 = lat_sc.vs_sc_2
    v_cc= lat_cc.v_cc

    l_max = lat_cc.depth

    transformed_basis = transform_lattice_j_perp(lat_cc)

    Ms = np.zeros((L, L))
    for n1 in lat_sc.indices:
        for n2 in lat_sc.indices:
            # determine area in which the second hole can be placed to form a cc pair
            temp = np.argwhere(lat_sc.bin_basis[n1][0]) - lat_sc.depth - 1
            temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
            lower_bound_1 = np.amin(temp, axis=0)
            upper_bound_1 = np.amax(temp, axis=0)

            temp = np.argwhere(lat_sc.bin_basis[n2][0]) - lat_sc.depth - 1
            temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
            lower_bound_2 = np.amin(temp, axis=0)
            upper_bound_2 = np.amax(temp, axis=0)

            j_min = np.maximum(lower_bound_1 - upper_bound_2 - 4, lower_bound_2 - l_max)
            j_max = np.minimum(upper_bound_1 - lower_bound_2 + 4, l_max - upper_bound_2)

            for jx in range(j_min[0], j_max[0] + 1):
                for jy in range(j_min[1], j_max[1] + 1):
                    if jx == 0 and jy == 0:
                        continue
                    # apply S ^(+)_i S^(+)_j on l.h.s. of expectation value, i.e.
                    # <cc|SS
                    state = add_holes(n1, n2, np.array([jx, jy]), lat_sc, lat_cc)
                    if type(state)==dict:
                        found, _ = lat_cc.basis.search(lat_cc.state_2_list_entry(state))
                    else:
                        found = False
                    if not found:
                        if type(state)== dict:
                            found, m_t = transformed_basis.search(lat_cc.state_2_list_entry(state))
                            if found:
                                ms = transformed_basis.list[m_t][lat_cc.L_size + 2:]
                                for m in ms:
                                    repr, _, m = lat_cc.is_representative[m]
                                    dM = 0.5 * 1/np.sqrt(2) * np.conj(v_cc[m]) * vs_sc_1[:,:,n1] * vs_sc_2[:,:,n2] * np.exp(1j * np.einsum('axy,a->xy', k2, np.array([jx, jy])))
                                    # factor 1/sqrt(2) comes from projection onto ferminoic states
                                    # factor 1/2 comes from H_J_perp
                                    if not repr:
                                        dM *= -1 * np.exp(-1j * np.einsum('axy,a->xy', k1 + k2, np.array([jx, jy])))
                                    Ms = Ms + dM
                        # apply S ^(-)_i S^(-)_j on r.h.s. of expectation value, i.e.
                        # SS|(sc)^2>
                        ms = add_holes_j_perp(n1, n2, np.array([jx, jy]), lat_sc, lat_cc)
                        for m in ms:
                            repr, _, m = lat_cc.is_representative[m]
                            dM = 0.5 * 1/np.sqrt(2) * np.conj(v_cc[m]) * vs_sc_1[:,:,n1] * vs_sc_2[:,:,n2] * np.exp(1j * np.einsum('axy,a->xy', k2, np.array([jx, jy])))
                            # factor 1/sqrt(2) comes from projection onto ferminoic states
                            # factor 1/2 comes from H_J_perp
                            if not repr:
                                dM *= -1 * np.exp(-1j * np.einsum('axy,a->xy', k1 + k2, np.array([jx, jy])))
                            Ms = Ms + dM
    Ms = np.pad(Ms, (0,1), mode='wrap')
    # return np.abs(Ms)
    return np.real_if_close(Ms, tol=1e-4)

def overlap_grid_t(j, j_perp, L, t2, lat_sc: Lat_sc, lat_cc: Lat_cc):
    '''
    Computes the absolute value of the overlap |M_t'(k, p)| = <psi_cc(k1+k2)|H_j|psi_sc(k1), psi_sc(k2)>
    of the cc wavefunction at momentum 0 and H applied to (sc)^2 wavefunction with momenta k and -k
    for k in a L x L grid 
    ------------
    ARGUMENTS:
    j: float: spin coupling constant
    L : int: linear size of momentum grid
    t2: float: next-neartest neighbor hopping constant t'
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    -------------
    RETURNS:
    |Ms|: array ph shape (L, L) of (float): absolute value of the overlaps
    '''
    mom_ind = np.indices((L,L))
    mom = index2momentum(mom_ind, L)
    k1 = mom
    k2 = np.ones((2,L,L)) * np.pi - k1

    # vs_sc_1, vs_sc_2, v_cc = compute_eigenvectors(L, j, j_perp, t2, lat_sc, lat_cc)
    vs_sc_1 = lat_sc.vs_sc_1
    vs_sc_2 = lat_sc.vs_sc_2
    v_cc= lat_cc.v_cc

    l_max = lat_cc.depth

    Ms = np.zeros((L, L))
    for n1 in lat_sc.indices:
        for n2 in lat_sc.indices:
            # determine area in which the second hole can be placed to form a cc pair
            temp = np.argwhere(lat_sc.bin_basis[n1][0]) - lat_sc.depth - 1
            temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
            lower_bound_1 = np.amin(temp, axis=0)
            upper_bound_1 = np.amax(temp, axis=0)

            temp = np.argwhere(lat_sc.bin_basis[n2][0]) - lat_sc.depth - 1
            temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
            lower_bound_2 = np.amin(temp, axis=0)
            upper_bound_2 = np.amax(temp, axis=0)

            j_min = np.maximum(lower_bound_1 - upper_bound_2 - 2, lower_bound_2 - l_max)
            j_max = np.minimum(upper_bound_1 - lower_bound_2 + 2, l_max - upper_bound_2)

            for jx in range(j_min[0], j_max[0] + 1):
                for jy in range(j_min[1], j_max[1] + 1):
                    if jx == 0 and jy == 0:
                        continue
                    state = add_holes(n1, n2, np.array([jx, jy]), lat_sc, lat_cc)
                    if type(state)==dict:
                        found, _ = lat_cc.basis.search(lat_cc.state_2_list_entry(state))
                    else:
                        found = False
                    if not found:
                        if type(state) == dict:
                            for delta in [np.array([1,1]), np.array([1,-1]), np.array([-1,-1]), np.array([-1,1])]:
                                ### Hopping of hole 1 (w.r.t. we chose the reference frame)
                                hole_pos = [state['hole_pos'][0] + delta, state['hole_pos'][1]]
                                test_lat = state['lat'].copy()
                                pos0 = tuple(np.array([1, 1]) * (lat_cc.depth+1))
                                pos1 = tuple(np.array([1, 1]) * (lat_cc.depth+1) + delta)
                                pos2 = tuple(np.array([1, 1]) * (lat_cc.depth+1) + np.array([jx, jy]))
                                test_lat[pos0] = test_lat[pos1]
                                test_lat[pos1] = False
                                test_lat[pos2] = False

                                test_lat, hole_pos = lat_cc.translation(test_lat, hole_pos, np.concatenate(([0], delta)))

                                test_state = {'lat': test_lat, 'hole_pos': hole_pos, 'seq': state['seq']}
                                found, m = lat_cc.basis.search(lat_cc.state_2_list_entry(test_state))
                                if found:
                                    repr, _, m = lat_cc.is_representative[m]
                                    dM = 1/np.sqrt(2) * np.conj(v_cc[m]) * vs_sc_1[:,:,n1] * vs_sc_2[:,:,n2] * np.exp(1j * np.einsum('axy,a->xy', k2, np.array([jx, jy]) - delta) - 1j * np.einsum('axy,a->xy', k1, delta))
                                    # factor 1/sqrt(2) comes from projection onto fermionic states
                                    if not repr:
                                        dM *= -1 * np.exp(-1j * np.einsum('axy,a->xy', k1 + k2, np.array([jx, jy]) - delta))
                                    Ms = Ms + dM

                                #### Hole 2 hops
                                hole_pos = [state['hole_pos'][0], state['hole_pos'][1] + delta] 
                                test_lat = state['lat'].copy()
                                pos1 = tuple(np.array([1, 1]) * (lat_cc.depth+1) + np.array([jx, jy]))
                                pos2 = tuple(np.array([1, 1]) * (lat_cc.depth+1) + np.array([jx, jy]) + delta)
                                test_lat[pos1] = test_lat[pos2]
                                test_lat[pos2] = False
                                test_state = {'lat': test_lat, 'hole_pos': hole_pos, 'seq': state['seq']}
                                found, m = lat_cc.basis.search(lat_cc.state_2_list_entry(test_state))
                                if found:
                                    repr, _, m = lat_cc.is_representative[m]
                                    dM = 1/np.sqrt(2) * np.conj(v_cc[m]) * vs_sc_1[:,:,n1] * vs_sc_2[:,:,n2] * np.exp(1j * np.einsum('axy,a->xy', k2, np.array([jx, jy])))
                                    # factor 1/sqrt(2) comes from projection onto ferminoic states
                                    if not repr:
                                        dM *= -1 * np.exp(-1j * np.einsum('axy,a->xy', k1 + k2, np.array([jx, jy]) + delta))
                                    Ms = Ms + dM
    Ms = np.pad(Ms, (0,1), mode='wrap')
    # return np.abs(Ms)
    return np.real_if_close(Ms, tol=1e-4)

def overlap_all_momenta(j, L, t2, lat_sc:Lat_sc, lat_cc:Lat_cc, p=-1):
    '''
    Computes the overlap M_t'(k, p) = <psi_cc(k1+k2)|H_j|psi_sc(k1), psi_sc(k2)>
    of the cc wavefunction at momentum k+p and H applied to (sc)^2 wavefunction with momenta k and p
    in a L x L grid 
    ------------
    ARGUMENTS:
    j: float: spin coupling constant
    L : int: linear size of momentum grid
    t2: float: next-neartest neighbor hopping constant t'
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    -------------
    RETURNS:
    |Ms|: array ph shape (L, L) of (float): absolute value of the overlaps
    '''
    mom_ind = np.indices((L,L))
    mom = index2momentum(mom_ind, L)
    k1 = mom.reshape((2, L, L, 1, 1))
    k2 = mom.reshape((2, 1, 1, L, L))
    k_plus_p_ind = sum_ind(mom_ind.reshape((2,L,L,1,1)), mom_ind.reshape((2,1,1,L,L)), L)

    # vs_sc_1, vs_sc_2, v_cc = compute_eigenvectors(L, j, j_perp, t2, lat_sc, lat_cc)
    vs_sc_1 = lat_sc.vs.reshape((L, L, 1, 1, lat_sc.basis.length))
    vs_sc_2 = lat_sc.vs.reshape((1, 1, L, L, lat_sc.basis.length))
    v_cc = lat_cc.vs

    l_max = lat_cc.depth

    transformed_basis = transform_lattice_j_perp(lat_cc)
    c2 = 0
    c3 = 0
    c4 = 0
    c5 = 0
    c6 = 0
    Ms = np.zeros((L, L, L, L))
    for n1 in lat_sc.indices1:
        for n2 in lat_sc.indices2:
            
            j_min = (np.ones(2,)*(-lat_cc.depth)).astype(int) #improve these bounds in the future
            j_max = (np.ones(2,)*(lat_cc.depth)).astype(int)

            seq1 = lat_sc.bin_basis[n1]['seq']
            seq_hole_1 = []
            if len(seq1) != 0:
                for i, step in enumerate(seq1):
                    seq_hole_1.append([0,step[0],step[1]]) #bring seq_hole_1 in the form [[0,x,y],...] add to dist_2_phys_dist
    
            for jx in range(j_min[0], j_max[0] + 1):        
                for jy in range(j_min[1], j_max[1] + 1):
                    # apply S ^(+)_i S^(+)_j on l.h.s. of expectation value, i.e.
                    # <cc|SS
                    state = add_holes(n1, n2, jx, jy, lat_sc, lat_cc)

                    if type(state)==dict:
                        c2 += 1
                        found, _ = lat_cc.basis.search(lat_cc.state_2_list_entry(state))    #why look for this state in the lat_cc basis? even if state is in lat_cc basis, it can also be in transformed basis 
                    else:
                        found = False
                    if not found:
                        if type(state)== dict:
                            found, m_t = transformed_basis.search(lat_cc.state_2_list_entry(state))
                            if found:
                                c3 += 1
                                ms = transformed_basis.list[m_t][lat_cc.L_size + 2:]
                                for m in ms:
                                    repr, _, m = lat_cc.is_representative[m]
                                    dM = 0.5 * 1/np.sqrt(2) * np.conj(v_cc[k_plus_p_ind[0,:,:,:,:], k_plus_p_ind[1,:,:,:,:], m]) * vs_sc_1[:,:,:,:,n1] * vs_sc_2[:,:,:,:,n2] * np.exp(1j * np.einsum('nabcd,n->abcd', k2, np.array([jx, jy])))
                                    # factor 1/sqrt(2) comes from projection onto fermionic states
                                    # factor 1/2 comes from H_J_perp
                                    if not repr:
                                        c4 += 1
                                        phys_dist = np.array(lat_cc.dist_2_phys_dist([jx, jy],seq_hole_1))
                                        dM *= -1 * np.exp(-1j * np.einsum('nabcd,n->abcd', k1 + k2, phys_dist))
                                    Ms = Ms + dM
                        # apply S ^(-)_i S^(-)_j on r.h.s. of expectation value, i.e.
                        # SS|(sc)^2>
                        ms = add_holes_j_perp(n1, n2, jx, jy, lat_sc, lat_cc)
                        
                        for m in ms:
                            # print(f'm={m}, check')
                            repr, _, m = lat_cc.is_representative[m]
                            dM = 0.5 * 1/np.sqrt(2) * np.conj(v_cc[k_plus_p_ind[0,:,:,:,:], k_plus_p_ind[1,:,:,:,:], m]) * vs_sc_1[:,:,:,:,n1] * vs_sc_2[:,:,:,:,n2] * np.exp(1j * np.einsum('nabcd,n->abcd', k2, np.array([jx, jy])))
                            # factor 1/sqrt(2) comes from projection onto fermionic states
                            # factor 1/2 comes from H_J_perp
                            
                            c5 += 1

                            if not repr:
                                c6+=1
                                phys_dist = np.array(lat_sc.dist_2_phys_dist([jx, jy],seq_hole_1))
                                dM *= p * np.exp(-1j * np.einsum('nabcd,n->abcd', k1 + k2, phys_dist))
                            Ms = Ms + dM
    Ms_j = np.pad(Ms, (0,1), mode='wrap')
    # print(f'c1={c1}')
    print(f'c2={c2}')
    print(f'c3={c3}')
    print(f'c4={c4}')
    print(f'c5={c5}')
    print(f'c6={c6}')
    
    # Ms = np.zeros((L, L, L, L))
    # for n1 in lat_sc.indices:
    #     for n2 in lat_sc.indices:
    #         # determine area in which the second hole can be placed to form a cc pair
    #         temp = np.argwhere(lat_sc.bin_basis[n1][0]) - lat_sc.depth - 1
    #         temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
    #         lower_bound_1 = np.amin(temp, axis=0)
    #         upper_bound_1 = np.amax(temp, axis=0)

    #         temp = np.argwhere(lat_sc.bin_basis[n2][0]) - lat_sc.depth - 1
    #         temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
    #         lower_bound_2 = np.amin(temp, axis=0)
    #         upper_bound_2 = np.amax(temp, axis=0)

    #         j_min = np.maximum(lower_bound_1 - upper_bound_2 - 2, lower_bound_2 - l_max)
    #         j_max = np.minimum(upper_bound_1 - lower_bound_2 + 2, l_max - upper_bound_2)

    #         for jx in range(j_min[0], j_max[0] + 1):
    #             for jy in range(j_min[1], j_max[1] + 1):
    #                 if jx == 0 and jy == 0:
    #                     continue
    #                 state = add_holes(n1, n2, np.array([jx, jy]), lat_sc, lat_cc)
    #                 if type(state)==dict:
    #                     found, _ = lat_cc.basis.search(lat_cc.state_2_list_entry(state))
    #                 else:
    #                     found = False
    #                 if not found:
    #                     if type(state) == dict:
    #                         for delta in [np.array([1,1]), np.array([1,-1]), np.array([-1,-1]), np.array([-1,1])]: #what about these deltas
    #                             ### Hopping of hole 1 (w.r.t. we chose the reference frame)
    #                             hole_pos = [state['hole_pos'][0] + delta, state['hole_pos'][1]]
    #                             test_lat = state['lat'].copy()
    #                             pos0 = tuple(np.array([1, 1]) * (lat_cc.depth+1))
    #                             pos1 = tuple(np.array([1, 1]) * (lat_cc.depth+1) + delta)
    #                             pos2 = tuple(np.array([1, 1]) * (lat_cc.depth+1) + np.array([jx, jy]))
    #                             test_lat[pos0] = test_lat[pos1]
    #                             test_lat[pos1] = False
    #                             test_lat[pos2] = False

    #                             test_lat, hole_pos = lat_cc.translation(test_lat, hole_pos, np.concatenate(([0], delta)))

    #                             test_state = {'lat': test_lat, 'hole_pos': hole_pos, 'seq': state['seq']}
    #                             found, m = lat_cc.basis.search(lat_cc.state_2_list_entry(test_state))
    #                             if found:
    #                                 repr, _, m = lat_cc.is_representative[m]
    #                                 dM = 1/np.sqrt(2) * np.conj(v_cc[k_plus_p_ind[0,:,:,:,:], k_plus_p_ind[1,:,:,:,:], m]) * vs_sc_1[:,:,:,:,n1] * vs_sc_2[:,:,:,:,n2] * np.exp(1j * np.einsum('nabcd,n->abcd', k2, np.array([jx, jy]) - delta) - 1j * np.einsum('nabcd,n->abcd', k1, delta))
    #                                 # factor 1/sqrt(2) comes from projection onto fermionic states
    #                                 if not repr:
    #                                     phys_dist = np.array(lat_sc.dist_2_phys_dist(np.array([jx, jy]) - delta))
    #                                     dM *= p * np.exp(-1j * np.einsum('nabcd,n->abcd', k1 + k2, phys_dist))
    #                                 Ms = Ms + dM

    #                             #### Hole 2 hops
    #                             hole_pos = [state['hole_pos'][0], state['hole_pos'][1] + delta] 
    #                             test_lat = state['lat'].copy()
    #                             pos1 = tuple(np.array([1, 1]) * (lat_cc.depth+1) + np.array([jx, jy]))
    #                             pos2 = tuple(np.array([1, 1]) * (lat_cc.depth+1) + np.array([jx, jy]) + delta)
    #                             test_lat[pos1] = test_lat[pos2]
    #                             test_lat[pos2] = False
    #                             test_state = {'lat': test_lat, 'hole_pos': hole_pos, 'seq': state['seq']}
    #                             found, m = lat_cc.basis.search(lat_cc.state_2_list_entry(test_state))
    #                             if found:
    #                                 repr, _, m = lat_cc.is_representative[m]
    #                                 dM = 1/np.sqrt(2) * np.conj(v_cc[k_plus_p_ind[0,:,:,:,:], k_plus_p_ind[1,:,:,:,:], m]) * vs_sc_1[:,:,:,:,n1] * vs_sc_2[:,:,:,:,n2] * np.exp(1j * np.einsum('nabcd,n->abcd', k2, np.array([jx, jy])))
    #                                 # factor 1/sqrt(2) comes from projection onto ferminoic states
    #                                 if not repr:
    #                                     phys_dist = np.array(lat_sc.dist_2_phys_dist(np.array([jx, jy]) + delta))
    #                                     dM *= p * np.exp(-1j * np.einsum('nabcd,n->abcd', k1 + k2, phys_dist))
    #                                 Ms = Ms + dM
    # Ms_t = np.pad(Ms, (0,1), mode='wrap')
    # return np.abs(Ms)
    return np.real_if_close(Ms_j, tol=1e-4) #, np.real_if_close(Ms_t, tol=1e-4)

def overlap_all_momenta_excited(j, L, t2, n_sc, n_cc, lat_sc:Lat_sc, lat_cc:Lat_cc, p=-1):
    '''
    Computes the overlap M_t'(k, p) = <psi_cc(k1+k2; n_cc)|H_j|psi_sc(k1; n_sc_1), psi_sc(k2; n_sc_2)>
    of the cc wavefunction at momentum k+p and H applied to (sc)^2 wavefunction with momenta k and p
    in a L x L grid 
    ------------
    ARGUMENTS:
    j: float: z componenet of the spin coupling constant
    j_perp: XY component of the spin coupling
    L : int: linear size of momentum grid
    t2: float: next-neartest neighbor hopping constant t'
    n_sc: number of energie levels for the sc
    n_cc: number of energuie levels for the cc
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    -------------
    RETURNS:
    |Ms|: array ph shape (L, L) of (float): absolute value of the overlaps
    '''
    mom_ind = np.indices((L,L))
    mom = index2momentum(mom_ind, L)
    k1 = mom.reshape((2, L, L, 1, 1))
    k2 = mom.reshape((2, 1, 1, L, L))
    k_plus_p_ind = sum_ind(mom_ind.reshape((2,L,L,1,1)), mom_ind.reshape((2,1,1,L,L)), L)

    # vs_sc_1, vs_sc_2, v_cc = compute_eigenvectors(L, j, j_perp, t2, lat_sc, lat_cc)
    vs_sc_1 = lat_sc.vs.reshape((L, L, 1, 1, lat_sc.basis.length, n_sc, 1, 1))
    vs_sc_2 = lat_sc.vs.reshape((1, 1, L, L, lat_sc.basis.length, 1, n_sc, 1))
    v_cc = lat_cc.vs.reshape((L, L, len(lat_cc.representatives), 1, 1, n_cc))

    l_max = lat_cc.depth

    transformed_basis = transform_lattice_j_perp(lat_cc)

    Ms = np.zeros((L, L, L, L, n_sc, n_sc, n_cc))
    for n1 in lat_sc.indices:
        for n2 in lat_sc.indices:
            # determine area in which the second hole can be placed to form a cc pair
            temp = np.argwhere(lat_sc.bin_basis[n1][0]) - lat_sc.depth - 1
            temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
            lower_bound_1 = np.amin(temp, axis=0)
            upper_bound_1 = np.amax(temp, axis=0)

            temp = np.argwhere(lat_sc.bin_basis[n2][0]) - lat_sc.depth - 1
            temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
            lower_bound_2 = np.amin(temp, axis=0)
            upper_bound_2 = np.amax(temp, axis=0)

            j_min = np.maximum(lower_bound_1 - upper_bound_2 - 4, lower_bound_2 - l_max)
            j_max = np.minimum(upper_bound_1 - lower_bound_2 + 4, l_max - upper_bound_2)

            for jx in range(j_min[0], j_max[0] + 1):
                for jy in range(j_min[1], j_max[1] + 1):
                    if jx == 0 and jy == 0:
                        continue
                    # apply S ^(+)_i S^(+)_j on l.h.s. of expectation value, i.e.
                    # <cc|SS
                    state = add_holes(n1, n2, np.array([jx, jy]), lat_sc, lat_cc)
                    if type(state)==dict:
                        found, _ = lat_cc.basis.search(lat_cc.state_2_list_entry(state))
                    else:
                        found = False
                    if not found:
                        if type(state)== dict:
                            found, m_t = transformed_basis.search(lat_cc.state_2_list_entry(state))
                            if found:
                                ms = transformed_basis.list[m_t][lat_cc.L_size + 2:]
                                for m in ms:
                                    repr, _, m = lat_cc.is_representative[m]
                                    dM = 0.5 * 1/np.sqrt(2) * np.conj(v_cc[k_plus_p_ind[0,:,:,:,:], k_plus_p_ind[1,:,:,:,:], m, :, :, :]) * vs_sc_1[:,:,:,:,n1,:,:,:] * vs_sc_2[:,:,:,:,n2,:,:,:] * np.exp(1j * np.einsum('nabcd,n->abcd', k2, np.array([jx, jy]))).reshape((1, 1, L, L, 1, 1, 1))
                                    # factor 1/sqrt(2) comes from projection onto fermionic states
                                    # factor 1/2 comes from H_J_perp
                                    if not repr:
                                        dM *= -1 * np.exp(-1j * np.einsum('nabcd,n->abcd', k1 + k2, np.array([jx, jy]))).reshape((L, L, L, L, 1, 1, 1))
                                    Ms = Ms + dM
                        # apply S ^(-)_i S^(-)_j on r.h.s. of expectation value, i.e.
                        # SS|(sc)^2>
                        ms = add_holes_j_perp(n1, n2, np.array([jx, jy]), lat_sc, lat_cc)
                        for m in ms:
                            repr, _, m = lat_cc.is_representative[m]
                            dM = 0.5 * 1/np.sqrt(2) * np.conj(v_cc[k_plus_p_ind[0,:,:,:,:], k_plus_p_ind[1,:,:,:,:], m]) * vs_sc_1[:,:,:,:,n1] * vs_sc_2[:,:,:,:,n2] * np.exp(1j * np.einsum('nabcd,n->abcd', k2, np.array([jx, jy]))).reshape((1, 1, L, L, 1, 1, 1))
                            # factor 1/sqrt(2) comes from projection onto fermionic states
                            # factor 1/2 comes from H_J_perp
                            if not repr:
                                dM *= p * np.exp(-1j * np.einsum('nabcd,n->abcd', k1 + k2, np.array([jx, jy]))).reshape((L, L, L, L, 1, 1, 1))
                            Ms = Ms + dM
    Ms_j = np.pad(Ms, (0,1), mode='wrap')
    return Ms_j

def load_plots(depth_sc, sign=''):
    '''
    Loads overlaps computed on cluster and plots them
    ------------
    ARGUMENTS:
    depth_sc: int: maximal string length of the sc in overlaps
    -------------
    RETURNS:
    None
    '''
    path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'sc-cc-overlaps')
    path_plot = os.path.join(os.path.dirname(os.getcwd()), 'plots', 'sc-cc-overlaps')

    # name = f'overlaps_{depth_sc}{sign}'
    # Ms = np.load(os.path.join(path, name + '.npy'))
    # if np.amax(np.abs(np.imag(Ms))) > 0:
    #     print('WARNING: overlap is complex')
    #     print(f'l_max = {depth_sc}, {sign}')
    #     print(f'max imaginary part = {np.amax(np.abs(np.imag(Ms))):.4f}')
    #     Ms = np.real(Ms)
    # fig, ax = plt.subplots(1, 1)
    # ax.set_xlabel('$k_x/\pi$')
    # ax.set_ylabel('$k_y/\pi$')
    # ax.set_title('overlaps, $l_{max}=$'+f'{depth_sc}')
    # im = ax.imshow(Ms.T, origin='lower', extent=[-1,1,-1,1], cmap='RdBu')
    # fig.colorbar(im)
    # plt.savefig(os.path.join(path_plot, name + '.pdf'))

    name = f'overlaps_j_{depth_sc}{sign}'
    Ms = np.load(os.path.join(path, name + '.npy'))
    if np.amax(np.abs(np.imag(Ms))) > 0:
        print('WARNING: overlap J is complex')
        print(f'l_max = {depth_sc}, {sign}')
        print(f'max imaginary part = {np.amax(np.abs(np.imag(Ms))):.4f}')
        Ms = np.real(Ms)
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('$k_x/\\pi$')
    ax.set_ylabel('$k_y/\\pi$')
    ax.set_title('$M_{J}(k), \\; l_{max}=$'+f'{depth_sc}')
    im = ax.imshow(Ms.T, origin='lower', extent=[-1,1,-1,1], cmap='RdBu')
    fig.colorbar(im)
    # plt.savefig(os.path.join(path_plot, name + '.pdf'))

    name = f'overlaps_t_{depth_sc}{sign}'
    Ms = np.load(os.path.join(path, name + '.npy'))
    if np.amax(np.abs(np.imag(Ms))) > 0:
        print('WARNING: overlap t2 is complex')
        print(f'l_max = {depth_sc}, {sign}')
        print(f'max imaginary part = {np.amax(np.abs(np.imag(Ms))):.4f}')
        Ms = np.real(Ms)
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('$k_x/\\pi$')
    ax.set_ylabel('$k_y/\\pi$')
    ax.set_title('$M_{t\'}(k),\\; l_{max}=$'+f'{depth_sc}')
    im = ax.imshow(Ms.T, origin='lower', extent=[-1,1,-1,1], cmap='RdBu')
    fig.colorbar(im)
    # plt.savefig(os.path.join(path_plot, name + '.pdf'))

    # plt.show()

def load_total(depth_sc, sign, j, t2=0.2):

    path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'sc-cc-overlaps')
    path_plot = os.path.join(os.path.dirname(os.getcwd()), 'plots', 'sc-cc-overlaps')

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('$k_x/\\pi$')
    ax.set_ylabel('$k_y/\\pi$')
    ax.set_title('$M(k), \\; l_{max}=$'+f'{depth_sc}')

    name = f'overlaps_j_{depth_sc}_{sign}'
    Ms_j = np.load(os.path.join(path, name + '.npy'))
    Ms_j = np.real(Ms_j)

    name = f'overlaps_t_{depth_sc}_{sign}'
    Ms_t = np.load(os.path.join(path, name + '.npy'))
    Ms_t = np.real(Ms_t)

    if sign == 'pos':
        Ms = Ms_j + t2/j * Ms_t
    elif sign == 'neg':
        Ms = Ms_j - t2/j * Ms_t
    else:
        raise ValueError('sign has to have the value "pos" or "neg"')
    im = ax.imshow(Ms.T, origin='lower', extent=[-1,1,-1,1], cmap='RdBu')
    fig.colorbar(im)

    # plt.show()

def load_plots_all_momenta(k, j, t2, depth_overlap, p=-1, path_overlaps=None):
    '''Load the overlaps V and energies of (sc) and (cc) as computed in the truncated basis'''
    L_data = 16
    if p == 1:
        parity = 'bos'
    elif p == -1:
        parity = 'fer'
    else:
        raise ValueError('Parity has to be +1 or -1.')
    if not path_overlaps:
        path_overlaps = '/Users/pit.bermes/Documents/t-J model/data/sc-cc-overlaps/all_momenta'
    V = j * np.load(os.path.join(path_overlaps, f'overlaps_{parity}_j_{depth_overlap}.npy')) + t2 * np.load(os.path.join(path_overlaps, f'overlaps_{parity}_t_{depth_overlap}.npy'))
    assert V.shape == (L_data+1, L_data+1, L_data+1, L_data+1)
    V = V[:-1, :-1, :-1, :-1]

    index = np.mod((np.round((k/np.pi + 1)*L_data/2)).astype(int), L_data)
    x = index[0]
    y = index[1]
    print('k_index =', index)
    interaction = np.roll((np.abs(V)**2)[::-1, ::-1, :, :], (x-L_data//2 + 1, y - L_data//2 + 1), axis=(0,1))
    interaction = np.einsum('abab->ab', interaction).reshape((L_data, L_data))

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('$p_x/\\pi$')
    ax.set_ylabel('$p_y/\\pi$')
    ax.set_title('$M(k-p, p), \\; l_{max}=$'+f'{depth_sc}, $k=({k[0]:.2f},{k[1]:.2f})$')
    im = ax.imshow(np.pad(interaction.T, (0,1), mode='wrap'), origin='lower', extent=[-1,1,-1,1], cmap='Oranges')
    fig.colorbar(im)

    # # ------------ start test for k=0 -------------
    # V = j * np.load(os.path.join(path_overlaps, f'overlaps_{parity}_j_{depth_overlap}.npy')) + t2 * np.load(os.path.join(path_overlaps, f'overlaps_{parity}_t_{depth_overlap}.npy'))
    # assert V.shape == (L_data+1, L_data+1, L_data+1, L_data+1)
    # V = V[:-1, :-1, :-1, :-1]

    # index = np.mod((np.round(k/np.pi + 1)*L_data/2).astype(int), L_data)
    # x = index[0]
    # y = index[1]
    # for x in range(L_data):
    #     for y in range(L_data):
    #         interaction[x,y] = np.abs(V[(L_data - x)%L_data, (L_data - y)%L_data, x, y])**2
    # fig, ax = plt.subplots(1, 1)
    # ax.set_xlabel('$p_x/\pi$')
    # ax.set_ylabel('$p_y/\pi$')
    # ax.set_title('$M(k-p, p), \; l_{max}=$'+f'{depth_sc}')
    # im = ax.imshow(np.pad(interaction.T, (0,1), mode='wrap'), origin='lower', extent=[-1,1,-1,1], cmap='Oranges')
    # fig.colorbar(im)
    # # ----------- end test ------------------
    # plt.show()

def convergence_plots(depth_sc_max, depth_sc_min=2, sign=''):
    '''
    Loads overlaps computed on cluster and plot difference for different l_max(sc)
    ------------
    ARGUMENTS:
    depth_sc_max: int: maximal string length of the sc
    depth_sc_max: int: minimal truncation of string length of the sc
    -------------
    RETURNS:
    None
    '''
    path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'sc-cc-overlaps')
    path_plot = os.path.join(os.path.dirname(os.getcwd()), 'plots', 'sc-cc-overlaps')

    name = f'overlaps_t_{depth_sc_max}' + sign
    Ms_max = np.load(os.path.join(path, name + '.npy'))
    Ms_max = np.real(Ms_max)
    fig, ax = plt.subplots(depth_sc_max - depth_sc_min, 1, sharex='all', sharey='all')
    for n, depth in enumerate(range(depth_sc_min, depth_sc_max)):
        name = f'overlaps_t_{depth}_pos'
        Ms = np.load(os.path.join(path, name + '.npy'))
        Ms = np.real(Ms)
        
        ax[n].set_xlabel('$k_x/\\pi$')
        ax[n].set_ylabel('$k_y/\\pi$')
        # ax[n].set_title('overlaps, $l_{max}=$'+f'{depth_sc}')
        im = ax[n].imshow((Ms_max - Ms).T, origin='lower', extent=[-1,1,-1,1], cmap='RdBu')
    fig.colorbar(im, ax=ax.ravel().tolist())
    # plt.savefig(os.path.join(path_plot, 'convegence_j' + '.pdf'))

def test_symmetries(V=False, parity=-1):
    # # 1) test symmetries under exchange of both momenta for cc ground state (pi, pi)
    # if type(V) == bool:
    #     V = np.load(os.path.join(path, 'overlaps_j_3.npy'))[:-1, :-1]
    # else:
    #     V = V[:-1, :-1]
    # print(f'shape of overlaps = {V.shape}')
    # L = V.shape[0]
    # V2 = np.roll(V[::-1, ::-1], (L//2 + 1, L//2 + 1), axis=(0,1))

    # fig, ax = plt.subplots(3, 2)
    # ax[0, 0].imshow(np.abs(V))
    # ax[0, 1].imshow(np.abs(V2))
    # ax[1, 0].imshow(np.real(V))
    # ax[1, 1].imshow(np.real(V2))
    # ax[2, 0].imshow(np.imag(V))
    # ax[2, 1].imshow(np.imag(V2))

    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(np.abs(V + V2))

    # print('not antisymmetric by', np.amax(np.abs(V + V2)))

    # plt.show()

    # 2) test symmetries under exchange of both momenta for any momenta
    if parity == 1:
        parity_name = 'bos'
    elif parity == -1:
        parity_name = 'fer'
    else:
        raise ValueError('parity eigenvalue has to be +1 or -1.')
    V = np.load(os.path.join(path, 'all_momenta', f'overlaps_{parity_name}_t_3.npy'))[:-1, :-1, :-1, :-1]
    import time
    print(f'''file last modified at {time.ctime(os.path.getmtime(os.path.join(path, 'all_momenta', f'overlaps_{parity_name}_t_3.npy')))}''')
    print(f'shape of overlaps = {V.shape}')
    L = V.shape[0]
    V2 = V.transpose(2,3,0,1)

    V = V.reshape((L**2, L**2))
    V2 = V2.reshape((L**2, L**2))

    fig, ax = plt.subplots(3, 2)
    ax[0, 0].imshow(np.abs(V))
    ax[0, 1].imshow(np.abs(V2))
    ax[1, 0].imshow(np.real(V))
    ax[1, 1].imshow(np.real(V2))
    ax[2, 0].imshow(np.imag(V))
    ax[2, 1].imshow(np.imag(V2))

    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.abs(V + V2))

    print(f'max of |V|: {np.amax(np.abs(V))}')
    print(np.amax(np.abs(V - parity * V2)))
    print(np.amax(np.real(V - parity * V2)))
    print(np.amax(np.imag(V - parity * V2)))
    plt.show()

if __name__ == '__main__':
    print('starting __main__')
    j = 1/3
    j_perp = j
    t2 = 0
    depth_sc = 8
    l_max_sc_overlaps = 2
    depth_cc = 10
    p = -1
    L = 16
    Lx = L
    Ly = 4
    # k1 = np.array([1, 1]) * np.pi / 2 * 1
    # k1 = np.random.random((2,))
    # k2 = 0*np.array([np.pi, np.pi]) - k1

    path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'sc-cc-overlaps')

    # t0 = perf_counter()
    lats = init_lattices_cyl(depth_sc, depth_cc, l_max_sc_overlaps=l_max_sc_overlaps, Ly=Ly, j_perp_div_j=j_perp/j, connected=True)
    # compute_eigenvectors_all_momenta(L, j, j_perp, t2, *lats, Ly=Ly)
    # Ms_j, Ms_t = overlap_all_momenta_cylinder(j, L, Ly, t2, *lats)
    # print(f'Lattices initialized in {perf_counter() - t0:.2f} seconds')

    # x = 0
    # y = 0
    # Ms_j = Ms_j[:-1,:-1,:-1,:-1]
    # Ms_t = Ms_t[:-1,:-1,:-1,:-1]
    # interaction = np.roll(np.real(Ms_j)[::-1, ::-1, :, :], (x-Lx//2 + 1, y - Ly//2 + 1), axis=(0,1))
    # interaction = np.einsum('abab->ab', interaction).reshape((Lx, Ly))

    # plt.imshow(interaction.T)
    # plt.figure()
    # interaction = np.roll(np.real(Ms_t)[::-1, ::-1, :, :], (x-Lx//2 + 1, y - Ly//2 + 1), axis=(0,1))
    # interaction = np.einsum('abab->ab', interaction).reshape((Lx, Ly))
    # plt.imshow(interaction.T)
    # plt.show()


    # t0 = perf_counter()
    # lats = init_lattices(depth_sc, depth_cc, l_max_sc_overlaps=l_max_sc_overlaps, j_perp_div_j=j_perp/j, connected=True)
    # print(f'Lattices initialized in {perf_counter() - t0:.2f} seconds')

    # Ms_j, Ms_t = overlap_all_momenta_cylinder(j, )

    # t0 = perf_counter()
    # compute_eigensys_all_momenta_exc(L, j, j_perp, t2, 2, 3, *lats, p)
    # Ms_j, Ms_t = overlap_all_momenta_excited(j, L, t2, 2, 3, *lats, p)

    # t0 = perf_counter()
    # compute_eigenvectors(L, j, j_perp, t2, *lats)
    # print(f'coputed eigenvectors in {perf_counter() - t0:.2f} seconds')

    # k1 = np.array([1.0, 0.7])
    # k2 = np.array([0.05, 0.01])

    # x = overlap_t(k1, k2, j, t2, *lats)
    # print('-------------------------------------------------')
    # y = overlap_t(k2, k1, j, t2, *lats)
    # print(np.abs(x + y))
    # print(x, y)
    # print(np.abs(x), np.abs(y), np.angle(x/y))

    # t0 = perf_counter()
    # Ms = overlap_grid(j, L, t2, *lats)
    # print(f'Computed overlap on grid in {perf_counter() - t0:.2f} seconds')
    # # np.save(os.path.join(path, f'overlaps_{depth_sc}.npy'), Ms)
    # if np.amax(np.abs(np.imag(Ms))) > 0:
    #     print('WARNING: Overlaps are not real')
    #     Ms = np.real(Ms)
    # fig, ax = plt.subplots(1, 1)
    # ax.set_xlabel('$k_x/\pi$')
    # ax.set_ylabel('$k_y/\pi$')
    # ax.set_title('overlaps, $l_{max}=$'+f'{depth_sc}')
    # im = ax.imshow(Ms.T, origin='lower', extent=[-1,1,-1,1], cmap='RdBu')
    # fig.colorbar(im)

    # t0 = perf_counter()
    # Ms = overlap_grid_j(j, L, t2, *lats)
    # print(f'Computed overlap J on grid in {perf_counter() - t0:.2f} seconds')
    # # np.save(os.path.join(path, f'overlaps_j_{depth_sc}.npy'), Ms)
    # if np.amax(np.abs(np.imag(Ms))) > 0:
    #     print('WARNING: Overlaps J are not real')
    #     Ms = np.real(Ms)
    # fig, ax = plt.subplots(1, 1)
    # ax.set_xlabel('$k_x/\pi$')
    # ax.set_ylabel('$k_y/\pi$')
    # ax.set_title('$M_{J}(k), \; l_{max}=$'+f'{depth_sc}')
    # im = ax.imshow(Ms.T, origin='lower', extent=[-1,1,-1,1], cmap='RdBu')
    # fig.colorbar(im)
    # # test_symmetries(Ms)

    # t0 = perf_counter()
    # Ms = overlap_grid_t(j, L, t2, *lats)
    # print(f'Computed overlap t\' on grid in {perf_counter() - t0:.2f} seconds')
    # # np.save(os.path.join(path, f'overlaps_t_{depth_sc}.npy'), Ms)
    # if np.amax(np.abs(np.imag(Ms))) > 0:
    #     print('WARNING: Overlaps t2 are not real')
    #     Ms = np.real(Ms)
    # fig, ax = plt.subplots(1, 1)
    # ax.set_xlabel('$k_x/\pi$')
    # ax.set_ylabel('$k_y/\pi$')
    # ax.set_title('$M_{t\'}(k),\; l_{max}=$'+f'{depth_sc}')
    # im = ax.imshow(Ms.T, origin='lower', extent=[-1,1,-1,1], cmap='RdBu')
    # fig.colorbar(im)
    # # test_symmetries(Ms)

    # print(Ms[L//2, -1])

    # plt.show()

    # ----------------------------------------------------------------------------

    # for l in range(5, 6):
    #     load_plots(l, '_neg')
    # # plt.show()

    # # j = 1/3
    # # t2 = 0.2
    # # load_total(5, 'pos', j, t2)
    # # load_total(5, 'neg', j, t2)
    # # plt.show()

    # # convergence_plots(4, 1)
    # plt.show()

    # ---------------- begin test ----------------
    # test_symmetries(parity=+1)
    # ---------------- end test ------------------


    ### compute eigenenergies
    jperps = [0.1*j, 0.5*j, j]
    for j_perp in jperps:
        compute_eigenenergies_all_momenta(L, j, j_perp, t2, *lats, p)
        # compute_eigenenergies_all_momenta(L, j, j_perp, t2, *lats, -p)

    # # --------------------------------------
    # k = np.array([-3/8, 0])*np.pi
    # load_plots_all_momenta(k, j=1/3, t2=0., depth_overlap=3, p=1)
    # load_plots_all_momenta(k, j=0, t2=-0.2, depth_overlap=3, p=1)
    # k = np.array([-1, 0])*np.pi
    # load_plots_all_momenta(k, j=1/3, t2=0., depth_overlap=3, p=1)
    # load_plots_all_momenta(k, j=0, t2=-0.2, depth_overlap=3, p=1)
    # plt.show()