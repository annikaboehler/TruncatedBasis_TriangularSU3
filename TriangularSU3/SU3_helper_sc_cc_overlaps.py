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

def init_lattices(l_max_sc, l_max_cc, l_max_sc_overlaps, j_perp_div_j=1., connected=True, honeycomb=False):
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
    lat_sc1 = Lat_sc1(l_max_sc, only_connected=connected, honeycomb=honeycomb)
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

def compute_eigenvectors(L, j, j_perp, t2, lat_sc1: Lat_sc1, lat_sc2: Lat_sc2,  lat_cc: Lat_cc):
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
            lat_sc1.compute_H(k1[:,x,y], t=1., t2=t2, j=j)
            v = lat_sc1.eigenvec(0)
            n0 = find_l0_state(lat_sc1)
            v = v * np.exp(-1j * np.angle(v[n0]))
            vs_sc_1.append(v)

            lat_sc2.compute_H(k2[:,x,y], t=1., t2=t2, j=j)
            v = lat_sc2.eigenvec(0)
            n0 = find_l0_state(lat_sc2)
            v = v * np.exp(-1j * np.angle(v[n0]))
            vs_sc_2.append(v)
    vs_sc_1 = np.array(vs_sc_1).reshape((L, L, lat_sc1.basis.length))
    vs_sc_2 = np.array(vs_sc_2).reshape((L, L, lat_sc2.basis.length))

    lat_sc1.vs_sc_1 = vs_sc_1
    lat_sc2.vs_sc_2 = vs_sc_2
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

def add_holes(n1, n2, jx, jy, lat_sc1: Lat_sc1, lat_sc2: Lat_sc2, lat_cc: Lat_cc, too_large=False):
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
    # print(f'too_large: {too_large}')
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

    extend = 2 if too_large else 0      # allow for strings that are longer by 1 site than lat_cc.depth

    skip_config = False  
    if np.amax(np.abs(sitelist2+offset-lat_sc1.depth-1)) > lat_cc.depth + extend: #skips all sc states that are too long
        skip_config = True  
        # print(f'sitelist2={sitelist2}, offset={offset}, lat_sc1.depth+1={lat_sc1.depth+1},lat_cc.depth + extend={lat_cc.depth + extend}')
    if np.amax(np.abs(sitelist1-lat_sc1.depth-1)) > lat_cc.depth + extend: #skips all sc states that are too long
        skip_config = True   
    if np.any([np.any(np.all(sitelist1 == row, axis=1)) for row in (sitelist2 + offset)]):   #skips all strings that would overlap, since the resulting state is not represented correctly at the moment
        skip_config = True
    
                    
    if not skip_config:
        seq = []
        for move in lat_sc1.bin_basis[n1]['seq']:
            seq.append([0,move[0],move[1]])
        for move in lat_sc2.bin_basis[n2]['seq']:
            seq.append([1,move[0],move[1]])
        seq.append([1,jx,jy])
        state = -1
        if not too_large:
            lat0 = copy.deepcopy(lat_cc.Neel_state[x])
            for i in sitelist1: 
                lat0[tuple(np.array(i)-lat_sc1.depth+lat_cc.depth)] = lat1[tuple(i)]
            for i in sitelist2:
                lat0[tuple(np.array(i)+np.array(offset)-lat_sc1.depth+lat_cc.depth)] = lat2[tuple(i)]
            
            debug_lat = (lat0-lat_cc.Neel_state[x])%3
            hole_pos = [np.ones((2,), dtype=int) * (lat_cc.depth + 1), np.ones((2,), dtype=int) * (lat_cc.depth + 1) + offset]
            for xx in hole_pos:
                lat0[tuple(xx)] = 0
                debug_lat[tuple(xx)] = 3
                # print(f'hole at: {xx}')
            state = {'lat': lat0, 'hole_pos': hole_pos, 'seq': seq}
            # print(f'hole_pos_cropped: {hole_pos}')
            # print(f'lat_cropped: {debug_lat}')
            # print()
        

        #uncropped lattice for t' coupling
        sublattice = triangular_Neel_size_L(lat_cc.L_size + 2,x)
        lat0_uncropped = sublattice.copy()
        for i in sitelist1:
            lat0_uncropped[tuple(np.array(i)-lat_sc1.depth+lat_cc.depth+1)] = lat1[tuple(i)]
        for i in sitelist2:
            lat0_uncropped[tuple(np.array(i)+np.array(offset)-lat_sc1.depth+lat_cc.depth+1)] = lat2[tuple(i)]
        # print(f'sitelists: {sitelist1}, {sitelist2}, offset: {offset}')
        
        debug_lat_uncropped = (lat0_uncropped - sublattice)%3
        # print(f'lat_uncropped: {debug_lat_uncropped}')
        hole_pos_uncropped = [np.ones((2,), dtype=int) * (lat_cc.depth + 2), np.ones((2,), dtype=int) * (lat_cc.depth + 2) + offset]
        for xx in hole_pos_uncropped:
            lat0_uncropped[tuple(xx)] = 0
            debug_lat_uncropped[tuple(xx)] = 3
            # print(f'hole at: {xx}')
        state_uncropped = {'lat': lat0_uncropped, 'hole_pos': hole_pos_uncropped, 'seq': seq}
        # print(f'hole_pos_uncropped: {hole_pos_uncropped}')
        # print(f'lat_uncropped: {debug_lat_uncropped}')

        # ### plot states for debugging
        # ms = -1
        # found,_ = lat_cc.basis.search(lat_cc.state_2_list_entry(state))
        # if not found: 
        #     transformed_basis = transform_lattice_j_perp(lat_cc)
        #     found, m_t = transformed_basis.search(lat_cc.state_2_list_entry(state))
        #     if found:
        #         ms = transformed_basis.list[m_t][lat_cc.L_size + 2:]
        # if not too_large:
        #     lat_cc_1 = (lat0-lat_cc.Neel_state[x])%3
        #     lat_cc_1[tuple(hole_pos[0])] = 3
        #     lat_cc_1[tuple(hole_pos[1])] = 4

        # else:
        #     lat_cc_1 = (lat0_uncropped-sublattice)%3
        #     lat_cc_1[tuple(hole_pos_uncropped[0])] = 3
        #     lat_cc_1[tuple(hole_pos_uncropped[1])] = 4

        # lat_sc_1 = latsc1.copy()
        # lat_sc_1[lat_sc1.depth+1,lat_sc1.depth+1] = 3
        # lat_sc_2 = latsc2.copy()
        # lat_sc_1[lat_sc1.depth+1,lat_sc1.depth+1] = 3
        
        # lats = []
        # lats.append(lat_sc_1)
        # lats.append(lat_sc_2)
        # lats.append(lat_cc_1)
        # # print(f'latsc1:{lat_sc1}')
        # # print(f'latsc2:{lat_sc2}')
        # # print(f'latcc:{lat_cc1}')

        # import matplotlib.pyplot as plt
        # from matplotlib.colors import ListedColormap
        # colors = ['pink', 'blue', 'green', 'black', 'white']
        # custom_cmap = ListedColormap(colors)
        # fig, axs = plt.subplots(1,3, figsize=(8,3))
        # for i in range(3):
        #     # seq = lat_sc.bin_basis[indices[i]]['seq']
        #     # seq = tuple(move.tolist() for move in seq)
        #     axs[i].imshow(lats[i], origin='lower',cmap=custom_cmap)
        # axs[1].set_title(f'n1: {n1}, n2: {n2}, offset: {offset}, ms: {ms}')
    else:
        # lat0 = copy.deepcopy(lat_cc.Neel_state[x])
        # seq = []
        # for i in sitelist1:
        #     lat0[tuple(np.array(i)-lat_sc1.depth+lat_cc.depth)] = lat1[tuple(i)]
        # for i in sitelist2:
        #     lat0[tuple(np.array(i)+np.array(offset)-lat_sc1.depth+lat_cc.depth)] = lat2[tuple(i)]
        # hole_pos = [np.ones((2,), dtype=int) * (lat_cc.depth + 1), np.ones((2,), dtype=int) * (lat_cc.depth + 1) + offset]


        # lat_cc_1 = (lat0-lat_cc.Neel_state[x])%3
        # lat_cc_1[tuple(hole_pos[0])] = 3
        # lat_cc_1[tuple(hole_pos[1])] = 4
        # # print(lat_cc_1)
        # lat_sc_1 = latsc1.copy()
        # lat_sc_1[lat_sc1.depth+1,lat_sc1.depth+1] = 3
        # lat_sc_2 = latsc2.copy()
        # lat_sc_1[lat_sc1.depth+1,lat_sc1.depth+1] = 3

        # lats = []
        # lats.append(lat_sc_1)
        # lats.append(lat_sc_2)
        # lats.append(lat_cc_1)

        # # print(f'latsc1:{lat_sc1}')
        # # print(f'latsc2:{lat_sc2}')
        # # print(f'latcc:{lat_cc1}')

        # import matplotlib.pyplot as plt
        # from matplotlib.colors import ListedColormap
        # colors = ['pink', 'blue', 'green', 'black', 'white']
        # custom_cmap = ListedColormap(colors)
        # fig, axs = plt.subplots(1,3, figsize=(8,3))
        # for i in range(3):
        #     # seq = lat_sc.bin_basis[indices[i]]['seq']
        #     # seq = tuple(move.tolist() for move in seq)
        #     axs[i].imshow(lats[i], origin='lower',cmap=custom_cmap)
        # phys_dist = np.zeros(2)
        # dist = offset
        # phys_dist[0], phys_dist[1] = np.sqrt(3)/2*dist[0], dist[1] - 1/2*dist[0]
        # phys_dist = np.linalg.norm(phys_dist)
        # plt.title(f'n1: {n1}, n2: {n2}, offset: {offset}, |offset|={phys_dist:.3f}') 


        state = -1 #not a dictionary, will get skipped  
        state_uncropped = -1      
    return state, state_uncropped

def triangular_Neel_size_L(L,sl):
        
        if L%2 != 1:
            raise Exception(f'L = {L} not possible, Neel state lattice must have odd number of sites') 
        size = int((L-3)/2)
        triangular_lattice = np.zeros((L, L), dtype=int)
        for i in range(L):
            for j in range(L):
                if (i+j+sl+size+1) % 3 == 0: # +self.depth+1 so center site is same for all lattice sizes
                    triangular_lattice[i,j] = 0
                elif (i+j+sl+size+1) % 3 == 1:
                    triangular_lattice[i,j] = 1
                else:
                    triangular_lattice[i,j] = 2
        triangular_lattice[size+1,size+1] = 0
        #print(triangular_lattice)
        return triangular_lattice

def cc_translation_size_L(L, lat, hole_pos, step, check=False):         #can only be used when step is in seq_hole_1  #Linus 1.0
        if step[0] != 0:
            print('wrong hole (translation)')
        x = step[1]
        y = step[2]

        if x > 0:
            lat[:x, :] = (lat[:x, :]-(L-3)/2)%3
        elif x < 0:
            lat[x:, :] = (lat[x:, :]+(L-3)/2)%3
        lat = np.roll(lat, -x, axis=0)

        if y > 0:
            lat[:,:y] = (lat[:,:y]-(L-3)/2)%3
        elif y < 0:
            lat[:,y:] = (lat[:,y:]+(L-3)/2)%3
        lat = np.roll(lat, -y, axis=1)

        hole_pos[0] = hole_pos[0] - step[1:]        #move 1st hole accordingly
        hole_pos[1] = hole_pos[1] - step[1:]        #move 2nd hole accordingly
        return lat, hole_pos

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

    lat2 = lat_sc2.bin_basis[n2]['lat'].copy()
    subl2 = lat_sc2.find_sublattice(lat_sc2.bin_basis[n2])
    latsc2 = (lat2-subl2)%3 
    latsc2[lat_sc2.depth+1,lat_sc2.depth+1] = 3 #to include hole in sitelist
    sitelist2 = np.array(np.argwhere(latsc2).tolist())
    offset = np.array([jx, jy])    
    x = lat_sc1.find_hole_sublattice(lat_sc1.bin_basis[n1]['seq'])

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
        lat0 = sublattice.copy()
        seq = [[0,x,0]]
        for i in sitelist1:
            lat0[tuple(np.array(i)-lat_sc1.depth+lat_cc.depth+2)] = lat1[tuple(i)]
        lat0[lat_cc.depth+3, lat_cc.depth+3] = 0     #mark hole site as 0
        for i in sitelist2:
            lat0[tuple(np.array(i)+np.array(offset)-lat_sc1.depth+lat_cc.depth+2)] = lat2[tuple(i)]
        lat0[tuple(np.array(offset)+lat_cc.depth+3)] = 0     #mark hole site as 0
        latcc = (lat0-sublattice)%3
        hole_pos = [np.ones((2,), dtype=int) * (lat_cc.depth + 1), offset + lat_cc.depth + 1]
        for xx in hole_pos:
            lat0[tuple(xx+2)] = 0      #set holes to zero
            latcc[tuple(xx+2)] = 0
        siteslist = np.argwhere(latcc).tolist()
        for site in siteslist:
            nx = [[1,0],[0,1],[1,1]]
            for nn in nx:
                if [site[0]+nn[0],site[1]+nn[1]] in siteslist:
                    lat1 = lat0.copy()
                    lat1[site[0],site[1]],lat1[site[0]+nn[0],site[1]+nn[1]]=lat1[site[0]+nn[0],site[1]+nn[1]],lat1[site[0],site[1]] #exchange both sites

                    #check if any spins are flipped outside L_size 
                    lat2 = lat1.copy()
                    lat2 = (lat2-sublattice)%3
                    lat2[2:-2, 2:-2] = 0 
                    if not np.all(lat2 == 0):
                        continue
                    
                    lat1 = lat1[2:-2, 2:-2]  #crop lattice back to cc size
                    state1 = {'lat': lat1, 'seq':seq, 'hole_pos': hole_pos}
                    list_entry = lat_cc.state_2_list_entry(state1)
                    found, m = lat_cc.basis.search(list_entry)
                    if found:
                        ms.append(m)

                        # #### plot states for debugging
                        # latcc1 = (lat_cc.bin_basis[m]['lat']-lat_cc.Neel_state[x])%3
                        # latcc1[tuple(hole_pos[0])] = 3
                        # latcc1[tuple(hole_pos[1])] = 4
                        # latsc1_ = latsc1.copy()
                        # latsc1_[lat_sc1.depth+1,lat_sc1.depth+1] = 3
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
                    # new = transformed_basis.add(list_entry + [transformed_basis.length])
                    # print(transformed_basis.length)
                    # new = transformed_basis.add(list_entry)
                    # if not new:
                    #     print(f'redundent cc state in trafo_basis, seq:{seq}')
                    _, n = transformed_basis.search(list_entry)
                    transformed_basis.list[n].append(i) #state n in transformed basis can have multiple original states leading to it
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
    # print(f'elements in transformed_basis.list N={N}')
    return transformed_basis

def overlap_all_momenta(j, L, t2, lat_sc1: Lat_sc1, lat_sc2: Lat_sc2, lat_cc: Lat_cc, p=-1):
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
    vs_sc_1 = lat_sc1.vs.reshape((L, L, 1, 1, lat_sc1.basis.length))
    vs_sc_2 = lat_sc2.vs.reshape((1, 1, L, L, lat_sc2.basis.length))
    v_cc = lat_cc.vs

    l_max = lat_cc.depth

    transformed_basis = transform_lattice_j_perp(lat_cc)
    c2 = 0
    c3 = 0
    c4 = 0
    c5 = 0
    c6 = 0
    Ms = np.zeros((L, L, L, L))
    for n1 in lat_sc1.indices1:
        for n2 in lat_sc2.indices2:
            
            j_min = (np.ones(2,)*(-lat_cc.depth)).astype(int) #improve these bounds in the future
            j_max = (np.ones(2,)*(lat_cc.depth)).astype(int)

            seq1 = lat_sc1.bin_basis[n1]['seq']
            seq_hole_1 = []
            if len(seq1) != 0:
                for i, step in enumerate(seq1):
                    seq_hole_1.append([0,step[0],step[1]]) #bring seq_hole_1 in the form [[0,x,y],...] add to dist_2_phys_dist
    
            for jx in range(j_min[0], j_max[0] + 1):        
                for jy in range(j_min[1], j_max[1] + 1):
                    # apply S ^(+)_i S^(+)_j on l.h.s. of expectation value, i.e.
                    # <cc|SS
                    state = add_holes(n1, n2, jx, jy, lat_sc1, lat_sc2, lat_cc)

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