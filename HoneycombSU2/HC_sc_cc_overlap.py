'''compute overlap between 2 sc-pairs and cc'''
import numpy as np
import matplotlib.pyplot as plt
import os.path
from HC_1_hole import StringBasisHC as basis_sc
from HC_2_holes import sorted_list
from HC_2_holes import StringBasis as basis_cc

def plot_lat(state, ax=None, end=False, fig=None, axs=None):
    if isinstance(state, dict):
        lat = state['lat'].astype(float)
        if 'hole_pos' in state.keys():
            lat[state['hole_pos'][0][0], state['hole_pos'][0][1]] = 0.5
            lat[state['hole_pos'][1][0], state['hole_pos'][1][1]] = 0.5
        else: 
            lat[lat.shape[0]//2, lat.shape[1]//2] = 0.5
    else:
        lat = state.astype(float)
    if not ax is None:
        axs[ax].imshow(lat, vmin=0, vmax=1)
        if end:
            plt.show()
    else:
        plt.imshow(lat, vmin=0, vmax=1)
        plt.show()

def index2momentum(i, Lx, Ly=0):
    if Ly==0:
        L = Lx
    else:
        assert i.shape[0] == 2
        L = np.array([Lx, Ly]).reshape((2,) + (len(i.shape)-1)*(1,))
    return (1.2*np.pi*(2*i/(L)-1))

def sum_ind(i, j, Lx, Ly=0):
    if Ly==0:
        L = Lx
    else:
        assert i.shape[0] == 2
        assert j.shape[0] == 2
        L = np.array([Lx, Ly]).reshape((2,) + (len(i.shape)-1)*(1,))
    return np.mod(np.round(i+j-L/2), L).astype(int)

def brick_to_hc_distance(d):
        ''' converts distance dx, dy on brickwall lattice to distance r1, r2 on honeycomb lattice '''
        assert np.sum(d)%2 == 0 #check that d is already resized to same sl distance
        r1 = -d[0]
        r2 = (d[1]+d[0])//2
        return r1, r2

def find_l0_state(lat: basis_sc, n_max=7):
    '''
    Find the first energy level which has considerable quasiparticle weight i.e. overlap with the zero string-length state.

    ARGUMENTS:
    lat: Lat_sc: an instance of the sc lattice class, where the Hamiltonian has already been computed.
    n_max : int (optional): the highest energy level considered

    OUTPUT:
    n: int: gives the number of the first energy level correspondig to the l=0 state. (n=0 corresponds to the ground state)
    '''
    _, vs = lat.eigensys(n_max, full=True)
    vs = np.abs(vs[:2, :])
    n = 0
    while n <= n_max:
        if vs[0, n] > vs[1, n] and vs[0, n] > 1e-4:
            break
        else:
            n += 1
    return n

def init_lattices(l_max_sc, l_max_cc, l_max_sc_overlaps, j_perp_div_j=1., connected=True):
    """
    initialize and construct truncated basis for sc and cc.
    save indices of sc_basis, where string is smaller (or equal) than l_max_sc_overlaps
    Also compute Hamiltonian matrix.
    ------------
    ARGUMENTS:
    l_max_sc (int): maximal string length in sc truncated basis
    l_max_cc (int): maximal string length in cc truncated basis
    l_max_sc_overlaps (int): maximal string lemgth considered in overlaps
    connected (bool): if True, we take only the connected strings into account
    -------------
    RETURNS:
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    """
    lat_sc_1 = basis_sc(l_max_sc, only_connected=connected, initial_sl=0)
    lat_sc_2 = basis_sc(l_max_sc, only_connected=connected, initial_sl=1)

    lat_sc_1.indices = list(np.argwhere(np.count_nonzero([x[0] for x in lat_sc_1.bin_basis], axis=(1,2)) <= l_max_sc_overlaps).flatten())
    lat_sc_1.indices = list(np.argwhere(np.count_nonzero([x[0] for x in lat_sc_2.bin_basis], axis=(1,2)) <= l_max_sc_overlaps).flatten())

    lat_cc = basis_cc(l_max_cc, only_connected=connected)

    print('initialized lattices')
    print(f'dim(H_sc_1) = {lat_sc_1.basis.length}')
    print(f'dim(H_sc_2) = {lat_sc_2.basis.length}')
    print(f'dim(H_cc) = {lat_cc.basis.length}')

    return lat_sc_1, lat_sc_2, lat_cc

def compute_eigenvectors_all_momenta(Lx, j, j_perp, t2, lat_sc: basis_sc, lat_cc: basis_cc, Ly=0, p=-1):
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
            lat_sc.compute_H(k1[:,x,y], 1., t2=t2, j=j)
            v = lat_sc.eigenvec(0)
            #n0 = find_l0_state(lat_sc) ?????????
            #v = v * np.exp(-1j * np.angle(v[n0])) # fix phase (set phase of l=0 state to zero)
            vs_sc.append(v)

            lat_cc.compute_H(k1[:,x,y], 1., j, j_perp, p=p)
            #v_cc = lat_cc.eigenvec(0, v0=v0) v0 ?????????
            v_cc = lat_cc.eigenvec(0)
            #v_cc = v_cc * np.exp(-1j * np.angle(v_cc[0])) # fix phase (set phase of l=0 state to zero)
            vs_cc.append(v_cc)

    vs_sc = np.array(vs_sc).reshape((Lx, Ly, lat_sc.basis.length))
    vs_cc = np.array(vs_cc).reshape((Lx, Ly, len(lat_cc.representatives)))

    lat_sc.vs = vs_sc
    lat_cc.vs = vs_cc
    print('computed eigenvectors')

def compute_eigensys_all_momenta_exc(L, j, t2, n_sc, n_cc, lat_sc: basis_sc, lat_cc: basis_cc, p=-1):
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
    

def add_holes_hc(state1, state2, hole_pos_2, lat_sc_1, lat_sc_2, lat_cc):
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
    j_max = np.amax(np.abs(hole_pos_2))
    jy = hole_pos_2[0]
    jx = hole_pos_2[1]
    sl_0 = state1['sl']
    sl_1 = state2['sl']
    lat1 = state1['lat']
    lat2 = state2['lat']
    
    lat1 = np.pad(lat1, (j_max, j_max), 'constant', constant_values=False) #pads such that hole 1 sits at center
    lat2 = np.pad(lat2, ((j_max + jy, j_max - jy), (j_max + jx, j_max - jx)), 'constant', constant_values=False) #pads such that hole 2 sites at jx,jy
    
    lat = np.logical_xor(lat1, lat2)
    # now crop lat so that it has the same dimension as lattices in Lat_cc
    dx = lat_sc_1.depth + j_max - lat_cc.depth
    if dx > 0: #new lattice is too big, need to crop
        count = np.count_nonzero(lat[0:dx,:]) + np.count_nonzero(lat[-dx:,:]) + np.count_nonzero(lat[:,0:dx]) + np.count_nonzero(lat[:,-dx:]) #count strings outside cropped region
        if count > 0: #if strings outside, state not possible
            return -1
        else:
            lat = lat[dx:-dx, dx:-dx]
    elif dx < 0:
        lat = np.pad(lat, (-dx, -dx), 'constant', constant_values=False)
    hole_pos = [np.ones((2,), dtype=int) * (lat_cc.depth + 1), np.ones((2,), dtype=int) * (lat_cc.depth + 1) + hole_pos_2] #holes at center and at hole_pos_2
    for x in hole_pos:
        lat[tuple(x)] = False #set hole positions to False (need to add sl here)
    state = {'lat': lat, 'hole_pos': hole_pos, 'sl': [sl_0, sl_1]}
    return state

def add_holes_hc_hopping(state1, state2, hole_pos_2, lat_sc_1, lat_sc_2, lat_cc):
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
    j_max = np.amax(np.abs(hole_pos_2))
    jy = hole_pos_2[0]
    jx = hole_pos_2[1]
    sl_0 = state1['sl']
    sl_1 = state2['sl']
    lat1 = state1['lat']
    lat2 = state2['lat']
    
    lat1 = np.pad(lat1, (j_max, j_max), 'constant', constant_values=False) #pads such that hole 1 sits at center
    lat2 = np.pad(lat2, ((j_max + jy, j_max - jy), (j_max + jx, j_max - jx)), 'constant', constant_values=False) #pads such that hole 2 sites at jx,jy
    
    lat = np.logical_xor(lat1, lat2)
    # now crop lat so that it has the same dimension as lattices in Lat_cc
    dx = lat_sc_1.depth + j_max - (lat_cc.depth+1) #make cc lattice one site bigger because need to include hoppings from unit cells at +/- a2
    if dx > 0: #new lattice is too big, need to crop
        count = np.count_nonzero(lat[0:dx,:]) + np.count_nonzero(lat[-dx:,:]) + np.count_nonzero(lat[:,0:dx]) + np.count_nonzero(lat[:,-dx:]) #count strings outside cropped region
        if count > 0: #if strings outside, state not possible
            return -1, -1
        else:
            lat = lat[dx:-dx, dx:-dx]
    elif dx < 0:
        lat = np.pad(lat, (-dx, -dx), 'constant', constant_values=False)
    hole_pos = [np.ones((2,), dtype=int) * (lat_cc.depth + 2), np.ones((2,), dtype=int) * (lat_cc.depth + 2) + hole_pos_2] #holes at center and at hole_pos_2
    for x in hole_pos:
        lat[tuple(x)] = False #set hole positions to False (need to add sl here)
    state = {'lat': lat, 'hole_pos': hole_pos, 'sl': [sl_0, sl_1]}
    lat_phys = lat[1:-1, 1:-1]
    state_phys = {'lat': lat_phys, 'hole_pos': hole_pos, 'sl': [sl_0, sl_1]}
    return state, state_phys #this is now one site larger than lat_cc states at each axis!

def crop_lattice(lat, depth_sc, depth_cc, hole_pos_2, sl):
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
    state = {'lat': lat, 'hole_pos': hole_pos, 'sl': sl}
    return state

def add_holes_j_perp(state1, state2, hole_pos_2, lat_sc_1, lat_sc_2, lat_cc, plot=False):
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
    j_max = np.amax(np.abs(hole_pos_2))
    jy = hole_pos_2[0]
    jx = hole_pos_2[1]
    lat1 = state1['lat']
    lat2 = state2['lat']
    sl_0 = state1['sl']
    sl_1 = state2['sl']
    lat1 = np.pad(lat1, (j_max, j_max), 'constant', constant_values=False)
    lat2 = np.pad(lat2, ((j_max + jy, j_max - jy), (j_max + jx, j_max - jx)), 'constant', constant_values=False)
    lat = np.logical_xor(lat1, lat2)


    hole_pos = [np.ones((2,), dtype=int) * (j_max + lat_sc_1.depth + 1), np.ones((2,), dtype=int) * (j_max + lat_sc_1.depth + 1) + hole_pos_2]
    for x in hole_pos:
        lat[tuple(x)] = False
    state = {'lat': lat, 'hole_pos': hole_pos, 'sl': [sl_0, sl_1]}
    state_prev = {'lat': lat, 'hole_pos': hole_pos, 'sl': [sl_0, sl_1]}
    ms = []

    siteslist = list(np.argwhere(lat))
    for site in siteslist:
        lat_temp = lat.copy()
        if (np.sum(site-hole_pos[0])+sl_0)%2==1: #bond upwards
            if lat_temp[site[0], site[1]] and lat_temp[site[0]+1, site[1]]:
                lat_temp[site[0], site[1]] = False
                lat_temp[site[0]+1 ,site[1]] = False
                state = crop_lattice(lat_temp, lat_sc_1.depth, lat_cc.depth, hole_pos_2, [sl_0, sl_1])
                if type(state)==dict:
                    found, m = lat_cc.basis.search(lat_cc.state_2_list_entry(state))
                    if found:
                        if plot:
                            fig, axs = plt.subplots(1,4, figsize=(12,4))
                            plot_lat(state1, ax=0, axs=axs, fig=fig)
                            plot_lat(state2, ax=1, axs=axs, fig=fig)
                            plot_lat(state_prev, ax=2, axs=axs, fig=fig)
                            plot_lat(state, ax=3, axs=axs, fig=fig, end=True)
                        ms.append(m)

        lat_temp = lat.copy()
        if lat_temp[site[0], site[1]] and lat_temp[site[0], site[1]+1]:
            lat_temp[site[0], site[1]] = False
            lat_temp[site[0], site[1]+1] = False
            state = crop_lattice(lat_temp, lat_sc_1.depth, lat_cc.depth, hole_pos_2, [sl_0, sl_1])
            if type(state)==dict:
                #plot_lat(state)
                found, m = lat_cc.basis.search(lat_cc.state_2_list_entry(state))
                if found:
                    if plot:
                        fig, axs = plt.subplots(1,4, figsize=(12,4))
                        plot_lat(state1, ax=0, axs=axs, fig=fig)
                        plot_lat(state2, ax=1, axs=axs, fig=fig)
                        plot_lat(state_prev, ax=2, axs=axs, fig=fig)
                        plot_lat(state, ax=3, axs=axs, fig=fig, end=True)
                    #print(m)
                    ms.append(m)
    return ms

def transform_hc_lattice_j_perp(lat_cc):
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
    transformed_basis = sorted_list(length_arr = lat_cc.L_size + 3)

    for i in range(lat_cc.basis.length):
        state = lat_cc.bin_basis[i]
        lat = state['lat']
        hole_pos = state['hole_pos']
        sl = state['sl']

        siteslist = list(np.argwhere(lat))

        for site in siteslist:
            lat1 = lat.copy()
            if (lat_cc.L_size*site[0]+site[1]+sl[0])%2==1: #bond downwards
                if lat1[site[0], site[1]] and lat1[site[0]+1, site[1]]: #flip along y
                    lat1[site[0], site[1]] = False
                    lat1[site[0]+1, site[1]] = False
                    a = lat1*np.matmul(np.ones((lat_cc.L_size, 1), dtype=np.uint32),
                        np.reshape(2**np.arange(lat_cc.L_size, dtype=np.uint32), (1, lat_cc.L_size)), dtype=np.uint32)
                    _ = transformed_basis.add((np.sum(a, axis=1).tolist() + [list(pos) for pos in hole_pos] +[sl]))
                    _, n = transformed_basis.search(np.sum(a, axis=1).tolist() + [list(pos) for pos in hole_pos] +[sl])
                    transformed_basis.list[n].append(i)
                    
                    lat1[site[0], site[1]] = True
                    lat1[site[0]+1, site[1]] = True
            
            if lat1[site[0], site[1]] and lat1[site[0],site[1]+1]:
                lat1[site[0], site[1]] = False
                lat1[site[0], site[1]+1] = False
                new_state = {'lat': lat1, 'hole_pos': hole_pos, 'sl': sl}
                a = lat1*np.matmul(np.ones((lat_cc.L_size, 1), dtype=np.uint32), np.reshape(2**np.arange(lat_cc.L_size, dtype=np.uint32), (1, lat_cc.L_size)), dtype=np.uint32)
                _ = transformed_basis.add(np.sum(a, axis=1).tolist() + [list(pos) for pos in hole_pos]+[sl])
                _, n = transformed_basis.search(np.sum(a, axis=1).tolist() + [list(pos) for pos in hole_pos]+[sl])
                transformed_basis.list[n].append(i)
    

    return transformed_basis


def overlap_all_momenta(j, L, t2, lat_sc_1, lat_sc_2, lat_cc, p=-1):
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
    k1 = mom.reshape((2, L, L, 1, 1)) #2 LxL grids: x momentum, y momentum
    k2 = mom.reshape((2, 1, 1, L, L)) #2 LxL grids: x momentum, y momentum
    k_plus_p_ind = sum_ind(mom_ind.reshape((2,L,L,1,1)), mom_ind.reshape((2,1,1,L,L)), L)

    # vs_sc_1, vs_sc_2, v_cc = compute_eigenvectors(L, j, j_perp, t2, lat_sc, lat_cc)
    vs_sc_1 = lat_sc_1.vs.reshape((L, L, 1, 1, lat_sc_1.basis.length))
    vs_sc_2 = lat_sc_2.vs.reshape((1, 1, L, L, lat_sc_2.basis.length))
    v_cc = lat_cc.vs

    l_max = lat_cc.depth

    transformed_basis = transform_hc_lattice_j_perp(lat_cc)

    Ms = np.zeros((L, L, L, L))
    for n1 in lat_sc_1.indices:
        for n2 in lat_sc_2.indices:
            state1 = lat_sc_1.bin_basis[n1]
            state2 = lat_sc_2.bin_basis[n2]
            lat1 = state1['lat']
            lat2 = state2['lat']
            sl1 = state1['sl']
            sl2 = state2['sl']

            # determine area in which the second hole can be placed to form a cc pair
            temp = np.argwhere(lat1) - lat_sc_1.depth - 1
            temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
            lower_bound_1 = np.amin(temp, axis=0)
            upper_bound_1 = np.amax(temp, axis=0)

            temp = np.argwhere(lat2) - lat_sc_2.depth - 1
            temp = np.concatenate((np.zeros((1,2), dtype=int), temp), axis=0)
            lower_bound_2 = np.amin(temp, axis=0)
            upper_bound_2 = np.amax(temp, axis=0)

            j_min = np.maximum(lower_bound_1 - upper_bound_2 - 4, lower_bound_2 - l_max)
            j_max = np.minimum(upper_bound_1 - lower_bound_2 + 4, l_max - upper_bound_2)

            for jx in range(j_min[0], j_max[0] + 1):
                for jy in range(j_min[1], j_max[1] + 1):
                    if jx == 0 and jy == 0: #holes cannot be on same site
                        continue
                    if sl1 == sl2 and (jx + jy) % 2 != 0: #same sublattice, must be an even number fo sites apart
                        continue
                    if sl1 != sl2 and (jx + jy) % 2 == 0: #different sublattice, must be an odd number fo sites apart
                        continue
                    
                    jy_hc = jy
                    if sl1 != sl2:
                        jy_hc += 2*(sl1-0.5)
                    jx_hc = jx
                    d_hc = np.array([jx_hc, jy_hc])
                    r1, r2 = brick_to_hc_distance(d_hc)
                    rx = np.sqrt(3)*r2 + np.sqrt(3)/2*r1
                    ry = 3/2*r1

                    # apply S ^(+)_i S^(+)_j on l.h.s. of expectation value, i.e.
                    # <cc|SS
                    state = add_holes_hc_test(n1, n2, np.array([jx, jy]), lat_sc_1, lat_sc_2, lat_cc)
                    if type(state)==dict:
                        found, _ = lat_cc.basis.search(lat_cc.state_2_list_entry(state))
                    else:
                        found = False
                    if not found:
                        if type(state)== dict:
                            found, m_t = transformed_basis.search(lat_cc.state_2_list_entry(state))
                            if found:
                                ms = transformed_basis.list[m_t][lat_cc.L_size + 2:]
                                for m in ms: #can more than one state lead to same transformed state?
                                    repr, _, m = lat_cc.is_representative[m]
                                    dM = 0.5 * 1/np.sqrt(2) * np.conj(v_cc[k_plus_p_ind[0,:,:,:,:], k_plus_p_ind[1,:,:,:,:], m]) * vs_sc_1[:,:,:,:,n1] * vs_sc_2[:,:,:,:,n2] * np.exp(1j * np.einsum('nabcd,n->abcd', k2, np.array([rx, ry])))
                                    # factor 1/sqrt(2) comes from projection onto fermionic states
                                    # factor 1/2 comes from H_J_perp
                                    if not repr:
                                        dM *= -1 * np.exp(-1j * np.einsum('nabcd,n->abcd', k1 + k2, np.array([jx, jy])))
                                    Ms = Ms + dM
                        # apply S ^(-)_i S^(-)_j on r.h.s. of expectation value, i.e.
                        # SS|(sc)^2>
                        ms = add_holes_j_perp(n1, n2, np.array([jx, jy]), lat_sc_1, lat_sc_2, lat_cc)
                        for m in ms:
                            repr, _, m = lat_cc.is_representative[m]
                            dM = 0.5 * 1/np.sqrt(2) * np.conj(v_cc[k_plus_p_ind[0,:,:,:,:], k_plus_p_ind[1,:,:,:,:], m]) * vs_sc_1[:,:,:,:,n1] * vs_sc_2[:,:,:,:,n2] * np.exp(1j * np.einsum('nabcd,n->abcd', k2, np.array([rx, ry])))
                            # factor 1/sqrt(2) comes from projection onto fermionic states
                            # factor 1/2 comes from H_J_perp
                            if not repr:
                                dM *= p * np.exp(-1j * np.einsum('nabcd,n->abcd', k1 + k2, np.array([rx, ry])))
                            Ms = Ms + dM
    Ms_j = np.pad(Ms, (0,1), mode='wrap')
    return np.real_if_close(Ms_j)

