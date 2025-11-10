import numpy as np
from time import perf_counter
from SU3_sc_cc_overlap import init_lattices, compute_eigenvectors_all_momenta, overlap_all_momenta
import os.path
import argparse

def main(args):
    j = args['j']
    j_perp = args['j_perp']
    t2 = args['t2']
    depth_sc = args['depth_sc']
    depth_cc = args['depth_cc']
    l_max_sc_overlaps = args['depth_overlaps']
    L = args['L']
    p = args['p']

    t0 = perf_counter()
    lats = init_lattices(depth_sc, depth_cc, l_max_sc_overlaps, j_perp_div_j=j_perp/j, connected=True)
    print(f'initialized lattices in {perf_counter() - t0:.2f} seconds')

    t0 = perf_counter()
    compute_eigenvectors_all_momenta(L, j, j_perp, t2, *lats, p=p)
    print(f'coputed eigenvectors in {perf_counter() - t0:.2f} seconds')

    path = '/project/th-scratch/p/Pit.Bermes/tj/data'

    # t0 = perf_counter()
    # Ms = overlap_grid(j, L, t2, *lats)
    # print(f'Computed overlap on grid in {perf_counter() - t0:.2f} seconds')
    # np.save(os.path.join(path, f'overlaps_{l_max_sc_overlaps}'+dop), Ms)

    t0 = perf_counter()
    Ms_j, Ms_t = overlap_all_momenta(j, L, t2, *lats, p)
    if p == -1:
        s = 'fer'
    elif p == 1:
        s = 'bos'
    print(f'Computed overlaps on grid in {perf_counter() - t0:.2f} seconds')
    np.save(os.path.join(path, f'overlaps_{s}_t_{l_max_sc_overlaps}_jperp{j_perp/j:.2f}_t2{t2:.2f}'), Ms_t)
    np.save(os.path.join(path, f'overlaps_{s}_j_{l_max_sc_overlaps}_jperp{j_perp/j:.2f}_t2{t2:.2f}'), Ms_j)

if __name__ == "__main__":
### implement argument parsing
    parser = argparse.ArgumentParser(description="Compute matrix element for sc-cc Feshbach calculations")
    parser.add_argument("-j", type=float,
                        help="spin interaction strength along z axis",
                        default=1/3)
    parser.add_argument("-j_perp", type=float,
                        help="spin interaction strength along x and y axis",
                        default=1/3)
    parser.add_argument("-t2", type=float,
                        help="NNN hopping constant",
                        default=0.)
    parser.add_argument("-d_sc", "--depth_sc", type=int, 
                        help="maximal string length of spinon-chargon basis",
                        default=10)
    parser.add_argument("-d_cc", "--depth_cc", type=int, 
                        help="maximal string length of chargon-chargon basis",
                        default=10)
    parser.add_argument("-d_o", "--depth_overlaps", type=int, 
                        help="maximal string length of spinon-chargon considered in overlaps",
                        default=3)
    parser.add_argument('-L', type=int,
                        help='momnetum resolution along one axis',
                        default=16)
    parser.add_argument('-p', type=int,
                        help='parity of quasiparticles, i.e. +1 for bosons and -1 for fermions.',
                        default=-1)

    args = vars(parser.parse_args())

    if not (args['p'] == -1 or args['p'] == 1):
        raise ValueError('parity eigenvalue must be +1 or -1')
    
    main(args)