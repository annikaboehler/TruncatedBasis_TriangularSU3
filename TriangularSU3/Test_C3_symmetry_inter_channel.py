import numpy as np
from scipy.spatial import KDTree
from pathlib import Path

def test_inversion_symmetry(k_grid, M0, M1, label="Overlaps"):
    """
    Tests inversion symmetry (k -> -k) and channel equivalence for two degenerate overlap channels.
    
    Parameters:
    -----------
    k_grid : numpy.ndarray of shape (2, N_k)
        Momentum grid coordinates [kx, ky].
    M0 : numpy.ndarray of shape (N_k,)
        First overlap channel (e.g., Ms_t2_0 or Ms_j_0).
    M1 : numpy.ndarray of shape (N_k,)
        Second overlap channel (e.g., Ms_t2_1 or Ms_j_1).
    label : str
        Label for output identification.
    """
    N_k = k_grid.shape[1]
    
    # 1. Inversion transformation: k -> -k
    k_inverted = -k_grid
    
    # 2. Match inverted k-points (-k) back to original grid using KDTree
    tree = KDTree(k_grid.T)
    distances, indices = tree.query(k_inverted.T)
    
    max_grid_dist = np.max(distances)
    if max_grid_dist > 1e-4:
        print(f"[{label}] WARNING: The underlying k_grid is not inversion invariant.")
        print(f"           Max grid displacement: {max_grid_dist:.2e}")
    
    # Evaluate channels at -k
    M0_inv = M0[indices]
    M1_inv = M1[indices]
    
    # --- Test 1: Channel 0 Inversion Self-Symmetry |M0(-k) - M0(k)| ---
    abs_diff_m0 = np.abs(M0_inv - M0)
    
    # --- Test 2: Channel Interchange Equivalence |M1(-k) - M0(k)| ---
    # In degenerate spaces, channel 0 at -k often maps to channel 1 at +k (or vice versa)
    abs_diff_cross = np.abs(M1_inv - M0)
    
    # --- Test 3: Total Subspace Invariance (|M0(-k)|^2 + |M1(-k)|^2) - (|M0(k)|^2 + |M1(k)|^2) ---
    I_k = np.abs(M0)**2 + np.abs(M1)**2
    I_inv = I_k[indices]
    abs_diff_norm = np.abs(I_inv - I_k)
    
    denom_norm = np.where(I_k < 1e-12, 1e-12, I_k)
    rel_diff_norm = abs_diff_norm / denom_norm
    valid_mask = I_k > 1e-8
    
    print(f"==================================================")
    print(f"      Inversion Symmetry (k -> -k) Test: {label}")
    print(f"==================================================")
    print(f" Max Grid Inversion Deviation : {max_grid_dist:.3e}")
    print(f" Subspace Norm Max Abs Error   : {np.max(abs_diff_norm):.3e}")
    print(f" Subspace Norm Mean Abs Error  : {np.mean(abs_diff_norm):.3e}")
    print(f" Subspace Norm Max Rel Error   : {np.max(rel_diff_norm[valid_mask]):.3e}")
    print(f" Subspace Norm Mean Rel Error  : {np.mean(rel_diff_norm[valid_mask]):.3e}")
    print(f" ------------------------------------------------")
    print(f" Channel 0 Direct Self-Error   : {np.mean(abs_diff_m0):.3e} (Mean Abs)")
    print(f" Channel Cross-Exchange Error  : {np.mean(abs_diff_cross):.3e} (Mean Abs)")
    print(f"==================================================\n")

    return abs_diff_norm


if __name__ == "__main__":
    from SU3_helper_sc_cc_overlaps import *

    L = 51
    j = 0.3
    j_perp = 0.3
    t = 1
    unit_cell = 0
    depth_sc = 4
    depth_cc = 4
    l_max_sc_overlaps = 4
    honeycomb = True
    system = 'SU2Hc_tri_grid' if honeycomb else 'SU3Tri_tri_grid'
    
    k_grid = make_triangular_grid_bz(L, grid_size=2*np.pi/np.sqrt(3) * L/(L-1))

    # Dynamically find the project root
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir.parent / "results" / "TRI" / "sc_cc_overlaps"

    file_t = results_dir / f"M_t_{system}_depth_sc={depth_sc}_depth_cc={depth_cc}_lmax_sc_overlaps={l_max_sc_overlaps}_jperp={j_perp}_j={j}_t={t}.npy"
    file_j = results_dir / f"M_j_perp_{system}_depth_sc={depth_sc}_depth_cc={depth_cc}_lmax_sc_overlaps={l_max_sc_overlaps}_jperp={j_perp}_j={j}_t={t}.npy"

    if not file_t.exists():
        print(f"❌ File not found:\n   {file_t}")
    else:
        Ms_t2_0, Ms_t2_1 = np.load(file_t)
        Ms_j_0, Ms_j_1 = np.load(file_j)

        # Run Inversion Tests
        test_inversion_symmetry(k_grid, Ms_t2_0, Ms_t2_1, label="M_{t'} Channels")
        test_inversion_symmetry(k_grid, Ms_j_0, Ms_j_1, label="M_{J_perp} Channels")