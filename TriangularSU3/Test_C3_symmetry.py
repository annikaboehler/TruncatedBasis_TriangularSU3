import numpy as np
from scipy.spatial import KDTree
from pathlib import Path

def test_C3_symmetry(k_grid, M_data, label="Overlap"):
    """
    Tests C_3 rotational symmetry for overlap data defined on a 2D momentum grid.
    """
    N_k = k_grid.shape[1]
    
    # 120-degree (C3) rotation matrix
    theta = 2.0 * np.pi / 3.0
    R_C3 = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    k_rotated = R_C3 @ k_grid
    tree = KDTree(k_grid.T)
    distances, indices = tree.query(k_rotated.T)
    
    max_grid_dist = np.max(distances)
    if max_grid_dist > 1e-4:
        print(f"[{label}] WARNING: The underlying k_grid is not C3 invariant.")
        print(f"           Max grid point displacement: {max_grid_dist:.2e}")
    
    M_rot = M_data[indices]
    abs_diff = np.abs(M_rot - M_data)
    
    denom = np.abs(M_data)
    denom_safe = np.where(denom < 1e-12, 1e-12, denom)
    rel_diff = abs_diff / denom_safe
    
    valid_mask = denom > 1e-8
    max_abs_err = np.max(abs_diff)
    mean_abs_err = np.mean(abs_diff)
    max_rel_err = np.max(rel_diff[valid_mask]) if np.any(valid_mask) else 0.0
    mean_rel_err = np.mean(rel_diff[valid_mask]) if np.any(valid_mask) else 0.0
    
    print(f"==================================================")
    print(f"        C_3 Symmetry Precision Test: {label}")
    print(f"==================================================")
    print(f" Max Grid Rotation Deviation: {max_grid_dist:.3e}")
    print(f" Max Absolute Error        : {max_abs_err:.3e}")
    print(f" Mean Absolute Error       : {mean_abs_err:.3e}")
    print(f" Max Relative Error        : {max_rel_err:.3e}")
    print(f" Mean Relative Error       : {mean_rel_err:.3e}")
    print(f"==================================================\n")
    
    return abs_diff, rel_diff


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

    # Dynamically find the project root (one directory up from where this script lives)
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir.parent / "results" / "TRI" / "sc_cc_overlaps"

    file_t = results_dir / f"M_t_{system}_depth_sc={depth_sc}_depth_cc={depth_cc}_lmax_sc_overlaps={l_max_sc_overlaps}_jperp={j_perp}_j={j}_t={t}.npy"
    file_j = results_dir / f"M_j_perp_{system}_depth_sc={depth_sc}_depth_cc={depth_cc}_lmax_sc_overlaps={l_max_sc_overlaps}_jperp={j_perp}_j={j}_t={t}.npy"

    print(f"Loading files from: {results_dir.resolve()}\n")

    if not file_t.exists():
        print(f"❌ File not found:\n   {file_t}")
        print("\nMake sure you ran 'SU3_calc_sc_cc_overlaps_tri_grid.py' to generate the data first!")
    else:
        Ms_t2_0, Ms_t2_1 = np.load(file_t)
        Ms_j_0, Ms_j_1 = np.load(file_j)

        # 1. Individual degenerate channels
        test_C3_symmetry(k_grid, Ms_t2_0, label="M_{t'}_0")
        test_C3_symmetry(k_grid, Ms_t2_1, label="M_{t'}_1")
        
        # 2. Basis-independent invariant quantity (|M0|^2 + |M1|^2)
        M_t_total = np.abs(Ms_t2_0)**2 + np.abs(Ms_t2_1)**2
        test_C3_symmetry(k_grid, M_t_total, label="|M_{t'}_0|^2 + |M_{t'}_1|^2 (Invariant Norm)")

        test_C3_symmetry(k_grid, Ms_j_0, label="M_{J_perp}_0")
        test_C3_symmetry(k_grid, Ms_j_1, label="M_{J_perp}_1")
        
        M_j_total = np.abs(Ms_j_0)**2 + np.abs(Ms_j_1)**2
        test_C3_symmetry(k_grid, M_j_total, label="|M_{J_perp}_0|^2 + |M_{J_perp}_1|^2 (Invariant Norm)")