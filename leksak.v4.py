
import re
import pandas as pd
import numpy as np
import fiona
from shapely.geometry import LineString, Point
from scipy.spatial.distance import cdist, pdist
from shapely.ops import unary_union
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.validation import make_valid
from sklearn.neighbors import KDTree
from shapely.prepared import prep
import argparse
import logging
import scipy.sparse as sp
from scipy.spatial.distance import pdist
from math import pi, log

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------------
# Helper Functions from cfgstats.py
# --------------------------------------------------

def compute_lambda(freq_ghz):
    """
    Compute wavelength in meters.
    
    Parameters:
    freq_ghz (float): Frequency in GHz.
    
    Returns:
    float: Wavelength in meters.
    """
    c = 3e8  # speed of light, m/s
    freq_hz = freq_ghz * 1e9
    return c / freq_hz

def compute_required_bmax(freq_ghz, target_ar_arcsec):
    """
    Compute required Bmax for target angular resolution based on AR.
    
    Parameters:
    freq_ghz (float): Frequency in GHz.
    target_ar_arcsec (float): Target AR in arcsec.
    
    Returns:
    float: Required Bmax in meters.
    """
    lam = compute_lambda(freq_ghz)
    rad_to_arcsec = 180 / pi * 3600
    return 1.22 * lam * rad_to_arcsec / target_ar_arcsec

def compute_required_l80(freq_ghz, target_theta_arcsec):
    """
    Compute required L80 for target empirical beam FWHM.
    
    Parameters:
    freq_ghz (float): Frequency in GHz.
    target_theta_arcsec (float): Target θ_res in arcsec.
    
    Returns:
    float: Required L80 in meters.
    """
    lam = compute_lambda(freq_ghz)
    rad_to_arcsec = 180 / pi * 3600
    return 0.574 * lam * rad_to_arcsec / target_theta_arcsec

# --------------------------------------------------
# Other Helper Functions
# --------------------------------------------------

def parse_cfg(path):
    rows = []
    with open(path) as fh:
        for L in fh:
            L = L.strip()
            if not L or L.startswith('#'):
                continue
            parts = re.split(r"\s+", L)
            if len(parts) == 5:
                x, y, z, diam, name = parts
            elif len(parts) == 4:
                name, x, y, z = parts
                diam = '12.'
            else:
                continue
            rows.append({
                'name': name,
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'diam': float(diam)
            })
    return pd.DataFrame(rows)

def compute_angular_deficit(existing_pts, new_pts, n_theta=16):
    dx = existing_pts[:, None, 0] - new_pts[None, :, 0]
    dy = existing_pts[:, None, 1] - new_pts[None, :, 1]
    ang = (np.arctan2(dy, dx) % (2 * np.pi)).ravel()
    bins = np.linspace(0, 2 * np.pi, n_theta + 1)
    hist, _ = np.histogram(ang, bins=bins)
    total = hist.sum()
    target = total / n_theta
    return hist, (target - hist), bins

def plot_normalized_histogram(baselines, R_max, nbins=50,
                              title=None, filename=None,
                              alpha=0.7, figsize=(6,4)):
    u = baselines / R_max
    edges = np.linspace(0, 1, nbins + 1)

    plt.figure(figsize=figsize)
    plt.hist(u, bins=edges, density=True, alpha=alpha,
             edgecolor='k', linewidth=0.5)
    plt.xlabel('Normalized baseline length ℓ / R_max')
    plt.ylabel('Density')
    if title:
        plt.title(title)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150)

    plt.show()

def plot_angular_deficit(hist, deficit, bins, save_path=None):
    centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='polar')
    bars = ax.bar(centers, np.clip(deficit, 0, None), width=width, bottom=0, alpha=0.7)
    ax.set_title("Angular deficit (under-served bins)", va='bottom')
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Saved angular deficit plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def generate_candidates(allowed_area, spacing, R_max, inner_radius, include_reuse, region_df, favored_weight):
    prep_allowed = prep(allowed_area)
    xmin, ymin, xmax, ymax = allowed_area.bounds
    grid_pts = [(x, y) for x in np.arange(xmin, xmax, spacing)
                for y in np.arange(ymin, ymax, spacing)
                if prep_allowed.contains(Point(x, y))]
    grid_df = pd.DataFrame(grid_pts, columns=['x', 'y'])
    grid_df['orig_name'] = None
    grid_df['type'] = 'grid'
    grid_df['cand_weight'] = 1.0

    if include_reuse:
        region_cand_df = region_df.rename(columns={'name': 'orig_name'})[['x', 'y', 'orig_name']].copy()
        region_cand_df['type'] = 'region'
        region_cand_df['cand_weight'] = favored_weight
        cand_df = pd.concat([grid_df, region_cand_df], ignore_index=True)
    else:
        cand_df = grid_df

    rad = np.hypot(cand_df.x, cand_df.y)
    keep = rad <= R_max
    logging.info(f"Clipped {np.count_nonzero(~keep)} candidates beyond {R_max} m")
    cand_df = cand_df.loc[keep].reset_index(drop=True)

    cand_radials = np.hypot(cand_df.x, cand_df.y)
    cand_df['region'] = np.where(cand_radials <= inner_radius, 'inner', 'outer')

    return cand_df

def precompute_hist_ec_2d(existing_pts, cand_pts, r_bins, theta_bins, ex_weights):
    N_r = len(r_bins) - 1   
    N_theta = len(theta_bins) - 1
    N_cells = N_r * N_theta
    N_exist = existing_pts.shape[0]
    N_cand = cand_pts.shape[0]

    d_ec_raw = cdist(existing_pts, cand_pts) 
    dx_ec = existing_pts[:, None, 0] - cand_pts[None, :, 0]
    dy_ec = existing_pts[:, None, 1] - cand_pts[None, :, 1]
    phi_ec_raw = (np.arctan2(dy_ec, dx_ec) % (2 * np.pi))

    ir_ec = np.digitize(d_ec_raw, r_bins) - 1 
    it_ec = np.digitize(phi_ec_raw, theta_bins) - 1 
    valid_ec = (
        (ir_ec >= 0) & (ir_ec < N_r) &
        (it_ec >= 0) & (it_ec < N_theta)
    )
    cell_ec = ir_ec * N_theta + it_ec 

    hist_ec_2d = np.zeros((N_cand, N_cells), dtype=float)
    for j in range(N_cand):
        valid_idx = np.where(valid_ec[:, j])[0]
        cells = cell_ec[valid_idx, j]
        weights = ex_weights[valid_idx]
        hist_ec_2d[j, :] = np.bincount(cells, weights=weights, minlength=N_cells)

    return hist_ec_2d
#"C:/Users/hampu/Downloads/alma.cycle11.10s.cfg"
#r"C:\Users\hampu\OneDrive\Skrivbord\alma2040\CASA\DDP\finalCFGs\OSF.focused\alma.cycle11.10.30_OSFV1.cfg"
def main():
    parser = argparse.ArgumentParser(description="ALMA Pad Addition for Better Angular Resolution")
    parser.add_argument('--cfg_path', default=r"C:/Users/hampu/Downloads/alma.cycle11.10s.cfg", help="Path to base configuration file")
    parser.add_argument('--kml_path', default="C:/Users/hampu/OneDrive/Skrivbord/alma2040/configREAL/v5.kml")
    parser.add_argument('--poly_layer', default="Regions OSF")
    parser.add_argument('--regions_kml', default="C:/Users/hampu/OneDrive/Skrivbord/alma2040/configREAL/v5.kml")
    parser.add_argument('--out_new', default="C:/Users/hampu/new_pads77_41.cfg")
    parser.add_argument('--out_plus', default="C:/Users/hampu/full_config200_v5.cfg")
    parser.add_argument('--n_new', type=int, default=157, help="Number of new pads to add")
    parser.add_argument('--spacing', type=float, default=25.0)
    parser.add_argument('--min_bl', type=float, default=270)
    parser.add_argument('--R_max', type=float, default=27000.0)#25500
    parser.add_argument('--inner_radius', type=float, default=4700)
    parser.add_argument('--favored_weight', type=float, default=0.8) 
    parser.add_argument('--include_reuse', action='store_true')
    parser.add_argument('--cable_weight', type=float, default=5e-5)
    parser.add_argument('--inner_ratio', type=float, default=1)  # Favor outer
    parser.add_argument('--outer_ratio', type=float, default=1)
    parser.add_argument('--fixed_r_max', type=float, default=54000)  # Adjusted for extreme baselines 50000
    parser.add_argument('--ratio_penalty_weight', type=float, default=0)
    parser.add_argument('--transition_width', type=float, default=7000.0)
    parser.add_argument('--density_radius', type=float, default=1500.0) 
    parser.add_argument('--density_weight', type=float, default=0.35)
    parser.add_argument('--target_res', type=float, default=0.002, help="Target empirical resolution θ_res in arcsec")
    parser.add_argument('--freq', type=float, default=343.5, help="Frequency in GHz")
    parser.add_argument('--sigma_b', type=float, default=0.0, help="Sigma for baseline distribution peak, 0 to auto-set based on target_res")
    args = parser.parse_args()
    args.include_reuse = True

    # Compute required L80 if target_res is set
    if args.target_res > 0:
        required_l80 = compute_required_l80(args.freq, args.target_res)
        if args.sigma_b == 0.0:
            # For Rayleigh 80th percentile r80 = sigma * sqrt(-2 * ln(0.2)) ≈ sigma * 1.794
            ln02 = log(0.2)
            r80_factor = (-2 * ln02)**0.5
            args.sigma_b = required_l80 / r80_factor
        args.fixed_r_max = required_l80 * 2.0
        args.R_max = required_l80
        logging.info(f"Target θ_res {args.target_res} arcsec requires L80 ~{required_l80:.0f} m, setting sigma_b={args.sigma_b:.0f} m, fixed_r_max={args.fixed_r_max:.0f} m, R_max={args.R_max:.0f} m")

    # Load base pads
    existing_df = parse_cfg(args.cfg_path)
    existing_df['weight'] = 1.0
    existing_pts = existing_df[['x', 'y']].values
    
    base_pts = existing_pts  # Set base_pts to base existing only

    # Find max NP ID in base config
    np_pads = existing_df[existing_df['name'].str.startswith('NP')]
    if not np_pads.empty:
        np_ids = np_pads['name'].str.extract(r'NP(\d+)').astype(int)[0]
        max_np = np_ids.max()
        ctr_start = max_np + 1
        logging.info(f"Base config has NP pads up to {max_np}; starting new NPs at {ctr_start}")
    else:
        ctr_start = 1
        logging.info("Base config has no NP pads; starting new NPs at 1")

    # Setup projection
    lat0, lon0 = -23.029, -67.755000
    proj4 = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    crs_loc = CRS.from_proj4(proj4)

    # Load allowed areas
    poly_layers = ["CarlosAllowed", "extraRegions"]
    polys = []
    for layer in poly_layers:
        try:
            regions_gdf = gpd.read_file(args.kml_path, layer=layer, driver="KML").to_crs(crs_loc)
            for poly in regions_gdf.geometry:
                clean = make_valid(poly.buffer(0))
                if clean.is_empty:
                    continue
                if clean.geom_type == "GeometryCollection":
                    polys.extend([g for g in clean.geoms if g.geom_type in ("Polygon", "MultiPolygon")])
                else:
                    polys.append(clean)
        except ValueError as e:
            logging.warning(f"Failed to load layer '{layer}': {e}")
    if not polys:
        raise ValueError("No valid polygons loaded from KML layers.")
    allowed_area = unary_union(polys)

    # Load region pads
    region_layers = ["Central Cluster", "iner cluster", "W", "P", "S"]
    region_pads = []
    for layer in region_layers:
        rg = gpd.read_file(args.regions_kml, layer=layer, driver="KML").to_crs(crs_loc)
        for _, row in rg.iterrows():
            region_pads.append({
                'name': row.get('Name', layer),
                'x': row.geometry.x,
                'y': row.geometry.y,
                'z': 0.0,
                'diam': 12.0,
                'weight': args.favored_weight
            })
    region_df = pd.DataFrame(region_pads)
    logging.info(f"Loaded {len(region_df)} region pads: {region_df['name'].unique()}")

    # All existing for baselines
    if args.include_reuse:
        all_existing = pd.concat([existing_df, region_df], ignore_index=True)
    else:
        all_existing = existing_df.copy()
    existing_pts = all_existing[['x', 'y']].values
    ex_weights = all_existing['weight'].values

    # Candidates
    cand_df = generate_candidates(allowed_area, args.spacing, args.R_max, args.inner_radius, args.include_reuse, region_df, args.favored_weight)
    print(f"Total candidates: {len(cand_df)}")
    print(f"Region candidates: {np.sum(cand_df['type'] == 'region')}")
    print(f"Inner candidates: {np.sum(cand_df['region'] == 'inner')}")

    cand_pts = cand_df[['x', 'y']].values
    cand_region = cand_df['region'].values
    cand_rad = np.hypot(cand_pts[:, 0], cand_pts[:, 1])
    radial_boost = np.ones(len(cand_pts))
    mask_inner = cand_rad < 2000
    radial_boost[mask_inner] = 1.2  
    mask_mid = (cand_rad > 6000) & (cand_rad < 8800)
    radial_boost[mask_mid] = 0.7
    mask_outer = cand_rad >8800
    radial_boost[mask_outer] = 0.7 

    # Precomputes
    tree = KDTree(cand_pts)
    neighbors = tree.query_radius(cand_pts, r=args.min_bl)
    allowed = np.ones(len(cand_pts), dtype=bool)

    d_base = cdist(base_pts, cand_pts)

    N_r = 24
    r_bins = np.linspace(0, args.fixed_r_max, N_r + 1)
    N_theta = 36
    theta_bins = np.linspace(0, 2 * np.pi, N_theta + 1)
    N_cells = N_r * N_theta

    hist_ec_2d = precompute_hist_ec_2d(
        existing_pts,
        cand_pts,
        r_bins,
        theta_bins,
        ex_weights
    )

    # Compute directed EE histogram
    N_exist = existing_pts.shape[0]
    dx_ee = existing_pts[:, None, 0] - existing_pts[None, :, 0]
    dy_ee = existing_pts[:, None, 1] - existing_pts[None, :, 1]
    d_ee = np.sqrt(dx_ee**2 + dy_ee**2)
    phi_ee = (np.arctan2(dy_ee, dx_ee) % (2 * np.pi))
    ir_ee = np.digitize(d_ee, r_bins) - 1
    it_ee = np.digitize(phi_ee, theta_bins) - 1
    valid_ee = (ir_ee >= 0) & (ir_ee < N_r) & (it_ee >= 0) & (it_ee < N_theta) & (~np.eye(N_exist, dtype=bool))
    row_ee, col_ee = np.where(valid_ee)
    cells_ee = ir_ee[valid_ee] * N_theta + it_ee[valid_ee]
    weights_ee = ex_weights[row_ee]
    ee_hist = np.bincount(cells_ee, weights=weights_ee, minlength=N_cells)
    ee_hist = ee_hist / 2.0
    ee_weighted = np.sum(ee_hist)
    cable_dists = np.linalg.norm(cand_pts - existing_pts[0], axis=1) if len(existing_pts) > 0 else np.zeros(len(cand_pts))

    # Initial angular
    hist_ang, def_ang, ang_bins = compute_angular_deficit(existing_pts, cand_pts, N_theta)
    plot_angular_deficit(hist_ang, def_ang, ang_bins, save_path="initial_angular_deficit.png")

    # Greedy optimization
    sel = []
    var_nn_2d = np.zeros((len(cand_pts), N_cells), dtype=float)
    curr_ec_2d = ee_hist.copy()
    curr_nn_2d = np.zeros(N_cells, dtype=float)
    total_ratio = args.inner_ratio + args.outer_ratio

    for i in range(args.n_new):
        count_in = sum(1 for j in sel if cand_df.iloc[j]['region'] == 'inner')
        count_out = len(sel) - count_in
        n_sel = len(sel) + 1
        want_in = int(np.floor(n_sel * args.inner_ratio / total_ratio))
        want_out = n_sel - want_in

        total_base_new = len(existing_pts) * n_sel
        total_new_new = n_sel * (n_sel - 1) // 2
        total_counts = total_base_new + total_new_new

        sigma_b = args.sigma_b
        base_target = np.zeros(N_cells)
        for r in range(N_r):
            bin_center = (r_bins[r] + r_bins[r+1]) / 2
            factor = (bin_center / sigma_b**2) * np.exp(-bin_center**2 / (2 * sigma_b**2))
            base_target[r*N_theta:(r+1)*N_theta] = factor
        taper_scale = 2530.0
        taper_factor = 1 - np.exp(-bin_center**2 / taper_scale**2)
        base_target[r*N_theta:(r+1)*N_theta] *= taper_factor
        if np.sum(base_target) > 0:
            target2d = (total_counts / np.sum(base_target)) * base_target
        else:
            target2d = np.full(N_cells, total_counts / N_cells)

        fixed_2d = curr_ec_2d + curr_nn_2d
        total_2d = fixed_2d[None, :] + hist_ec_2d + var_nn_2d
        err_2d = total_2d - target2d[None, :]
        J2d_all = np.sum(err_2d ** 2, axis=1)
        Jc_all = args.cable_weight * cable_dists
        Jtot_all = (J2d_all + Jc_all) * cand_df['cand_weight'].values * radial_boost

        dev_in = max(0, want_in - count_in)
        dev_out = max(0, want_out - count_out)
        scale_factor = 1 / (1 + np.exp((cand_rad - args.inner_radius) / args.transition_width))
        is_inner = (cand_region == 'inner')
        penalty = np.zeros(len(cand_pts))
        penalty[is_inner] = scale_factor[is_inner] * dev_out
        penalty[~is_inner] = scale_factor[~is_inner] * dev_in
        Jtot_all += args.ratio_penalty_weight * penalty

        all_neighbors = tree.query_radius(cand_pts, r=args.density_radius)
        neighbor_counts = np.array([len(neigh) - 1 for neigh in all_neighbors])
        Jtot_all += args.density_weight * neighbor_counts

        not_sel = ~np.isin(np.arange(len(cand_pts)), sel)
        min_bl_mask = np.min(d_base, axis=0) >= args.min_bl
        mask = allowed & not_sel & min_bl_mask

        if np.sum(mask) == 0:
            raise RuntimeError(f"No valid candidate at step {i+1}")

        masked_J = np.where(mask, Jtot_all, np.inf)
        bestj = np.argmin(masked_J)
        bestJ = masked_J[bestj]

        sel.append(bestj)
        curr_ec_2d += hist_ec_2d[bestj]
        curr_nn_2d += var_nn_2d[bestj]
        k = bestj
        dx_add = cand_pts[k, 0] - cand_pts[:, 0]
        dy_add = cand_pts[k, 1] - cand_pts[:, 1]
        d_add = np.hypot(dx_add, dy_add)
        phi_add = (np.arctan2(dy_add, dx_add) % (2 * np.pi))
        ir_add = np.digitize(d_add, r_bins) - 1
        it_add = np.digitize(phi_add, theta_bins) - 1
        valid_add = (ir_add >= 0) & (ir_add < N_r) & (it_add >= 0) & (it_add < N_theta) & (d_add > 1e-6)
        cells_add = ir_add[valid_add] * N_theta + it_add[valid_add]
        m_add = np.where(valid_add)[0]
        np.add.at(var_nn_2d, (m_add, cells_add), 1.0)
        bad = neighbors[bestj]
        allowed[bad] = False
        allowed[bestj] = True

        logging.info(f"Step {i+1}/{args.n_new}: added {bestj}, cost={bestJ:.1f}")

        if (i + 1) % 5 == 0:
            current_existing = np.vstack([existing_pts, cand_pts[sel[:-1]]]) if sel[:-1] else existing_pts
            new_this_step = cand_pts[[sel[-1]]]
            hist_ang, def_ang, ang_bins = compute_angular_deficit(current_existing, new_this_step, N_theta)
            plot_angular_deficit(hist_ang, def_ang, ang_bins, save_path=f"step_{i+1}_angular_deficit.png")

    sel_df = cand_df.iloc[sel].reset_index(drop=True)
    names = []
    ctr = ctr_start
    for orig, typ in zip(sel_df.orig_name, sel_df.type):
        if typ == 'region' and orig:
            names.append(orig)
        else:
            names.append(f"NP{ctr:03d}")
            ctr += 1
    sel_df['name'] = names
    sel_df['z'] = 0.0
    sel_df['diam'] = 12.0
    new_df = sel_df[['name', 'x', 'y', 'z', 'diam']]

    reused = sel_df[sel_df.type == 'region']['name'].tolist()
    if reused:
        logging.info(f"Reused {len(reused)} region pads: {reused}")
    else:
        logging.info("No region pads reused.")

    header = [
        "# observatory=ALMA",
        "# coordsys=LOC (local tangent plane)",
        "# x y z diam pad#"
    ]
    with open(args.out_new, 'w') as fh:
        fh.write("\n".join(header) + "\n")
        for _, row in new_df.iterrows():
            fh.write(f"{row.x:.3f} {row.y:.3f} {row.z:.3f} {row.diam:.1f} {row['name']}\n")
    logging.info(f"Wrote new pads to {args.out_new}")

    with open(args.out_plus, 'w') as fh:
        fh.write("\n".join(header) + "\n")
        for _, row in existing_df.iterrows():
            fh.write(f"{row.x:.3f} {row.y:.3f} {row.z:.3f} {row.diam:.1f} {row['name']}\n")
        for _, row in new_df.iterrows():
            fh.write(f"{row.x:.3f} {row.y:.3f} {row.z:.3f} {row.diam:.1f} {row['name']}\n")
    logging.info(f"Wrote full config to {args.out_plus}")

    all_pads = pd.concat([existing_df[['x', 'y']], new_df[['x', 'y']]], ignore_index=True)
    n_exist = len(existing_df)
    exist_plot = all_pads.iloc[:n_exist]
    new_plot = all_pads.iloc[n_exist:]

    is_reused = sel_df['type'] == 'region'
    reused_plot = new_df[is_reused]
    grid_new_plot = new_df[~is_reused]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(exist_plot.x, exist_plot.y, c='C0', label=f'Existing ({n_exist})', s=20, alpha=0.8)
    ax.scatter(grid_new_plot.x, grid_new_plot.y, c='C1', label=f'New grid ({len(grid_new_plot)})', s=20, alpha=0.8)
    ax.scatter(reused_plot.x, reused_plot.y, c='C2', label=f'Reused ({len(reused_plot)})', s=20, alpha=0.8)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("ALMA Pads: Existing vs. New")
    ax.legend()
    plt.tight_layout()
    plt.savefig("pads_scatter.png")
    plt.show()

    bl_bins = np.linspace(0, args.fixed_r_max, 65)
    bl = pdist(all_pads[['x', 'y']].values)

    Rmax = args.fixed_r_max

    plot_normalized_histogram(
        baselines=bl,
        R_max=Rmax,
        nbins=65,
        title=f"Norm. Baseline distribution",
        filename="baseline_norm_hist.png"
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(bl, bins=bl_bins, edgecolor='black')
    ax.set_xlabel("Baseline (m)")
    ax.set_ylabel("Count")
    ax.set_title(f"Baseline Distribution ({len(all_pads)} pads)")
    plt.tight_layout()
    plt.savefig("baseline_hist.png")
    plt.show()

    edges = [0, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000, 24000]
    labels = [f"{edges[k]}–{edges[k+1]} m" for k in range(len(edges)-1)]
    rads = np.hypot(all_pads.x, all_pads.y)
    all_pads['ring'] = pd.cut(rads, bins=edges, labels=labels, right=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    for lbl, grp in all_pads.groupby('ring'):
        ax.scatter(grp.x, grp.y, label=f"{lbl} ({len(grp)})", alpha=0.8)
    for R in edges[1:]:
        circle = plt.Circle((0, 0), R, fill=False, ls='--', color='gray', alpha=0.5)
        ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Pad Locations by Radial Bin")
    ax.legend(loc='upper right', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig("radial_scatter.png")
    plt.show()

if __name__ == "__main__":
    main()