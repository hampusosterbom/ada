import re
import pandas as pd
import numpy as np
from shapely.geometry import Point
from scipy.spatial.distance import cdist, pdist
from shapely.ops import unary_union
from pyproj import CRS
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.validation import make_valid
from sklearn.neighbors import KDTree
from shapely.prepared import prep
import argparse
import logging

# Configure logging for informative output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper Functions: Parse config files, generate candidate pads, and compute histograms for optimization

def parse_cfg(path):
    """Parse ALMA config file into DataFrame with pad coordinates and properties."""
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

def generate_candidates(allowed_area, spacing, R_max, inner_radius, include_reuse, region_df, favored_weight):
    """Generate candidate pad locations within allowed areas."""
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
    """Precompute 2D histogram of baseline distances and angles from existing to candidate pads."""
    N_r = len(r_bins) - 1
    N_theta = len(theta_bins) - 1
    N_cells = N_r * N_theta
    N_exist, N_cand = existing_pts.shape[0], cand_pts.shape[0]

    d_ec_raw = cdist(existing_pts, cand_pts)
    dx_ec = existing_pts[:, None, 0] - cand_pts[None, :, 0]
    dy_ec = existing_pts[:, None, 1] - cand_pts[None, :, 1]
    phi_ec_raw = (np.arctan2(dy_ec, dx_ec) % (2 * np.pi))

    ir_ec = np.digitize(d_ec_raw, r_bins) - 1
    it_ec = np.digitize(phi_ec_raw, theta_bins) - 1
    valid_ec = (ir_ec >= 0) & (ir_ec < N_r) & (it_ec >= 0) & (it_ec < N_theta)
    cell_ec = ir_ec * N_theta + it_ec

    hist_ec_2d = np.zeros((N_cand, N_cells), dtype=float)
    for j in range(N_cand):
        valid_idx = np.where(valid_ec[:, j])[0]
        cells = cell_ec[valid_idx, j]
        weights = ex_weights[valid_idx]
        hist_ec_2d[j, :] = np.bincount(cells, weights=weights, minlength=N_cells)

    return hist_ec_2d

# Data Loading: Load existing pads, region data, and setup coordinate projection

def load_data(args):
    """Load existing pads, region data, and setup coordinate projection system."""
    existing_df = parse_cfg(args.cfg_path)
    existing_df['weight'] = 1.0
    existing_pts = existing_df[['x', 'y']].values

    # Find starting NP counter for new pad names
    np_pads = existing_df[existing_df['name'].str.startswith('NP')]
    if not np_pads.empty:
        np_ids = np_pads['name'].str.extract(r'NP(\d+)').astype(int)[0]
        ctr_start = np_ids.max() + 1
    else:
        ctr_start = 1

    # Setup azimuthal equidistant projection centered at ALMA
    lat0, lon0 = -23.029, -67.755000
    proj4 = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    crs_loc = CRS.from_proj4(proj4)

    # Load allowed areas from multiple KML layers (e.g., CarlosAllowed, extraRegions)
    polys = []
    for layer in args.poly_layers:
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
    logging.info(f"Loaded {len(polys)} polygons from layers: {args.poly_layers}")

    # Load region pads from KML
    region_layers = ["Central Cluster", "iner cluster", "W", "P", "S"]
    region_pads = []
    for layer in region_layers:
        try:
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
        except ValueError as e:
            logging.warning(f"Failed to load region layer '{layer}': {e}")
    region_df = pd.DataFrame(region_pads)
    logging.info(f"Loaded {len(region_df)} region pads: {region_df['name'].unique()}")

    # Combine existing and region pads if reuse is enabled
    if args.include_reuse:
        all_existing = pd.concat([existing_df, region_df], ignore_index=True)
    else:
        all_existing = existing_df.copy()
    existing_pts = all_existing[['x', 'y']].values
    ex_weights = all_existing['weight'].values if 'weight' in all_existing else np.ones(len(all_existing))

    return existing_df, existing_pts, ex_weights, ctr_start, allowed_area, region_df

# Optimization: Perform greedy optimization to select new pad locations for optimal baseline distribution

def optimize_pads(args, existing_df, existing_pts, ex_weights, ctr_start, allowed_area, region_df):
    """Select new pad locations to optimize baseline-length distribution using a greedy algorithm."""
    cand_df = generate_candidates(allowed_area, args.spacing, args.R_max, args.inner_radius, 
                                 args.include_reuse, region_df, args.favored_weight)
    logging.info(f"Total candidates: {len(cand_df)} | Region: {np.sum(cand_df['type'] == 'region')} | "
                 f"Inner: {np.sum(cand_df['region'] == 'inner')}")

    cand_pts = cand_df[['x', 'y']].values
    cand_region = cand_df['region'].values
    cand_rad = np.hypot(cand_pts[:, 0], cand_pts[:, 1])
    radial_boost = np.ones(len(cand_pts))
    mask_inner = cand_rad < 2000
    radial_boost[mask_inner] = 1
    mask_mid1 = (cand_rad > 1500) & (cand_rad < 2500)
    radial_boost[mask_mid1] = 0.5
    mask_mid = (cand_rad > 4000) & (cand_rad < 6000)
    radial_boost[mask_mid] = 0.5
    mask_outer = cand_rad > 6500
    radial_boost[mask_outer] = 0.2

    tree = KDTree(cand_pts)
    neighbors = tree.query_radius(cand_pts, r=args.min_bl)
    allowed = np.ones(len(cand_pts), dtype=bool)

    base_pts = existing_df[['x', 'y']].values
    d_base = cdist(base_pts, cand_pts)

    N_r = 24
    r_bins = np.linspace(0, args.fixed_r_max, N_r + 1)
    N_theta = 36
    theta_bins = np.linspace(0, 2 * np.pi, N_theta + 1)
    N_cells = N_r * N_theta

    hist_ec_2d = precompute_hist_ec_2d(existing_pts, cand_pts, r_bins, theta_bins, ex_weights)

    # Compute existing-to-existing baseline histogram
    N_exist = existing_pts.shape[0]
    dx_ee = existing_pts[:, None, 0] - existing_pts[None, :, 0]
    dy_ee = existing_pts[:, None, 1] - existing_pts[None, :, 1]
    d_ee = np.sqrt(dx_ee**2 + dy_ee**2)
    phi_ee = (np.arctan2(dy_ee, dx_ee) % (2 * np.pi))
    ir_ee = np.digitize(d_ee, r_bins) - 1
    it_ee = np.digitize(phi_ee, theta_bins) - 1
    valid_ee = (ir_ee >= 0) & (ir_ee < N_r) & (it_ee >= 0) & (it_ee < N_theta) & (~np.eye(N_exist, dtype=bool))
    cells_ee = ir_ee[valid_ee] * N_theta + it_ee[valid_ee]
    row_ee = np.where(valid_ee)[0]
    weights_ee = ex_weights[row_ee]
    ee_hist = np.bincount(cells_ee, weights=weights_ee, minlength=N_cells) / 2.0

    cable_dists = np.linalg.norm(cand_pts - existing_pts[0], axis=1)

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

        base_target = np.zeros(N_cells)
        for r in range(N_r):
            bin_center = (r_bins[r] + r_bins[r+1]) / 2
            factor = (bin_center / args.sigma_b**2) * np.exp(-bin_center**2 / (2 * args.sigma_b**2))
            taper_scale = 2030.0
            taper_factor = 1 - np.exp(-bin_center**2 / taper_scale**2)
            base_target[r*N_theta:(r+1)*N_theta] = factor * taper_factor
        target2d = (total_counts / np.sum(base_target)) * base_target

        fixed_2d = curr_ec_2d + curr_nn_2d
        total_2d = fixed_2d[None, :] + hist_ec_2d + var_nn_2d
        err_2d = total_2d - target2d[None, :]
        J2d_all = np.sum(err_2d ** 2, axis=1)
        Jc_all = args.cable_weight * cable_dists
        Jtot_all = (J2d_all + Jc_all) * cand_df['cand_weight'].values * radial_boost

        # Apply soft ratio penalty for inner/outer balance
        dev_in = max(0, want_in - count_in)
        dev_out = max(0, want_out - count_out)
        scale_factor = 1 / (1 + np.exp((cand_rad - args.inner_radius) / args.transition_width))
        is_inner = (cand_region == 'inner')
        penalty = np.zeros(len(cand_pts))
        penalty[is_inner] = scale_factor[is_inner] * (dev_out if dev_in <= 0 else 0)
        penalty[~is_inner] = scale_factor[~is_inner] * (dev_in if dev_out <= 0 else 0)
        Jtot_all += args.ratio_penalty_weight * penalty

        # Apply density penalty to prevent clustering
        all_neighbors = tree.query_radius(cand_pts, r=args.density_radius)
        neighbor_counts = np.array([len(n) - 1 for n in all_neighbors])
        Jtot_all += args.density_weight * neighbor_counts

        # Apply masks for valid candidates
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

        # Update new-to-new baselines
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

    # Build DataFrame for new pads
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
    logging.info(f"Reused {len(reused)} region pads: {reused}" if reused else "No region pads reused.")

    return new_df, existing_df, fixed_r_max

# Output Writing: Generate configuration files for new and combined pad sets

def write_outputs(args, new_df, existing_df):
    """Write new and combined pad configurations to .cfg files."""
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

    with open(args.out_plus57, 'w') as fh:
        fh.write("\n".join(header) + "\n")
        for _, row in existing_df.iterrows():
            fh.write(f"{row.x:.3f} {row.y:.3f} {row.z:.3f} {row.diam:.1f} {row['name']}\n")
        for _, row in new_df.iterrows():
            fh.write(f"{row.x:.3f} {row.y:.3f} {row.z:.3f} {row.diam:.1f} {row['name']}\n")
    logging.info(f"Wrote combined config to {args.out_plus57}")

# Plotting: Generate essential visualizations for pad locations and baseline distribution

def generate_plots(new_df, existing_df, fixed_r_max):
    """Generate scatter plot of pad locations and normalized baseline histogram."""
    all_pads = pd.concat([existing_df[['x', 'y']], new_df[['x', 'y']]], ignore_index=True)
    n_exist = len(existing_df)
    exist_plot = all_pads.iloc[:n_exist]
    new_plot = all_pads.iloc[n_exist:]

    # Plot pad locations
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(exist_plot.x, exist_plot.y, c='C0', label=f'Existing ({n_exist})', s=20, alpha=0.8)
    ax.scatter(new_plot.x, new_plot.y, c='C1', label=f'New ({len(new_plot)})', s=20, alpha=0.8)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("ALMA Pads: Existing vs. New")
    ax.legend()
    plt.tight_layout()
    plt.savefig("pads_scatter.png")
    plt.show()

    # Plot normalized baseline histogram
    bl = pdist(all_pads[['x', 'y']].values)
    plt.figure(figsize=(6, 4))
    u = bl / fixed_r_max
    edges = np.linspace(0, 1, 65)
    plt.hist(u, bins=edges, density=True, alpha=0.7, edgecolor='k', linewidth=0.5)
    plt.xlabel('Normalized baseline length ℓ / R_max')
    plt.ylabel('Density')
    plt.title("Norm. Baseline distribution")
    plt.tight_layout()
    plt.savefig("baseline_norm_hist.png", dpi=150)
    plt.show()

# Main: Parse arguments and orchestrate the optimization pipeline

def main():
    """Run ALMA pad optimization to generate new configurations."""
    parser = argparse.ArgumentParser(description="ALMA Pad Optimization")
    parser.add_argument('--cfg_path', default="C:/Users/hampu/Downloads/alma.cycle11.10s.cfg")
    parser.add_argument('--kml_path', default="C:/Users/hampu/OneDrive/Skrivbord/alma2040/configREAL/v5.kml")
    parser.add_argument('--poly_layers', default="CarlosAllowed,extraRegions", 
                       help="Comma-separated list of KML layers for allowed areas")
    parser.add_argument('--regions_kml', default="C:/Users/hampu/OneDrive/Skrivbord/alma2040/configREAL/v5.kml")
    parser.add_argument('--out_new', default="C:/Users/hampu/new_pads_57.cfg")
    parser.add_argument('--out_plus57', default="C:/Users/hampu/OneDrive/Skrivbord/alma2040/regionsV2/alma.cycle11.10.30osf.gg.cfg")
    parser.add_argument('--n_new', type=int, default=30)
    parser.add_argument('--spacing', type=float, default=25.0)
    parser.add_argument('--min_bl', type=float, default=175)
    parser.add_argument('--R_max', type=float, default=25000)
    parser.add_argument('--inner_radius', type=float, default=4700)
    parser.add_argument('--favored_weight', type=float, default=30000)
    parser.add_argument('--include_reuse', action='store_true', default=False)
    parser.add_argument('--cable_weight', type=float, default=5e-4)
    parser.add_argument('--inner_ratio', type=float, default=1)
    parser.add_argument('--outer_ratio', type=float, default=1)
    parser.add_argument('--fixed_r_max', type=float, default=50000.0)
    parser.add_argument('--ratio_penalty_weight', type=float, default=0)
    parser.add_argument('--transition_width', type=float, default=5000.0)
    parser.add_argument('--density_radius', type=float, default=1700.0)
    parser.add_argument('--density_weight', type=float, default=0.35)
    parser.add_argument('--sigma_b', type=float, default=10000, help="Sigma for Gaussian target distribution")
    args = parser.parse_args()

    # Convert comma-separated poly_layers to list
    args.poly_layers = args.poly_layers.split(',')

    existing_df, existing_pts, ex_weights, ctr_start, allowed_area, region_df = load_data(args)
    new_df, existing_df, fixed_r_max = optimize_pads(
        args, existing_df, existing_pts, ex_weights, ctr_start, allowed_area, region_df
    )
    write_outputs(args, new_df, existing_df)
    generate_plots(new_df, existing_df, fixed_r_max)

if __name__ == "__main__":
    main()