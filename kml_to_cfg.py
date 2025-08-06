import re
import pandas as pd
import geopandas as gpd
import numpy as np
import requests
import argparse
import logging
import fiona
from pathlib import Path
from pyproj import CRS, Transformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_cfg(path):
    path = str(Path(path).resolve())  # Normalize path
    rows = []
    try:
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
    except FileNotFoundError:
        logging.error(f"Config file not found: {path}")
        raise
    return pd.DataFrame(rows)

def get_elevations(lats, lons, default_elev=5000):
    if len(lats) == 0:
        return []
    locations_str = '|'.join(f"{lat},{lon}" for lat, lon in zip(lats, lons))
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations_str}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        elevations = [result['elevation'] for result in data['results']]
        return elevations
    except requests.RequestException as e:
        logging.warning(f"Open Elevation API failed: {e}. Using default elevation {default_elev} m")
        return [default_elev] * len(lats)

def generate_cfg_from_kml(base_cfg_path, kml_path, folder_name, output_path, lat0=-23.029, lon0=-67.755):
    """
    Generate an ALMA config file by merging a base .cfg with KML points (converted to NP pads).
    
    Parameters:
        base_cfg_path (str): Path to the base .cfg file
        kml_path (str): Path to the KML file
        folder_name (str): Name of the folder/layer in the KML file
        output_path (str): Path to write the new combined .cfg file
        lat0, lon0 (float): Center lat/lon for ALMA local projection
    """
    # Normalize paths
    base_cfg_path = str(Path(base_cfg_path).resolve())
    kml_path = str(Path(kml_path).resolve())
    output_path = str(Path(output_path).resolve())

    existing_df = parse_cfg(base_cfg_path)

    np_pads = existing_df[existing_df['name'].str.startswith('NP')]
    if not np_pads.empty:
        np_ids = np_pads['name'].str.extract(r'NP(\d+)').astype(int)[0]
        ctr_start = np_ids.max() + 1
        logging.info(f"Starting new NP pads at NP{ctr_start:03d}")
    else:
        ctr_start = 1
        logging.info("No existing NP pads; starting at NP001")

    # Set local projection
    proj4 = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    crs_loc = CRS.from_proj4(proj4)

    # Read KML layer
    try:
        available_layers = fiona.listlayers(kml_path)
        if folder_name not in available_layers:
            raise ValueError(f"Layer '{folder_name}' not found in KML. Available layers: {available_layers}")
        gdf = gpd.read_file(kml_path, layer=folder_name, driver="KML")
    except ValueError as e:
        logging.error(f"Could not read KML layer '{folder_name}': {e}")
        raise

    if gdf.empty:
        logging.warning(f"No markers found in folder '{folder_name}'")
        return

    # Ensure CRS is WGS84 (lat/lon)
    gdf = gdf.set_crs('epsg:4326', inplace=True)

    # Query elevation for center reference point
    elev_center = get_elevations([lat0], [lon0], default_elev=5000)
    if not elev_center:
        raise ValueError("Failed to retrieve reference center elevation")
    elev_center = elev_center[0]
    logging.info(f"Reference center elevation: {elev_center} m")

    # Query elevations for KML points
    lats = gdf.geometry.y.values
    lons = gdf.geometry.x.values
    elevations = get_elevations(lats, lons, default_elev=5000)

    # Compute relative elevations
    relative_elev = [elev - elev_center for elev in elevations]

    # Full 3D transformation: WGS84 (lat/lon/alt) → AEQD (x/y/z)
    transformer = Transformer.from_crs("epsg:4326", crs_loc, always_xy=True)
    xyz = np.array([transformer.transform(lon, lat, rel_elev) for lon, lat, rel_elev in zip(lons, lats, relative_elev)])
    xs, ys, zs = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    new_df = pd.DataFrame({
        'name': [f"NP{ctr_start + i:03d}" for i in range(len(xs))],
        'x': xs,
        'y': ys,
        'z': zs,
        'diam': 12.0
    })

    logging.info(f"Adding {len(new_df)} new NP pads with relative z values")

    # Write output file
    header = [
        "# observatory=ALMA",
        "# coordsys=LOC (local tangent plane)",
        "# x y z diam pad#"
    ]
    combined_df = pd.concat([existing_df, new_df])
    try:
        with open(output_path, 'w') as fh:
            fh.write("\n".join(header) + "\n")
            for _, row in combined_df.iterrows():
                fh.write(f"{row.x:.3f} {row.y:.3f} {row.z:.3f} {row.diam:.1f} {row['name']}\n")
    except IOError as e:
        logging.error(f"Failed to write output file {output_path}: {e}")
        raise

    logging.info(f"Wrote config to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge KML points into ALMA config")
    parser.add_argument('--base_cfg', required=True, help="Path to base .cfg file")
    parser.add_argument('--kml_path', required=True, help="Path to .kml file")
    parser.add_argument('--folder_name', required=True, help="Name of folder in KML")
    parser.add_argument('--out_cfg', required=True, help="Output .cfg file path")
    args = parser.parse_args()

    generate_cfg_from_kml(
        base_cfg_path=args.base_cfg,
        kml_path=args.kml_path,
        folder_name=args.folder_name,
        output_path=args.out_cfg
    )

if __name__ == "__main__":
    main()