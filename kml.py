import re
import pandas as pd
import geopandas as gpd
import numpy as np
import requests
import argparse
import logging
from pyproj import CRS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def get_elevations(lats, lons):
    if len(lats) == 0:
        return []
    locations_str = '|'.join(f"{lat},{lon}" for lat, lon in zip(lats, lons))
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations_str}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        elevations = [result['elevation'] for result in data['results']]
        return elevations
    except requests.RequestException as e:
        logging.error(f"Error querying Open Elevation API: {e}")
        raise
#"C:\Users\hampu\OneDrive\Skrivbord\alma2040\CASA\realgg\C73-4(2)\alma.cycle73.cfg"
#r"C:\Users\hampu\OneDrive\Skrivbord\alma2040\CASA\DDP\finalCFGs\OSF.focused\alma.cycle11.10.30_OSFV1.cfg"
def main():
    parser = argparse.ArgumentParser(description="Extract markers from KML and add to ALMA config with elevations")
    parser.add_argument('--base_cfg', default=r"C:\Users\hampu\OneDrive\Skrivbord\alma2040\CASA\realgg\alma.cycle73.cfg",help="Path to base .cfg file")
    parser.add_argument('--kml_path', default=r"C:\Users\hampu\Downloads\ALMA pads.kml", help="Path to .kml file")
    parser.add_argument('--folder_name', default="test", help="Name of the folder (layer) in KML containing markers")
    parser.add_argument('--out_cfg', default="C80.cfg", help="Path to output combined .cfg file")
    args = parser.parse_args()

    # Load existing pads from base config
    existing_df = parse_cfg(args.base_cfg)

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

    # Setup ALMA local projection
    lat0, lon0 = -23.029, -67.755
    proj4 = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    crs_loc = CRS.from_proj4(proj4)

    # Read KML layer (folder) with markers (points)
    try:
        gdf = gpd.read_file(args.kml_path, layer=args.folder_name, driver="KML")
    except ValueError as e:
        logging.error(f"Failed to load layer '{args.folder_name}': {e}")
        return

    if gdf.empty:
        logging.warning(f"No markers found in folder '{args.folder_name}'")
        return

    # Ensure CRS is WGS84 (lat/lon)
    gdf = gdf.set_crs('epsg:4326', inplace=True)

    # Extract lat/lon for elevation query
    lats = gdf.geometry.y.values
    lons = gdf.geometry.x.values

    elevations = get_elevations(lats, lons)

    # Get reference elevation at array center (lat0, lon0)
    ref_elev = get_elevations([lat0], [lon0])[0]

    # Subtract reference to get LOC-compatible Z
    relative_z = [z - ref_elev for z in elevations]
    gdf['z'] = relative_z

    # Transform to local ALMA CRS
    gdf = gdf.to_crs(crs_loc)

    # Create new pads DataFrame
    n_new = len(gdf)
    names = [f"NP{ctr_start + i:03d}" for i in range(n_new)]
    new_df = pd.DataFrame({
        'name': names,
        'x': gdf.geometry.x,
        'y': gdf.geometry.y,
        'z': gdf['z'],
        'diam': 12.0
    })

    logging.info(f"Added {n_new} new NP pads with elevations from API")

    # Write combined config
    header = [
        "# observatory=ALMA",
        "# coordsys=LOC (local tangent plane)",
        "# x y z diam pad#"
    ]
    with open(args.out_cfg, 'w') as fh:
        fh.write("\n".join(header) + "\n")
        for _, row in existing_df.iterrows():
            fh.write(f"{row.x:.3f} {row.y:.3f} {row.z:.3f} {row.diam:.1f} {row['name']}\n")
        for _, row in new_df.iterrows():
            fh.write(f"{row.x:.3f} {row.y:.3f} {row.z:.3f} {row.diam:.1f} {row['name']}\n")
    logging.info(f"Wrote combined config to {args.out_cfg}")

if __name__ == "__main__":
    main()