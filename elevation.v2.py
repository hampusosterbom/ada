import re
import time
import requests
import pandas as pd
import numpy as np
from pyproj import CRS, Transformer

# ----------------------------------------------------------------------------- 
# File-paths
# ----------------------------------------------------------------------------- 
INPUT_CFG = r"C:\Users\hampu\full_config200_v5.cfg"#"C:\Users\hampu\full_config200.cfg"#"C:/Users/hampu/full_config1.cfg"#"C:/Users/hampu/Downloads/alma2040_300.cfg"  # input .cfg file
OUTPUT_CFG = "C:/Users/hampu/OneDrive/Skrivbord/alma2040/alma.cycle200_v5.cfg"  # output file with corrected z-values

# ALMA reference lat/lon (to compute z-offsets)
LAT0, LON0 = -23.029, -67.755000

# ----------------------------------------------------------------------------- 
# Helpers to parse/write CASA .cfg
# ----------------------------------------------------------------------------- 
def parse_cfg(path):
    """
    Parse a CASA .cfg file into a DataFrame with x, y, z, diam, pad columns.
    Handles lines with 5 columns (x, y, z, diam, pad).
    """
    rows = []
    with open(path) as fh:
        for L in fh:
            L = L.strip()
            if not L or L.startswith('#'):
                continue
            x, y, z, diam, pad = re.split(r'\s+', L)
            rows.append({'x': float(x), 'y': float(y), 'z': float(z),
                         'diam': float(diam), 'pad': pad})
    return pd.DataFrame(rows)

def write_cfg(path, df):
    """
    Write a CASA-style .cfg with header and five columns: x y z diam pad#
    Access fields via row['…'] to avoid attribute collisions.
    """
    with open(path, 'w') as fh:
        fh.write("# observatory=ALMA\n")
        fh.write("# coordsys=LOC (local tangent plane)\n")
        fh.write("# x y z diam pad#\n")
        for _, row in df.iterrows():
            fh.write(
                f"{row['x']:.6f} "
                f"{row['y']:.6f} "
                f"{row['z']:.6f} "
                f"{row['diam']:.1f} "
                f"{row['pad']}\n"
            )

# ----------------------------------------------------------------------------- 
# Load the input configuration
# ----------------------------------------------------------------------------- 
df = parse_cfg(INPUT_CFG)  # Read the single .cfg file

# ----------------------------------------------------------------------------- 
# Build reverse transformer (LOC → lat/lon)
# ----------------------------------------------------------------------------- 
proj4 = (f"+proj=aeqd +lat_0={LAT0} +lon_0={LON0} "
         "+datum=WGS84 +units=m +no_defs")
crs_loc = CRS.from_proj4(proj4)
rev_tf = Transformer.from_crs(crs_loc, "EPSG:4326", always_xy=True)

# ----------------------------------------------------------------------------- 
# Get reference elevation via API
# ----------------------------------------------------------------------------- 
def get_elev(lat, lon):
    """
    Fetch elevation for a given latitude and longitude using Open Elevation API.
    Returns NaN if the request fails or no elevation is found.
    """
    url = "https://api.open-elevation.com/api/v1/lookup"
    resp = requests.get(url, params={"locations": f"{lat},{lon}"})
    try:
        data = resp.json()
    except ValueError:
        return np.nan
    if "results" in data and data["results"]:
        return data["results"][0].get("elevation", np.nan)
    return np.nan

# Fetch array-center elevation
ref_lat, ref_lon = LAT0, LON0
ref_elev = get_elev(ref_lat, ref_lon)
print(f"Reference elevation at ({ref_lat}, {ref_lon}): {ref_elev} m")

# ----------------------------------------------------------------------------- 
# Reverse-project & fetch each pad’s elevation
# ----------------------------------------------------------------------------- 
lats, lons, elevs = [], [], []
for x, y in zip(df.x, df.y):
    lon, lat = rev_tf.transform(x, y)
    lons.append(lon)
    lats.append(lat)
    # Fetch elevation and pause to avoid rate-limiting
    elev = get_elev(lat, lon)
    elevs.append(elev)
    time.sleep(0.3)  # Adjust if API rate limits are stricter

# Add lat, lon, and elevation to DataFrame
df['lon'] = lons
df['lat'] = lats
df['elev_geoid'] = elevs

# Compute z-offset relative to array center
df['z'] = df['elev_geoid'] - ref_elev

# Check for any NaN elevations
if df['elev_geoid'].isna().any():
    print("Warning: Some pads have missing elevation data. Check API connectivity or coordinates.")
    print(df[df['elev_geoid'].isna()][['pad', 'x', 'y', 'lat', 'lon']])

# ----------------------------------------------------------------------------- 
# Write out the updated configuration
# ----------------------------------------------------------------------------- 
write_cfg(OUTPUT_CFG, df)
print(f"Wrote configuration with corrected z-values to {OUTPUT_CFG}")